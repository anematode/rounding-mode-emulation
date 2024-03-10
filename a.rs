#![allow(dead_code, unused_variables, unused_macros, unused_doc_comments)]

//!
//! This module emulates directed rounding modes for reproducing interval arithmetic calculations that
//! are done in an environment that supports directed rounding, which is useful for faster and slightly
//! more accurate interval arithmetic. For example, a WebAssembly target has no way to change the
//! rounding mode, but it can be useful for reproducibility reasons to emulate a native target. On
//! native targets, the fallback is not used and cannot be called explicitly. There are code paths
//! for SSE, AVX (for the VEX encoding), AVX512 (for embedded rounding control), and AArch64.
//!
//! The implemented functions are the basic IEEE 754 methods whose semantics actually depend on the
//! rounding mode: add, sub, mul, div, sqrt, fused multiply-add, and fused multiply-subtract.
//! They may be called directly (e.g. add_f64_rd) or passing the rounding mode as a parameter
//! (e.g. add_f64_round(.., RoundingMode::Down)). The four supported rounding modes are:
//!
//! | Mode | Enum | Example |
//! |------|------|---------|
//! | to nearest | RoundingMode::Nearest | fma_f64_rne, fms_f64_rne (others – use the operation directly) |
//! | toward +∞ | RoundingMode::Up | add_f64_ru |
//! | toward -∞ | RoundingMode::Down | add_f64_rd |
//! | toward 0 | RoundingMode::Zero | add_f64_rz |
//!
//! Behavior should be correct regarding infinities, NaN, and signed zero, with the exception of
//! sNaN propagation and the sign of NaNs. The latter differs from processor to processor and
//! therefore should not be relied on. The former may eventually be implemented. The methods have
//! been fuzzed against x86 and ARM processors. No other properties of the floating-point
//! environment are accessible or changeable.
//!
//! This module should only be used with targets which do not require floating-point emulation.
//! Otherwise, if emulation is required, for example softfloat, that emulation should be used
//! directly.
//!
//! # Example
//!
//! ```
//! fn main() {
//!     // 1.7976931348623157e308 + -5e-324 = 1.7976931348623155e308
//!     println!("{:?} + {:?} = {:?}", f64::MAX, -5e-324, fp_emu::add_f64_rd(f64::MAX, -5e-324));
//!     // 1.7976931348623157e308 + 5e-324 = inf
//!     println!("{:?} + {:?} = {:?}", f64::MAX, 5e-324, fp_emu::add_f64_ru(f64::MAX, 5e-324));
//!     // 1e-200 * 1e-200 = 5e-324
//!     println!("{:?} * {:?} = {:?}", 1e-200, 1e-200, fp_emu::mul_f64_ru(1e-200, 1e-200));
//!     // sqrt(2.0) ∈ (1.4142135623730949234300, 1.4142135623730951454746)
//!     println!("sqrt(2.0) ∈ ({:.22}, {:.22})", fp_emu::sqrt_f64_rd(2.0), fp_emu::sqrt_f64_ru(2.0));
//!     // sqrt(81.0) ∈ (9.0000000000000000000000, 9.0000000000000000000000)
//!     println!("sqrt(81.0) ∈ ({:.22}, {:.22})", fp_emu::sqrt_f64_rd(81.0), fp_emu::sqrt_f64_ru(81.0));
//!     // fma(1 + ε, 1 + ε, -1 - 2ε) = 4.930380657631324e-32 = ε^2
//!     println!("fma(1 + ε, 1 + ε, -1 - 2ε) = {:?} = ε^2",
//!              fp_emu::fma_f64_rd(1.0 + f64::EPSILON, 1.0 + f64::EPSILON, -1.0 - 2.0 * f64::EPSILON));
//! }
//! ```
//!
//! # Implementation details
//!
//! Rounded addition are done by computing the round-to-nearest value `s <- a + b`,
//! then using the following formula (see [Wikipedia](https://en.wikipedia.org/wiki/2Sum)):
//!
//!     p <- maxInMag(a, b)
//!     q <- minInMag(a, b)
//!
//!     err <- q - (s - p)
//!
//! The error is exact for all a, b. By checking the value of err, the direction to adjust the
//! result can be determined. The Wikipedia article gives a branchless version of this algorithm,
//! but it is susceptible to overflow. Subtraction is essentially implemented as a + (-b).
//!
//! Rounded multiplication is done using an algorithm often seen in double-double arithmetic. You
//! can find details [here](https://stackoverflow.com/a/14285800/13458117). FMA is mostly taken from
//! the glibc implementation, with changes to not require access to the floating-point environment.
//! Rounded division is done by first computing the division, then finding the error with the same
//! algorithm from rounded multiplication. We can do something similar with square roots.
//!

const FMA_SPLITTER: f64 = 134217729.0;
const SPLIT_DANGER: f64 = f64::MIN_POSITIVE * P2_53;
const P2_53: f64 = 9007199254740992.0;
const P2_106: f64 = P2_53 * P2_53;
const P2_N106: f64 = 1.0 / P2_106;
const P2_N53: f64 = 1.0 / P2_53;
const P2_N108: f64 = P2_N106 * 0.25;
const P2_108: f64 = P2_106 * 4.0;
const F64_TINY: f64 = 5e-324;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512dq")))]
fn get_two_sum_error(sum: f64, mut a: f64, mut b: f64) -> f64 {
    if a.abs() > b.abs() {
        let tmp = a;
        a = b;
        b = tmp;
    }

    a - (sum - b)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512dq"))]
fn get_two_sum_error(sum: f64, a: f64, b: f64) -> f64 {
    use std::arch::asm;

    let mut p: f64;
    let mut q: f64;

    /// Branchless and probably better TP
    unsafe {
        asm!(
        "vrangesd {p}, {a}, {b}, 7",
        "vrangesd {q}, {a}, {b}, 6",
        a = in(xmm_reg) a,
        b = in(xmm_reg) b,
        p = out(xmm_reg) p,
        q = out(xmm_reg) q,
        );
    }

    q - (sum - p)
}

/// Classic 2Sum algorithm summing two doubles exactly. The branchless 2Sum algorithm sometimes fails
/// for large a, b.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;

    (x, get_two_sum_error(x, a, b))
}

fn two_split_base(a: f64) -> (f64, f64) {
    let x = a * FMA_SPLITTER;
    let a_hi = x - (x - a);
    let a_lo = a - a_hi;

    (a_hi, a_lo)
}

/// Indicates how much the two_split algorithm scaled the result
#[derive(Debug, PartialEq)]
pub enum TwoSplitScale {
    /// Output is the same scaling as the input
    None,
    /// Output is 2^53 times the input
    Scale53,
    /// Output is 2^-53 times the input
    ScaleN53,
}

fn two_split(a: f64) -> (f64, f64, TwoSplitScale) {
    if a.abs() < SPLIT_DANGER {
        let res = two_split_base(a * P2_53);

        return (res.0, res.1, TwoSplitScale::Scale53);
    }

    // ~sqrt(2 ^ 1023), to avoid certain overflows with numbers whose product is very close to
    // f64::MAX
    if a.abs() > 9.480751908109177e+153 {
        let res = two_split_base(a * P2_N53);

        return (res.0, res.1, TwoSplitScale::ScaleN53);
    }

    let b = two_split_base(a);
    (b.0, b.1, TwoSplitScale::None)
}

/// Returns a pair of floating-point numbers whose exact sum is the exact value of the given
/// multiplication. Should behave exactly like the FMA solution.
fn two_prod_pedantic_fallback(mut a: f64, mut b: f64) -> (f64, f64) {
    let x = a * b;

    if x == 0.0 {
        if a == 0.0 || b == 0.0 {
            return (x, 0.0);
        }

        return (x, x);
    } else if x.is_infinite() {
        return (
            x,
            if a.is_finite() && b.is_finite() {
                f64::INFINITY.copysign(-x)
            } else {
                f64::NAN
            },
        );
    } else if x.is_nan() {
        return (x, x);
    }

    if a.abs() > b.abs() {
        let tmp = a;
        a = b;
        b = tmp;
    }

    let (mut a_hi, mut a_lo, ss1) = two_split(a);
    let (b_hi, b_lo, ss2) = two_split(b);
    let lo: f64;

    let mut sc_down = 1.0f64; // final
    let mut sc_up = 1.0f64; // intermediate

    if ss1 == TwoSplitScale::Scale53 {
        sc_down = P2_N53;
        sc_up = P2_53;
    }

    if ss1 == TwoSplitScale::ScaleN53 {
        sc_down *= P2_106;
        sc_up *= P2_N106;
    } else if ss2 == TwoSplitScale::ScaleN53 {
        sc_down *= P2_53;
        sc_up *= P2_N53;
    }

    if x.abs() < SPLIT_DANGER {
        a_hi *= P2_106;
        a_lo *= P2_106;

        sc_up *= P2_106;
        sc_down *= P2_N106;
    }

    lo = (((a_hi * b_hi - x * sc_up) + a_hi * b_lo) + a_lo * b_hi) + a_lo * b_lo;

    return (x, lo * sc_down);
}

#[cfg(not(any(all(target_arch = "x86_64", target_feature = "fma"), all(target_arch = "aarch64", target_feature = "neon")
)))]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    two_prod_pedantic_fallback(a, b)
}

#[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
pub fn two_prod(mut a: f64, b: f64) -> (f64, f64) {
    use std::arch::asm;
    let x = a * b;

    unsafe {
        asm!(
        "vfmsub213sd {a}, {b}, {x}",
        a = inout(xmm_reg) a,
        b = in(xmm_reg) b,
        x = in(xmm_reg) x
        );
    }

    (x, a)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    use std::arch::asm;
    let x = a * b;
    let mut res;

    unsafe {
        asm!(
        "fnmsub {res:d}, {a:d}, {b:d}, {x:d}",
        a = in(vreg) a,
        b = in(vreg) b,
        x = in(vreg) x,
        res = out(vreg) res,
        );
    }

    (x, res)
}


/**
 * Rounding mode and target-specific rounding mode constants.
 */
#[derive(Clone, Copy)]
pub enum RoundingMode {
    Nearest = 0,
    Up = 1,
    Down = 2,
    Zero = 3,
}

const DEFAULT_NEON_FPCR: u64 = 0x00000000;
const NEON_FPCR_RU: u64 = 0x00400000;
const NEON_FPCR_RD: u64 = 0x00800000;
const NEON_FPCR_RZ: u64 = 0x00c00000;

// Data for ldmxcsr
const X86_MXCSR_VALUES: [u32; 4] = [
    0x1f80, // RU
    0x3f80, // RD
    0x5f80, // RU
    0x7f80, // RZ
];

macro_rules! aarch64_fpcr {
    ("Up") => { NEON_FPCR_RU };
    ("Zero") => { NEON_FPCR_RZ };
    ("Down") => { NEON_FPCR_RD };
    ("Nearest") => { DEFAULT_NEON_FPCR };
}

macro_rules! get_avx512_round {
    ("Up") => { "{{ru-sae}}" };
    ("Zero") => { "{{rz-sae}}" };
    ("Down") => { "{{rd-sae}}" };
    ("Nearest") => { "{{rn-sae}}" };
}

macro_rules! get_mxcsr_offs {
    ("Up") => { 2 };
    ("Zero") => { 3 };
    ("Down") => { 1 };
    ("Nearest") => { 0 };
}

fn f64_next_up_no_special(a: f64) -> f64 {
    assert_ne!(a.to_bits(), (-0.0f64).to_bits());
    assert!(!a.is_infinite());
    assert!(!a.is_nan());

    let diff: u64 = if a >= 0.0 { 1u64 } else { u64::MAX };

    f64::from_bits(a.to_bits().wrapping_add(diff))
}

fn f64_next_down_no_special(a: f64) -> f64 {
    assert_ne!(a.to_bits(), 0u64);
    assert!(!a.is_infinite());
    assert!(!a.is_nan());

    let diff: u64 = if a > 0.0 { u64::MAX } else { 1u64 };

    f64::from_bits(a.to_bits().wrapping_add(diff))
}

/**
 * Fused multiply-add emulation.
 */
#[derive(Debug)]
struct UnpackedF64(bool, i32, u64);

fn unpack_f64(a: f64) -> UnpackedF64 {
    let b = a.to_bits();
    let exp = ((b >> 52) & 0x7ff) as i32;

    // (hi * 2^30 + lo) * 2^exp is the float's value
    UnpackedF64((b >> 63) != 0, exp, b & ((1 << 52) - 1))
}

fn pack_f64(s: UnpackedF64) -> f64 {
    assert_ne!(s.1, 0x7ff);

    f64::from_bits(((s.0 as u64) << 63) | ((s.1 as u64) << 52) | s.2)
}

const IEEE754_DOUBLE_BIAS: i32 = 1023;
const DBL_MANT_DIG: i32 = 53;

/// Direct and painful translation of the glibc implementation (sysdeps/ieee754/dbl-64/s_fma.c)
/// TODO: evaluate whether multi-precision solution is faster
fn fma_f64_fallback(mut a: f64, mut b: f64, mut c: f64, round: RoundingMode) -> f64 {
    let add = |x: f64, y: f64| -> f64 {
        add_f64_round(x, y, round)
    };

    let mul = |x: f64, y: f64| -> f64 {
        mul_f64_round(x, y, round)
    };

    fn adj_to_odd(a: f64, inexact: bool) -> f64 {
        return f64::from_bits(a.to_bits() | (inexact as u64));
    }

    let u = unpack_f64(a);
    let mut v = unpack_f64(b);
    let w = unpack_f64(c);

    let mut adjust: i32 = 0;

    if u.1 + v.1 >= 0x7ff + IEEE754_DOUBLE_BIAS - DBL_MANT_DIG || // maybe overflow
        u.1 >= 0x7ff - DBL_MANT_DIG ||  // too large to adjust for some calculations
        v.1 >= 0x7ff - DBL_MANT_DIG ||
        w.1 >= 0x7ff - DBL_MANT_DIG ||
        u.1 + v.1 <= IEEE754_DOUBLE_BIAS + DBL_MANT_DIG /* maybe subnormal multiplication */ {
        let get_m = || {
            return mul(a, b);
        };

        if w.1 == 0x7ff && u.1 != 0x7ff && v.1 != 0x7ff {
            // direction does not matter here
            return (c + a) + b;
        }

        if c == 0.0 {
            return if a == 0.0 || b == 0.0 {
                add(get_m(), c)
            } else {
                get_m()
            };
        }

        // one of a,b,c is non-finite or one of a,b is +-0
        if u.1 == 0x7ff || v.1 == 0x7ff || w.1 == 0x7ff || a == 0.0 || b == 0.0 {
            return get_m() + c;
        }

        // Must overflow
        if u.1 + v.1 > 0x7ff + IEEE754_DOUBLE_BIAS {
            return get_m();
        }

        // multiplication is less than 1/4 * tiny
        if u.1 + v.1 < IEEE754_DOUBLE_BIAS - DBL_MANT_DIG - 2 {
            let tiny = F64_TINY.copysign(a * b);

            if w.1 >= 3 {
                return add(tiny, c);
            }

            // handle very small denormal numbers
            return mul(add(c * P2_106, tiny), P2_N106);
        }

        // Possible intermediate calculation overflow, adjust values to prevent
        if u.1 + v.1 >= 0x7ff + IEEE754_DOUBLE_BIAS - DBL_MANT_DIG {
            if u.1 > v.1 {
                a *= P2_N53;
            } else {
                b *= P2_N53;
            }

            if w.1 > DBL_MANT_DIG {
                c *= P2_N53;
            }

            adjust = 1;
        } else if w.1 >= 0x7ff - DBL_MANT_DIG {
            if u.1 + v.1 <= IEEE754_DOUBLE_BIAS + 2 * DBL_MANT_DIG {
                if u.1 > v.1 {
                    a *= P2_108;
                } else {
                    b *= P2_108;
                }
            } else if u.1 > v.1 {
                if u.1 > DBL_MANT_DIG {
                    a *= P2_N53;
                }
            } else if v.1 > DBL_MANT_DIG {
                b *= P2_N53;
            }

            c *= P2_N53;

            adjust = 1;
        } else if u.1 >= 0x7ff - DBL_MANT_DIG {
            a *= P2_N53;
            b *= P2_53;
        } else if v.1 >= 0x7ff - DBL_MANT_DIG {
            b *= P2_N53;
            a *= P2_53;
        } else {
            if u.1 > v.1 {
                a *= P2_108;
            } else {
                b *= P2_108;
            }

            if w.1 <= 4 * DBL_MANT_DIG + 6 {
                c *= P2_108;

                adjust = -1;
            }
        }
    }

    // a,b,c all finite and nonzero
    if (a == 0.0 || b == 0.0) && c == 0.0 {
        return add(a * b, c);
    }

    // Two prod
    let x1 = a * FMA_SPLITTER;
    let y1 = b * FMA_SPLITTER;
    let m1 = a * b;
    let x1 = (a - x1) + x1;
    let y1 = (b - y1) + y1;
    let x2 = a - x1;
    let y2 = b - y1;
    let m2 = (((x1 * y1 - m1) + x1 * y2) + x2 * y1) + x2 * y2;

    let a1 = c + m1;
    let t1 = a1 - c;
    let t2 = a1 - t1;
    let t1 = m1 - t1;
    let t2 = c - t2;
    let a2 = t1 + t2;

    if a1 == 0.0 && m2 == 0.0 {
        return add(c, m1);
    }

    let mut res = add_f64_rz(a2, m2);
    let mut inexact = get_two_sum_error(res, a2, m2) != 0.0;

    if adjust < 0 {
        res = adj_to_odd(res, inexact);
        b = add_f64_rz(a1, res);
        inexact = get_two_sum_error(b, a1, res) != 0.0;
    }

    if adjust == 0 {
        if res.is_finite() {
            res = adj_to_odd(res, inexact);
        }

        return add(a1, res);
    } else if adjust > 0 {
        res = adj_to_odd(res, inexact);

        return mul(add(a1, res), P2_53);
    } else {
        if !inexact {
            return mul(b, P2_N108);
        }

        v = unpack_f64(b);
        if v.1 > 108 {
            return mul(add(a1, res), P2_N108);
        } else if v.1 == 108 {
            let mant: u64 = ((v.2 & 3) << 1) | (inexact as u64);
            let w = mul(pack_f64(UnpackedF64(b < 0.0, 0i32, mant)), 0.25);
            let v = mul(f64::from_bits(b.to_bits() & !3u64), P2_N108);

            return add(w, v);
        }

        b = adj_to_odd(b, inexact);
        return mul(b, P2_N108);
    }
}

macro_rules! gen_fma {
    ($x86_insn:literal, $aarch64_insn:literal, $name:ident, $round:tt, $name_fallback:ident) => {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        pub fn $name(mut a: f64, b: f64, c: f64) -> f64 {
            use std::arch::asm;
            unsafe {
                asm!(
                    concat!($x86_insn, " {a}, {b}, {c}, ", get_avx512_round!($round)),
                    a = inout(xmm_reg) a,
                    b = in(xmm_reg) b,
                    c = in(xmm_reg) c
                );
            }
            a
        }

        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f"), target_feature = "fma"))]
        pub fn $name(mut a: f64, b: f64, c: f64) -> f64 {
            use std::arch::asm;
            unsafe {
                if $round == "Nearest" {
                    // skip mxcsr change
                    asm!(
                        concat!($x86_insn, " {a}, {b}, {c}"),
                        a = inout(xmm_reg) a,
                        b = in(xmm_reg) b,
                        c = in(xmm_reg) c
                    );
                } else {
                    asm!(
                        concat!("vldmxcsr [{ROUND_BASE} + 4*", get_mxcsr_offs!($round), "]"),
                        concat!($x86_insn, " {a}, {b}, {c}"),
                        "vldmxcsr [{ROUND_BASE}]",
                        a = inout(xmm_reg) a,
                        b = in(xmm_reg) b,
                        c = in(xmm_reg) c,
                        ROUND_BASE = in(reg) X86_MXCSR_VALUES.as_ptr(),
                    );
                }
            }
            a
        }

        #[cfg(target_arch = "aarch64")]
        pub fn $name(a: f64, b: f64, c: f64) -> f64 {
            use std::arch::asm;
            let mut res;

            unsafe {
                if $round == "Nearest" {
                    asm!(
                        concat!($aarch64_insn, " {res:d}, {a:d}, {b:d}, {c:d}"),
                        a = in(vreg) a,
                        b = in(vreg) b,
                        c = in(vreg) c,
                        res = lateout(vreg) res,
                    );
                } else {
                    asm!(
                        "msr fpcr, {MODE}",
                        concat!($aarch64_insn, " {res:d}, {a:d}, {b:d}, {c:d}"),
                        "msr fpcr, {DEFAULT}",
                        MODE = in(reg) aarch64_fpcr!($round),
                        DEFAULT = in(reg) DEFAULT_NEON_FPCR,
                        a = in(vreg) a,
                        b = in(vreg) b,
                        c = in(vreg) c,
                        res = lateout(vreg) res,
                    );
                }
            }

            res
        }

        #[cfg(not(any(all(target_arch = "x86_64", target_feature = "fma"), target_arch = "aarch64")))]
        pub fn $name(a: f64, b: f64, c: f64) -> f64 {
            $name_fallback(a, b, c)
        }
    };
}

fn fma_f64_ru_fallback(a: f64, b: f64, c: f64) -> f64 {
    fma_f64_fallback(a, b, c, RoundingMode::Up)
}

fn fma_f64_rd_fallback(a: f64, b: f64, c: f64) -> f64 {
    fma_f64_fallback(a, b, c, RoundingMode::Down)
}

fn fma_f64_rz_fallback(a: f64, b: f64, c: f64) -> f64 {
    fma_f64_fallback(a, b, c, RoundingMode::Zero)
}

fn fma_f64_rne_fallback(a: f64, b: f64, c: f64) -> f64 {
    fma_f64_fallback(a, b, c, RoundingMode::Nearest)
}

gen_fma!("vfmadd213sd", "fmadd", fma_f64_ru, "Up", fma_f64_ru_fallback);
gen_fma!("vfmadd213sd", "fmadd", fma_f64_rz, "Zero", fma_f64_rz_fallback);
gen_fma!("vfmadd213sd", "fmadd", fma_f64_rd, "Down", fma_f64_rd_fallback);
gen_fma!("vfmadd213sd", "fmadd", fma_f64_rne, "Nearest", fma_f64_rne_fallback);

fn fms_f64_rd_fallback(a: f64, b: f64, c: f64) -> f64 {
    fma_f64_fallback(a, b, -c, RoundingMode::Down)
}

fn fms_f64_rz_fallback(a: f64, b: f64, c: f64) -> f64 {
    fma_f64_fallback(a, b, -c, RoundingMode::Zero)
}

fn fms_f64_ru_fallback(a: f64, b: f64, c: f64) -> f64 {
    fma_f64_fallback(a, b, -c, RoundingMode::Up)
}

fn fms_f64_rne_fallback(a: f64, b: f64, c: f64) -> f64 {
    fma_f64_fallback(a, b, -c, RoundingMode::Nearest)
}

gen_fma!("vfmsub213sd", "fnmsub", fms_f64_ru, "Up", fms_f64_ru_fallback);
gen_fma!("vfmsub213sd", "fnmsub", fms_f64_rz, "Zero", fms_f64_rz_fallback);
gen_fma!("vfmsub213sd", "fnmsub", fms_f64_rd, "Down", fms_f64_rd_fallback);
gen_fma!("vfmsub213sd", "fnmsub", fms_f64_rne, "Nearest", fma_f64_rne_fallback);


/// Directionally rounded fused multiply-add.
///
/// # Example
/// ```
/// // fma(1 + ε, 1 + ε, -1 - 2ε) = 4.930380657631324e-32 = ε^2
/// println!("fma(1 + ε, 1 + ε, -1 - 2ε) = {:?} = ε^2",
///     fp_emu::fma_f64_rd(1.0 + f64::EPSILON, 1.0 + f64::EPSILON, -1.0 - 2.0 * f64::EPSILON));
pub fn fma_f64_round(a: f64, b: f64, c: f64, rounding_mode: RoundingMode) -> f64 {
    match rounding_mode {
        RoundingMode::Up => { fma_f64_ru(a, b, c) }
        RoundingMode::Down => { fma_f64_rd(a, b, c) }
        RoundingMode::Zero => { fma_f64_rz(a, b, c) }
        _ => { fma_f64_rne(a, b, c) }
    }
}

/// Directionally rounded fused multiply-subtract.
///
/// # Example
/// ```
/// // fma(1 + ε, 1 + ε, 1 + 2ε) = 4.930380657631324e-32 = ε^2
/// println!("fma(1 + ε, 1 + ε, 1 + 2ε) = {:?} = ε^2",
///     fp_emu::fma_f64_rd(1.0 + f64::EPSILON, 1.0 + f64::EPSILON, 1.0 + 2.0 * f64::EPSILON));
pub fn fms_f64_round(a: f64, b: f64, c: f64, rounding_mode: RoundingMode) -> f64 {
    match rounding_mode {
        RoundingMode::Up => { fms_f64_ru(a, b, c) }
        RoundingMode::Down => { fms_f64_rd(a, b, c) }
        RoundingMode::Zero => { fms_f64_rz(a, b, c) }
        _ => { fms_f64_rne(a, b, c) }
    }
}

/**
 * add/sub/mul/div emulation.
 */

fn generic_ru_clamp_inf(res: f64, a: f64, b: f64) -> f64 {
    assert!(res.is_infinite());

    if a.is_infinite() || b.is_infinite() || res == f64::INFINITY {
        res
    } else {
        -f64::MAX
    }
}

fn generic_rz_clamp_inf(res: f64, a: f64, b: f64) -> f64 {
    assert!(res.is_infinite());

    if a.is_infinite() || b.is_infinite() {
        res
    } else {
        f64::MAX.copysign(res)
    }
}

fn generic_rd_clamp_inf(res: f64, a: f64, b: f64) -> f64 {
    assert!(res.is_infinite());

    if a.is_infinite() || b.is_infinite() || res == -f64::INFINITY {
        res
    } else {
        f64::MAX
    }
}

macro_rules! gen_arity_2 {
    ($name:ident, $round:tt, $name_fallback:ident, $x86_asm_insn:literal, $aarch64_asm_insn:literal) => {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        pub fn $name(a: f64, b: f64) -> f64 {
            use std::arch::asm;
            let mut res: f64;

            unsafe {
                asm!(
                    concat!("v", $x86_asm_insn, " {res}, {a}, {b}, ", get_avx512_round!($round)),
                    a = in(xmm_reg) a,
                    b = in(xmm_reg) b,
                    res = lateout(xmm_reg) res
                );
            }

            res
        }

        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f"), target_feature = "avx"))]
        pub fn $name(a: f64, b: f64) -> f64 {
            use std::arch::asm;
            let mut res = 0.0f64;
            unsafe {
                asm!(
                    concat!("vldmxcsr [{ROUND_BASE} + 4*", get_mxcsr_offs!($round), "]"),
                    concat!("v", $x86_asm_insn, " {res}, {a}, {b}"),
                    "vldmxcsr [{ROUND_BASE}]",
                    res = lateout(xmm_reg) res,
                    a = in(xmm_reg) a,
                    b = in(xmm_reg) b,
                    ROUND_BASE = in(reg) X86_MXCSR_VALUES.as_ptr(),
                )
            }
            res
        }

        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx")))]
        pub fn $name(mut a: f64, b: f64) -> f64 {
            use std::arch::asm;
            unsafe {
                asm!(
                    concat!("ldmxcsr [{ROUND_BASE} + 4*", get_mxcsr_offs!($round), "]"),
                    concat!($x86_asm_insn, " {a}, {b}"),
                    "ldmxcsr [{ROUND_BASE}]",
                    a = inout(xmm_reg) a,
                    b = in(xmm_reg) b,
                    ROUND_BASE = in(reg) X86_MXCSR_VALUES.as_ptr(),
                )
            }
            a
        }

        #[cfg(target_arch = "aarch64")]
        pub fn $name(a: f64, b: f64) -> f64 {
            use std::arch::asm;
            let mut res;
            unsafe {
                asm!(
                    "msr fpcr, {MODE}",
                    concat!($aarch64_asm_insn, " {res:d}, {a:d}, {b:d}"),
                    "msr fpcr, {DEFAULT}",
                    MODE = in(reg) aarch64_fpcr!($round),
                    DEFAULT = in(reg) DEFAULT_NEON_FPCR,
                    a = in(vreg) a,
                    b = in(vreg) b,
                    res = lateout(vreg) res     // won't conflict with DEFAULT, so fine
                );
            }
            res
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        pub fn $name(a: f64, b: f64) -> f64 {
            $name_fallback(a, b)
        }
    };
}

fn add_f64_ru_fallback(a: f64, b: f64) -> f64 {
    let (res, roundoff) = two_sum(a, b);

    if res.is_nan() {
        res
    } else if res.is_infinite() {
        generic_ru_clamp_inf(res, a, b)
    } else {
        // The only way res could be -0.0 is if a and b are both -0.0, which skips f64_next_up_no_special.
        // The only way res could be +inf would result in roundoff = nan, also skipping the next up call.
        if roundoff > 0.0 {
            f64_next_up_no_special(res)
        } else {
            res
        }
    }
}

fn add_f64_rd_fallback(a: f64, b: f64) -> f64 {
    let (res, roundoff) = two_sum(a, b);

    if res.is_infinite() {
        generic_rd_clamp_inf(res, a, b)
    } else if res == 0.0 {
        if a.to_bits() == 0 && b.to_bits() == 0 {
            0.0
        } else {
            -0.0
        }
    } else {
        if roundoff < 0.0 {
            f64_next_down_no_special(res)
        } else {
            res
        }
    }
}

fn add_f64_rz_fallback(a: f64, b: f64) -> f64 {
    let (res, roundoff) = two_sum(a, b);

    if res.is_nan() || res == 0.0 {
        res
    } else if res.is_infinite() {
        generic_rz_clamp_inf(res, a, b)
    } else {
        if roundoff * (1.0f64).copysign(res) < 0.0 {
            if res > 0.0 {
                f64_next_down_no_special(res)
            } else if res < 0.0 {
                f64_next_up_no_special(res)
            } else {
                res
            }
        } else {
            res
        }
    }
}

fn sub_f64_ru_fallback(a: f64, b: f64) -> f64 {
    add_f64_ru_fallback(a, -b)
}

fn sub_f64_rd_fallback(a: f64, b: f64) -> f64 {
    -add_f64_ru_fallback(-a, b)
}

fn sub_f64_rz_fallback(a: f64, b: f64) -> f64 {
    add_f64_rz_fallback(a, -b)
}

fn mul_f64_ru_fallback(a: f64, b: f64) -> f64 {
    let (prod, roundoff) = two_prod(a, b);

    if prod.is_infinite() {
        generic_ru_clamp_inf(prod, a, b)
    } else if prod != 0.0 && prod.abs() < SPLIT_DANGER {  // avoids denormal insanity
        let exact_res = mul_f64_ru_fallback(a * P2_53, b * P2_53);
        let denormal_roundoff = exact_res - prod * P2_106;

        if denormal_roundoff > 0.0 {
            f64_next_up_no_special(prod)
        } else {
            prod
        }
    } else if prod.to_bits() == 0 {
        if a == 0.0 || b == 0.0 {
            prod
        } else {
            F64_TINY
        }
    } else if roundoff > 0.0 {
        f64_next_up_no_special(prod)
    } else {
        prod
    }
}

fn mul_f64_rd_fallback(a: f64, b: f64) -> f64 {
    let (prod, roundoff) = two_prod(a, b);

    if prod.is_infinite() {
        generic_rd_clamp_inf(prod, a, b)
    } else if prod != 0.0 && prod.abs() < SPLIT_DANGER {
        let exact_res = mul_f64_rd_fallback(a * P2_53, b * P2_53);
        let denormal_roundoff = exact_res - prod * P2_106;

        if denormal_roundoff < 0.0 {
            f64_next_down_no_special(prod)
        } else {
            prod
        }
    } else if prod.to_bits() == (-0.0f64).to_bits() {
        if a == 0.0 || b == 0.0 {
            prod
        } else {
            -F64_TINY
        }
    } else if roundoff < 0.0 {
        f64_next_down_no_special(prod)
    } else {
        prod
    }
}

fn mul_f64_rz_fallback(a: f64, b: f64) -> f64 {
    let prod = a * b;

    if prod.is_infinite() {
        generic_rz_clamp_inf(prod, a, b)
    } else if prod > 0.0 {
        mul_f64_rd_fallback(a, b)
    } else if prod < 0.0 {
        mul_f64_ru_fallback(a, b)
    } else {
        prod
    }
}

fn div_err(mut a: f64, b: f64) -> (f64, f64 /* sign meaningful only */) {
    let q = a / b;
    let (mut a2, mut err) = two_prod(q, b);

    if a2 != 0.0 && (a2.abs() < SPLIT_DANGER || err.abs() < SPLIT_DANGER) {
        a *= P2_106;
        (a2, err) = two_prod(q * P2_106, b);
    } else if a.abs() > f64::MAX * 0.5 {   // possible overflow when multiplying back up
        a *= P2_N53;
        (a2, err) = two_prod(q * P2_N53, b);
    }

    err = (a - a2) - err;
    (q, err * 1.0f64.copysign(b))
}

fn div_f64_ru_fallback(a: f64, b: f64) -> f64 {
    let (q, err) = div_err(a, b);

    if q.is_infinite() {
        if b == 0.0 {
            q
        } else {
            generic_ru_clamp_inf(q, a, b)
        }
    } else if err > 0.0 {
        f64_next_up_no_special(q)
    } else {
        q
    }
}

fn div_f64_rd_fallback(a: f64, b: f64) -> f64 {
    let (q, err) = div_err(a, b);

    if q.is_infinite() {
        if b == 0.0 {
            q
        } else {
            generic_rd_clamp_inf(q, a, b)
        }
    } else if err < 0.0 {
        f64_next_down_no_special(q)
    } else {
        q
    }
}

fn div_f64_rz_fallback(a: f64, b: f64) -> f64 {
    let (q, err) = div_err(a, b);

    if q.is_infinite() {
        if b == 0.0 {
            q
        } else {
            generic_rz_clamp_inf(q, a, b)
        }
    } else if err < 0.0 && q > 0.0 {
        f64_next_down_no_special(q)
    } else if err > 0.0 && q < 0.0 {
        f64_next_up_no_special(q)
    } else {
        q
    }
}

/**
 * Generate target-specific implementations and fallbacks.
 */
gen_arity_2!(add_f64_ru, "Up", add_f64_ru_fallback, "addsd", "fadd");
gen_arity_2!( add_f64_rd, "Down", add_f64_rd_fallback, "addsd", "fadd");
gen_arity_2!( add_f64_rz, "Zero", add_f64_rz_fallback, "addsd", "fadd");

gen_arity_2!( sub_f64_ru, "Up", sub_f64_ru_fallback, "subsd", "fsub");
gen_arity_2!( sub_f64_rd, "Down", sub_f64_rd_fallback, "subsd", "fsub");
gen_arity_2!(sub_f64_rz, "Zero", sub_f64_rz_fallback, "subsd", "fsub");

gen_arity_2!(mul_f64_ru, "Up", mul_f64_ru_fallback,"mulsd","fmul");
gen_arity_2!(mul_f64_rd, "Down", mul_f64_rd_fallback, "mulsd", "fmul");
gen_arity_2!(mul_f64_rz, "Zero", mul_f64_rz_fallback, "mulsd", "fmul");

gen_arity_2!(div_f64_ru, "Up", div_f64_ru_fallback, "divsd", "fdiv");
gen_arity_2!(div_f64_rd, "Down", div_f64_rd_fallback, "divsd", "fdiv");
gen_arity_2!(div_f64_rz, "Zero", div_f64_rz_fallback, "divsd", "fdiv");

pub fn add_f64_round(a: f64, b: f64, rounding_mode: RoundingMode) -> f64 {
    match rounding_mode {
        RoundingMode::Up => { add_f64_ru(a, b) }
        RoundingMode::Down => { add_f64_rd(a, b) }
        RoundingMode::Zero => { add_f64_rz(a, b) }
        _ => { a + b }
    }
}

pub fn sub_f64_round(a: f64, b: f64, rounding_mode: RoundingMode) -> f64 {
    match rounding_mode {
        RoundingMode::Up => { sub_f64_ru(a, b) }
        RoundingMode::Down => { sub_f64_rd(a, b) }
        RoundingMode::Zero => { sub_f64_rz(a, b) }
        _ => { a - b }
    }
}

pub fn mul_f64_round(a: f64, b: f64, rounding_mode: RoundingMode) -> f64 {
    match rounding_mode {
        RoundingMode::Up => { mul_f64_ru(a, b) }
        RoundingMode::Down => { mul_f64_rd(a, b) }
        RoundingMode::Zero => { mul_f64_rz(a, b) }
        _ => { a * b }
    }
}

pub fn div_f64_round(a: f64, b: f64, rounding_mode: RoundingMode) -> f64 {
    match rounding_mode {
        RoundingMode::Up => { div_f64_ru(a, b) }
        RoundingMode::Down => { div_f64_rd(a, b) }
        RoundingMode::Zero => { div_f64_rz(a, b) }
        _ => { a / b }
    }
}


/**
 * Square root.
 */

macro_rules! gen_sqrt {
    ($name:ident, $round:tt, $name_fallback:ident) => {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        pub fn $name(a: f64) -> f64 {
            use std::arch::asm;
            let mut res: f64;

            unsafe {
                asm!(
                    concat!("vsqrtsd {res}, {a}, {a}, ", get_avx512_round!($round)),
                    a = in(xmm_reg) a,
                    res = lateout(xmm_reg) res
                );
            }

            res
        }

        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f"), target_feature = "avx"))]
        pub fn $name(a: f64) -> f64 {
            use std::arch::asm;
            let mut res = 0.0f64;
            unsafe {
                asm!(
                    concat!("vldmxcsr [{ROUND_BASE} + 4*", get_mxcsr_offs!($round), "]"),
                    "vsqrtsd {res}, {a}, {a}",
                    "vldmxcsr [{ROUND_BASE}]",
                    res = lateout(xmm_reg) res,
                    a = in(xmm_reg) a,
                    ROUND_BASE = in(reg) X86_MXCSR_VALUES.as_ptr(),
                )
            }
            res
        }

        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx")))]
        pub fn $name(a: f64) -> f64 {
            use std::arch::asm;
            let mut res: f64;
            unsafe {
                asm!(
                    concat!("ldmxcsr [{ROUND_BASE} + 4*", get_mxcsr_offs!($round), "]"),
                    "sqrtsd {res}, {a}",
                    "ldmxcsr [{ROUND_BASE}]",
                    res = out(xmm_reg) res,
                    a = in(xmm_reg) a,
                    ROUND_BASE = in(reg) X86_MXCSR_VALUES.as_ptr(),
                )
            }
            res
        }

        #[cfg(target_arch = "aarch64")]
        pub fn $name(a: f64) -> f64 {
            use std::arch::asm;
            let mut res;

            unsafe {
                asm!(
                    "msr fpcr, {MODE}",
                    "fsqrt {res:d}, {a:d}",
                    "msr fpcr, {DEFAULT}",
                    MODE = in(reg) aarch64_fpcr!($round),
                    DEFAULT = in(reg) DEFAULT_NEON_FPCR,
                    a = in(vreg) a,
                    res = lateout(vreg) res     // won't conflict with DEFAULT, so fine
                );
            }

            res
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        pub fn $name(a: f64) -> f64 {
            $name_fallback(a)
        }
    };
}

fn sqrt_err(mut a: f64) -> (f64, f64 /* only sign is meaningful */) {
    if a == 0.0 {
        return (a, 0.0);
    }

    let sqrt = f64::sqrt(a);
    let (mut a2, mut err) = two_prod(sqrt, sqrt);

    if a2 != 0.0 && (a2.abs() < SPLIT_DANGER || err.abs() < SPLIT_DANGER) {
        a *= P2_106;
        (a2, err) = two_prod(sqrt * P2_106, sqrt);
    } else if a.abs() > f64::MAX * 0.5 {   // possible overflow when going back up
        a *= P2_N53;
        (a2, err) = two_prod(sqrt * P2_N53, sqrt);
    }

    err = (a - a2) - err;
    (sqrt, err)
}

fn next_down_positive(a: f64) -> f64 {
    f64::from_bits(a.to_bits() - 1)
}

fn next_up_positive(a: f64) -> f64 {
    f64::from_bits(a.to_bits() + 1)
}

fn sqrt_f64_rd_fallback(a: f64) -> f64 {
    let (sqrt, err) = sqrt_err(a);
    if sqrt == 0.0 || !sqrt.is_finite() {
        // no clamping down/up since sqrt(MAX_VALUE) is finite
        sqrt
    } else if err < 0.0 {
        next_down_positive(sqrt)
    } else {
        sqrt
    }
}

fn sqrt_f64_rz_fallback(a: f64) -> f64 {
    sqrt_f64_rd_fallback(a)
}

fn sqrt_f64_ru_fallback(a: f64) -> f64 {
    let (sqrt, err) = sqrt_err(a);
    if sqrt == 0.0 || !sqrt.is_finite() {
        sqrt
    } else if err > 0.0 {
        next_up_positive(sqrt)
    } else {
        sqrt
    }
}

gen_sqrt!(sqrt_f64_ru, "Up", sqrt_f64_ru_fallback);
gen_sqrt!(sqrt_f64_rd, "Down", sqrt_f64_rd_fallback);
gen_sqrt!(sqrt_f64_rz, "Zero", sqrt_f64_rz_fallback);

pub fn sqrt_f64_round(a: f64, rounding_mode: RoundingMode) -> f64 {
    match rounding_mode {
        RoundingMode::Up => { sqrt_f64_ru(a) }
        RoundingMode::Down => { sqrt_f64_rd(a) }
        RoundingMode::Zero => { sqrt_f64_rz(a) }
        _ => { f64::sqrt(a) }
    }
}

/**
 * Brute-force correctness tests.
 */

use lazy_static::lazy_static;
lazy_static! {
    pub static ref DIFFICULT_F64: Vec<f64> = gen_f64();
}

fn same_value_zero(a: f64, b: f64) -> bool {
    return a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
}

fn gen_f64() -> Vec<f64> {
    let mut v = Vec::new();

    v.extend_from_slice(&[
        0.0,
        0.5,
        1.0,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        3.08159533890857928181e307f64,
        f64::sqrt(f64::MAX),
        f64::sqrt(f64::EPSILON),
    ]);

    // yucky denormal numbers
    for i in 0..(1 << 10) {
        v.push(f64::from_bits(i));
    }

    for i in 1..=2047u64 {
        v.push(f64::from_bits(i << 52));
    }

    for i in 0..v.len() {
        v.push(v[i].copysign(-1.0));
    }

    let mut rng: u64 = 0;
    for _i in 0..(1 << 14) {
        rng = rng.wrapping_mul(0x12031);
        rng = rng.wrapping_add(3205812);
        rng = (rng >> 4) | (rng << 60);

        let asf64 = f64::from_bits(rng);
        if !asf64.is_nan() {
            v.push(f64_next_down_no_special(asf64));
            v.push(asf64);
            v.push(f64::from_bits(rng & !0xffffffu64));
            v.push(f64_next_up_no_special(asf64));
        }
    }

    v
}

fn check_1(res: f64, expected: f64, a: f64, b: f64, op: &str) {
    if !same_value_zero(res, expected) {
        println!(
            concat!("{:?} {} {:?} = {:?} is correct, got {:?} (bits: {:?} (correct), {:?} (got))"),
            a,
            op,
            b,
            expected,
            res,
            expected.to_bits(),
            res.to_bits()
        );
        assert!(false);
    }
}

fn check_fma(res: f64, expected: f64, a: f64, b: f64, c: f64, op: &str) {
    if !same_value_zero(res, expected) {
        println!(
            r#"{:?} {} {:?} + {:?} = {:?} is correct, got {:?}
        (bits: {:?} (correct), {:?} (got), {:?} (a), {:?} (b), {:?} (c)"#,
            a,
            op,
            b,
            c,
            expected,
            res,
            expected.to_bits(),
            res.to_bits(),
            a.to_bits(),
            b.to_bits(),
            c.to_bits()
        );
        assert!(false);
    }
}

macro_rules! gen_fuzz {
    (f64_1 $test_name:ident, $default:ident, $fallback:ident, $op:literal) => {
        #[test]
        #[ignore]
        fn $test_name() {
            let n: usize = DIFFICULT_F64.len();
            for i in 0..n {
                for j in 0..n {
                    let a = DIFFICULT_F64[i];
                    let b = DIFFICULT_F64[j];

                    let expected = $default(a, b);
                    let res = $fallback(a, b);

                    check_1(res, expected, a, b, $op);
                }
            }
        }
    };

    (f64_3 $test_name:ident, $default:ident, $fallback:ident, $op:literal, $limit2:literal) => {
        #[test]
        #[ignore]
        fn $test_name() {
            const FUZZ_LIMIT_1: usize = 1000;
            const FUZZ_LIMIT_2: i64 = $limit2;

            let mut rng: u64 = 0;

            let mut rng_next = || -> f64 {
                rng = rng.wrapping_mul(0x12031);
                rng = rng.wrapping_add(3205812);
                rng = (rng >> 4) | (rng << 60);

                if rng < 0x10000000000u64 {
                    return DIFFICULT_F64[(rng % 8192) as usize];
                }

                return f64::from_bits(rng);
            };

            let simple_f64 = &DIFFICULT_F64[0..FUZZ_LIMIT_1];
            for &a in simple_f64 {
                for &b in simple_f64 {
                    for &c in simple_f64 {
                        let expected = $default(a, b, c);
                        let res = $fallback(a, b, c);

                        check_fma(res, expected, a, b, c, $op);
                    }
                }
            }

            for _i in 0..FUZZ_LIMIT_2 {   // increase to fuzz harder
                let a = rng_next();
                let b = rng_next();
                let c = rng_next();
                let expected = $default(a, b, c);
                let res = $fallback(a, b, c);

                check_fma(res, expected, a, b, c, $op);
            }
        }
    };

    (f64_4 $test_name:ident, $default:ident, $fallback:ident, $op:literal) => {
        #[test]
        #[ignore]
        fn $test_name() {
            for &a in DIFFICULT_F64.iter() {
                let expected = $default(a);
                let res = $fallback(a);

                check_1(res, expected, a, 0.0, $op);
            }
        }
    };
}

gen_fuzz!(f64_3 fuzz_fma_ru, fma_f64_ru, fma_f64_ru_fallback, "*", 100000000);
gen_fuzz!(f64_3 fuzz_fma_rd, fma_f64_rd, fma_f64_rd_fallback, "*", 100000000);
gen_fuzz!(f64_3 fuzz_fma_rz, fma_f64_rz, fma_f64_rz_fallback, "*", 100000000);
gen_fuzz!(f64_3 fuzz_fma_rne, fma_f64_rne, fma_f64_rne_fallback, "*", 100000000);

gen_fuzz!(f64_3 fuzz_fms_ru, fms_f64_ru, fms_f64_ru_fallback, "*", 100000);
gen_fuzz!(f64_3 fuzz_fms_rd, fms_f64_rd, fms_f64_rd_fallback, "*", 100000);
gen_fuzz!(f64_3 fuzz_fms_rz, fms_f64_rz, fms_f64_rz_fallback, "*", 100000);
gen_fuzz!(f64_3 fuzz_fms_rne, fms_f64_rne, fms_f64_rne_fallback, "*", 100000);

gen_fuzz!(f64_1 fuzz_mul_f64_ru, mul_f64_ru, mul_f64_ru_fallback, "*");
gen_fuzz!(f64_1 fuzz_mul_f64_rd, mul_f64_rd, mul_f64_rd_fallback, "-");
gen_fuzz!(f64_1 fuzz_mul_f64_rz, mul_f64_rz, mul_f64_rz_fallback, "-");

gen_fuzz!(f64_1 fuzz_add_f64_ru, add_f64_ru, add_f64_ru_fallback, "+");
gen_fuzz!(f64_1 fuzz_add_f64_rd, add_f64_rd, add_f64_rd_fallback, "+");
gen_fuzz!(f64_1 fuzz_add_f64_rz, add_f64_rz, add_f64_rz_fallback, "+");

gen_fuzz!(f64_1 fuzz_sub_f64_ru, sub_f64_ru, sub_f64_ru_fallback, "-");
gen_fuzz!(f64_1 fuzz_sub_f64_rd, sub_f64_rd, sub_f64_rd_fallback, "-");
gen_fuzz!(f64_1 fuzz_sub_f64_rz, sub_f64_rz, sub_f64_rz_fallback, "-");

gen_fuzz!(f64_1 fuzz_div_f64_ru, div_f64_ru, div_f64_ru_fallback, "/");
gen_fuzz!(f64_1 fuzz_div_f64_rd, div_f64_rd, div_f64_rd_fallback, "/");
gen_fuzz!(f64_1 fuzz_div_f64_rz, div_f64_rz, div_f64_rz_fallback, "/");

gen_fuzz!(f64_4 fuzz_sqrt_f64_ru, sqrt_f64_ru, sqrt_f64_ru_fallback, "sqrt");
gen_fuzz!(f64_4 fuzz_sqrt_f64_rd, sqrt_f64_rd, sqrt_f64_rd_fallback, "sqrt");
gen_fuzz!(f64_4 fuzz_sqrt_f64_rz, sqrt_f64_rz, sqrt_f64_rz_fallback, "sqrt");
