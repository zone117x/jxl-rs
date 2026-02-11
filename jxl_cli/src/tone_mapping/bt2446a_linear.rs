// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! BT.2446a-inspired tone mapping in the linear RGB domain.
//!
//! This is a fast approximation, NOT a spec-compliant BT.2446 Method A implementation.
//! It differs from the spec (`bt2446a.rs`) in two structural ways:
//!
//! 1. **Luminance vs luma**: computes luminance as a weighted sum of linear channels
//!    (`Y = lr·R + lg·G + lb·B`), then gamma-encodes Y for the knee curve. The spec
//!    gamma-encodes each channel first, then computes luma (`Y' = lr·R' + lg·G' + lb·B'`).
//!    These are not equivalent — `(ΣwᵢCᵢ)^(1/γ) ≠ Σwᵢ(Cᵢ^(1/γ))`. For pure BT.2020
//!    red (R=1,G=0,B=0) the values fed into the knee differ by 2.1× (0.556 vs 0.263).
//!    For achromatic content the two are identical.
//!
//! 2. **Ratio domain**: scales linear RGB channels directly by `Y_mapped / Y_lin`.
//!    The spec-compliant path (`bt2446a.rs`) scales gamma-encoded channels, which is
//!    equivalent to `C_out = C · ratio^γ` in linear space — a stronger compression
//!    per channel for the same luminance ratio.
//!
//! In practice the two produce visually similar results because most image content
//! is not extremely saturated, and the knee curve is smooth (nearby inputs → nearby
//! outputs). The difference is most visible on bright, highly saturated highlights.
//!
//! Roughly ~40% faster than `bt2446a.rs` (2 `powf` calls per pixel vs 6).
//!
//! Primaries-agnostic: works with any color space via the `luminances` parameter.

use super::common;

/// Full BT.2446a tone curve for linear luminance values.
///
/// Wraps `bt2446a_knee` with gamma encode and linearize:
/// 1. BT.1886 OETF (gamma encode): `y^(1/2.4)`
/// 2. `bt2446a_knee` (log compress → knee → inverse log)
/// 3. BT.1886 EOTF (linearize): `x^2.4`
#[inline]
pub fn bt2446a_map(params: &common::Bt2446aParams, y: f32) -> f32 {
    let y_prime = y.powf(1.0 / 2.4);
    common::bt2446a_knee(params, y_prime).powf(2.4)
}

/// BT.2446a-linear tone mapping on interleaved RGB data.
///
/// Computes linear luminance, applies `bt2446a_map`, scales all channels
/// by the ratio.
///
/// `data` is interleaved `[R, G, B, R, G, B, …]` in linear light,
/// where 1.0 = source peak luminance.
pub fn tone_map_bt2446a_linear(
    params: &common::Bt2446aParams,
    luminances: [f32; 3],
    data: &mut [f32],
) {
    let [lr, lg, lb] = luminances;
    let num_pixels = data.len() / 3;

    for i in 0..num_pixels {
        let base = i * 3;
        let r = data[base];
        let g = data[base + 1];
        let b = data[base + 2];

        let y_lin = lr * r + lg * g + lb * b;
        if y_lin <= 0.0 {
            continue;
        }

        let ratio = bt2446a_map(params, y_lin) / y_lin;
        data[base] = r * ratio;
        data[base + 1] = g * ratio;
        data[base + 2] = b * ratio;
    }
}
