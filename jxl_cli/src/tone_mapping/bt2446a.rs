// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! BT.2446a spec-compliant tone mapping in the Y'CbCr' domain.
//!
//! Gamma-encodes each channel with BT.1886 OETF (γ = 1/2.4), computes
//! Y' luma, applies the BT.2446a knee curve, scales all gamma-encoded
//! channels by the ratio, then gamma-decodes back to linear.
//!
//! This is the reference method from ITU-R BT.2446-1 Method A.

use super::common::{Bt2446aParams, bt2446a_knee};

/// BT.2446a spec-compliant tone mapping on interleaved RGB data.
///
/// Gamma-encodes each channel, computes Y' luma, applies the knee curve,
/// scales all gamma-encoded channels by the ratio, then gamma-decodes.
///
/// `data` is interleaved `[R, G, B, R, G, B, …]` in linear light,
/// where 1.0 = source peak luminance.
pub fn tone_map_bt2446a(params: &Bt2446aParams, luminances: [f32; 3], data: &mut [f32]) {
    let [lr, lg, lb] = luminances;
    let num_pixels = data.len() / 3;

    for i in 0..num_pixels {
        let base = i * 3;
        let r_prime = data[base].max(0.0).powf(1.0 / 2.4);
        let g_prime = data[base + 1].max(0.0).powf(1.0 / 2.4);
        let b_prime = data[base + 2].max(0.0).powf(1.0 / 2.4);

        let y_prime = lr * r_prime + lg * g_prime + lb * b_prime;
        if y_prime <= 0.0 {
            continue;
        }

        let ratio = bt2446a_knee(params, y_prime) / y_prime;

        data[base] = (r_prime * ratio).max(0.0).powf(2.4);
        data[base + 1] = (g_prime * ratio).max(0.0).powf(2.4);
        data[base + 2] = (b_prime * ratio).max(0.0).powf(2.4);
    }
}
