// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! BT.2446a curve in IPTPQc4 perceptual space (libplacebo-style).
//!
//! Converts linear RGB to the IPTPQc4 perceptual color space, applies
//! the BT.2446a knee curve to the I (intensity) channel, scales the
//! P and T (chroma) channels proportionally, then converts back.
//!
//! This provides the best perceptual color preservation for saturated
//! HDR content, at the cost of additional computation (two matrix
//! multiplies + PQ encode/decode per pixel).
//!
//! Requires BT.2020 primaries (validated at pipeline construction time).
//! The tone mapping stage runs before CMS conversion, so input is in the
//! image's native primaries.

use crate::color::tf;

use super::ToneMappingStage;
use super::common;

// IPTPQc4 matrices from BT.2124 / libplacebo.
// RGB(BT.2020) → LMS
const RGB_TO_LMS: [[f32; 3]; 3] = [
    [0.412109, 0.523925, 0.063965],
    [0.166748, 0.720459, 0.112793],
    [0.024170, 0.075440, 0.900390],
];

// LMS_PQ → IPT
const LMS_PQ_TO_IPT: [[f32; 3]; 3] = [
    [2048.0 / 4096.0, 2048.0 / 4096.0, 0.0 / 4096.0],
    [6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0],
    [17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0],
];

// IPT → LMS_PQ (inverse of LMS_PQ_TO_IPT)
const IPT_TO_LMS_PQ: [[f32; 3]; 3] = inv_3x3(LMS_PQ_TO_IPT);

// LMS → RGB(BT.2020) (inverse of RGB_TO_LMS)
const LMS_TO_RGB: [[f32; 3]; 3] = inv_3x3(RGB_TO_LMS);

/// Compile-time 3x3 matrix inverse (Cramer's rule).
const fn inv_3x3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let k = m[2][2]; // using 'k' to avoid shadowing

    let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;

    [
        [
            (e * k - f * h) * inv_det,
            (c * h - b * k) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * k) * inv_det,
            (a * k - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]
}

/// 3x3 matrix × vector multiply.
#[inline(always)]
fn mat_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub(super) fn process_row(
    stage: &ToneMappingStage,
    xsize: usize,
    row_r: &mut [f32],
    row_g: &mut [f32],
    row_b: &mut [f32],
) {
    let src_it = stage.source_intensity_target;

    for i in 0..xsize {
        let r = row_r[i];
        let g = row_g[i];
        let b = row_b[i];

        // Step 1: Linear RGB → LMS
        let lms = mat_mul(&RGB_TO_LMS, [r.max(0.0), g.max(0.0), b.max(0.0)]);

        // Step 2: PQ-encode each LMS channel
        let mut lms_pq = lms;
        tf::linear_to_pq(src_it, &mut lms_pq);

        // Step 3: LMS_PQ → IPT
        let ipt = mat_mul(&LMS_PQ_TO_IPT, lms_pq);
        let i_orig = ipt[0];

        if i_orig <= 0.0 {
            continue;
        }

        // Step 4: Apply BT.2446a knee to I channel.
        // I is in PQ domain [0, 1], similar to gamma-encoded luma.
        let i_mapped = common::bt2446a_knee(stage.bt2446a(), i_orig);

        // Step 5: Scale P and T by compression ratio
        let ratio = i_mapped / i_orig;
        let ipt_mapped = [i_mapped, ipt[1] * ratio, ipt[2] * ratio];

        // Step 6: IPT → LMS_PQ
        let lms_pq_out = mat_mul(&IPT_TO_LMS_PQ, ipt_mapped);

        // Step 7: PQ-decode each LMS channel (to linear, using source IT)
        let mut lms_out = lms_pq_out;
        tf::pq_to_linear(src_it, &mut lms_out);

        // Step 8: LMS → Linear RGB
        let rgb_out = mat_mul(&LMS_TO_RGB, lms_out);

        row_r[i] = rgb_out[0];
        row_g[i] = rgb_out[1];
        row_b[i] = rgb_out[2];
    }
}
