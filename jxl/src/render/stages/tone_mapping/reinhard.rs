// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::color::tf;

use super::ToneMappingStage;

/// Precomputed Reinhard-in-PQ parameters.
#[derive(Debug, Clone, Copy)]
pub(super) struct ReinhardParams {
    /// PQ signal value corresponding to the desired intensity target (knee point).
    pub pq_knee: f32,
    /// Reinhard softness parameter: half the PQ range between knee and source peak.
    pub reinhard_scale: f32,
}

impl ReinhardParams {
    pub fn new(source_intensity_target: f32, desired_intensity_target: f32) -> Self {
        // PQ signal of desired_intensity_target
        let mut pq_knee_buf = [desired_intensity_target / source_intensity_target];
        tf::linear_to_pq(source_intensity_target, &mut pq_knee_buf);
        let pq_knee = pq_knee_buf[0];

        // PQ signal of source_intensity_target
        let mut pq_peak_buf = [1.0_f32];
        tf::linear_to_pq(source_intensity_target, &mut pq_peak_buf);
        let pq_peak = pq_peak_buf[0];

        // Reinhard softness: half the PQ range above the knee.
        // f(knee) = knee, f'(knee) = 1.0 (C1 continuous), f(∞) → knee + 2*scale.
        let reinhard_scale = (pq_peak - pq_knee) * 0.5;

        Self {
            pq_knee,
            reinhard_scale,
        }
    }
}

/// Reinhard-style tone curve in PQ domain.
///
/// Content at or below the desired intensity target is preserved exactly.
/// Content above is smoothly compressed with a C1-continuous knee:
///   f(x) = knee + excess * scale / (excess + scale)
pub(super) fn process_row(
    stage: &ToneMappingStage,
    xsize: usize,
    row_r: &mut [f32],
    row_g: &mut [f32],
    row_b: &mut [f32],
) {
    let src_it = stage.source_intensity_target;
    let dst_it = stage.desired_intensity_target;
    let [lr, lg, lb] = stage.luminances;
    let pq_knee = stage.reinhard.pq_knee;
    let scale_param = stage.reinhard.reinhard_scale;

    for i in 0..xsize {
        let r = row_r[i];
        let g = row_g[i];
        let b = row_b[i];

        let y_lin = lr * r + lg * g + lb * b;

        if y_lin <= 0.0 {
            continue;
        }

        // Convert luminance to PQ domain
        let mut y_pq_buf = [y_lin];
        tf::linear_to_pq(src_it, &mut y_pq_buf);
        let y_pq = y_pq_buf[0];

        // Apply tone curve: identity below knee, Reinhard above
        let y_pq_out = if y_pq <= pq_knee {
            y_pq
        } else {
            let excess = y_pq - pq_knee;
            pq_knee + excess * scale_param / (excess + scale_param)
        };

        // Convert back to linear
        let mut y_mapped_buf = [y_pq_out];
        tf::pq_to_linear(dst_it, &mut y_mapped_buf);
        let y_mapped_lin = y_mapped_buf[0];

        // Scale RGB by luminance compression ratio
        let ratio = y_mapped_lin / y_lin;

        row_r[i] = r * ratio;
        row_g[i] = g * ratio;
        row_b[i] = b * ratio;
    }
}
