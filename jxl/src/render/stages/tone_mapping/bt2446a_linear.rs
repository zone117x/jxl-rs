// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::ToneMappingStage;
use super::common;

/// BT.2446a curve applied to linear RGB luminance (fast approximation).
///
/// Computes luminance in linear space, applies the full BT.2446a curve
/// (gamma encode → log compress → knee → inverse log → linearize),
/// then scales all RGB channels by the luminance compression ratio.
pub(super) fn process_row(
    stage: &ToneMappingStage,
    xsize: usize,
    row_r: &mut [f32],
    row_g: &mut [f32],
    row_b: &mut [f32],
) {
    let [lr, lg, lb] = stage.luminances;

    for i in 0..xsize {
        let r = row_r[i];
        let g = row_g[i];
        let b = row_b[i];

        // Compute linear luminance (normalized, 1.0 = source_it nits)
        let y_lin = lr * r + lg * g + lb * b;

        if y_lin <= 0.0 {
            continue; // Black stays black
        }

        // Apply BT.2446a tone curve to luminance
        let y_mapped = common::bt2446a_map(&stage.bt2446a, y_lin);

        // Scale RGB by luminance compression ratio (preserves hue/saturation)
        let ratio = y_mapped / y_lin;

        row_r[i] = r * ratio;
        row_g[i] = g * ratio;
        row_b[i] = b * ratio;
    }
}
