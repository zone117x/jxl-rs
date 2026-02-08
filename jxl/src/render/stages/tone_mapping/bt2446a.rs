// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::ToneMappingStage;
use super::common;

/// BT.2446 Method A in the Y'CbCr' domain per ITU-R BT.2446-1.
///
/// This is the spec-compliant implementation:
/// 1. Gamma-encode each RGB channel: R' = R^(1/2.4), etc.
/// 2. Compute gamma-encoded luma: Y' = lr*R' + lg*G' + lb*B'
/// 3. Apply BT.2446a knee curve to Y' (log compress → knee → inverse log)
/// 4. Compute CbCr chroma from gamma-encoded channels
/// 5. Scale chroma by Y'_mapped / Y'_orig
/// 6. Convert YCbCr back to gamma-encoded RGB
/// 7. Gamma-decode to linear: R = R'^2.4
///
/// Differences from Bt2446aLinear: luminance is computed in gamma-encoded
/// (perceptual) space, and chroma is scaled in the CbCr domain rather than
/// uniformly scaling linear RGB. This can desaturate very bright highlights.
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

        // Step 1: Gamma-encode each channel
        // Use max(0, x) to handle negative values from out-of-gamut content
        let r_prime = r.max(0.0).powf(1.0 / 2.4);
        let g_prime = g.max(0.0).powf(1.0 / 2.4);
        let b_prime = b.max(0.0).powf(1.0 / 2.4);

        // Step 2: Compute gamma-encoded luma
        let y_prime = lr * r_prime + lg * g_prime + lb * b_prime;

        if y_prime <= 0.0 {
            continue;
        }

        // Step 3: Apply BT.2446a knee curve (already in gamma domain)
        let y_prime_mapped = common::bt2446a_knee(&stage.bt2446a, y_prime);

        // Step 4: Compute CbCr
        let cb = (b_prime - y_prime) / (2.0 * (1.0 - lb));
        let cr = (r_prime - y_prime) / (2.0 * (1.0 - lr));

        // Step 5: Scale chroma by compression ratio
        let ratio = y_prime_mapped / y_prime;
        let cb_out = cb * ratio;
        let cr_out = cr * ratio;

        // Step 6: Convert YCbCr back to gamma-encoded RGB
        let r_prime_out = y_prime_mapped + 2.0 * (1.0 - lr) * cr_out;
        let g_prime_out =
            y_prime_mapped - 2.0 * (1.0 - lb) * (lb / lg) * cb_out - 2.0 * (1.0 - lr) * (lr / lg) * cr_out;
        let b_prime_out = y_prime_mapped + 2.0 * (1.0 - lb) * cb_out;

        // Step 7: Gamma-decode to linear
        row_r[i] = r_prime_out.max(0.0).powf(2.4);
        row_g[i] = g_prime_out.max(0.0).powf(2.4);
        row_b[i] = b_prime_out.max(0.0).powf(2.4);
    }
}
