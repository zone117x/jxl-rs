// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! BT.2446a tone mapping per ITU-R BT.2446-1 (spec-compliant).
//!
//! Applies the BT.2446a knee curve in the gamma-encoded (Y') domain:
//! 1. Gamma-encode each RGB channel: R' = R^(1/2.4), etc.
//! 2. Compute luma: Y' = lr·R' + lg·G' + lb·B'
//! 3. Apply BT.2446a knee curve to Y' → Y'_mapped
//! 4. Scale each gamma-encoded channel by ratio = Y'_mapped / Y'
//! 5. Gamma-decode back to linear: R = R'_out^2.4
//!
//! Step 4 is an optimization of the spec's Y'CbCr decomposition. The spec
//! computes Cb/Cr, scales them by the same ratio as Y', then reconstructs
//! R'G'B'. But when luma and chroma share the same ratio, the CbCr round-trip
//! is algebraically a no-op — see proof in the loop body comments.
//!
//! Differences from `bt2446a_linear.rs`: luma Y' is computed from individually
//! gamma-encoded channels (per the spec), not from gamma-encoding the linear
//! luminance. These differ for saturated colors since x^(1/γ) does not
//! distribute over addition.
//!
//! Primaries-agnostic: works with any color space via the `luminances` parameter.

use super::ToneMappingStage;
use super::common;

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
            continue; // Black stays black
        }

        // Step 3: Apply BT.2446a knee curve (already in gamma domain)
        let y_prime_mapped = common::bt2446a_knee(stage.bt2446a(), y_prime);

        // Step 4: Scale each gamma-encoded channel by the compression ratio.
        //
        // This is equivalent to the spec's Y'CbCr decomposition when luma and
        // chroma are scaled by the same ratio. Proof for R' (G', B' analogous):
        //
        //   Cr      = (R' - Y') / (2·(1-lr))
        //   Cr_out  = Cr · ratio           where ratio = Y'_m / Y'
        //   R'_out  = Y'_m + 2·(1-lr) · Cr_out
        //           = Y'_m + (R' - Y') · ratio
        //           = Y'_m + R'·ratio - Y'·ratio
        //           = Y'_m + R'·ratio - Y'_m
        //           = R' · ratio           ∎
        let ratio = y_prime_mapped / y_prime;
        let r_prime_out = r_prime * ratio;
        let g_prime_out = g_prime * ratio;
        let b_prime_out = b_prime * ratio;

        // Step 5: Gamma-decode to linear
        row_r[i] = r_prime_out.max(0.0).powf(2.4);
        row_g[i] = g_prime_out.max(0.0).powf(2.4);
        row_b[i] = b_prime_out.max(0.0).powf(2.4);
    }
}
