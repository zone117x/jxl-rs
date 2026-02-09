// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Rec. 2408 / BT.2390-style tone mapping matching libjxl's `Rec2408ToneMapperBase`.
//!
//! Operates in the PQ perceptual domain with a cubic Hermite spline knee curve,
//! followed by `GamutMapScalar` desaturation. This is the algorithm libjxl uses
//! in its render pipeline (`stage_tone_mapping.cc`).
//!
//! Unlike the BT.2446a variants, output is re-normalized:
//! - Input:  1.0 = `source_intensity_target` nits
//! - Output: 1.0 = `desired_intensity_target` nits
//!
//! See also `Rec2408ToneMapper` in `api/color.rs` for the single-pixel version
//! used by the ICC profile path.

use super::ToneMappingStage;

/// libjxl's default `preserve_saturation` for the render pipeline path.
const PRESERVE_SATURATION: f32 = 0.1;

/// Precomputed Rec. 2408 / BT.2390 parameters for PQ-domain tone mapping.
///
/// Matches libjxl's `Rec2408ToneMapperBase` (see also `Rec2408ToneMapper` in `api/color.rs`).
#[derive(Debug, Clone, Copy)]
pub(super) struct Rec2408Params {
    pub pq_mastering_min: f32,
    pub pq_mastering_range: f32,
    pub inv_pq_mastering_range: f32,
    pub min_lum: f32,
    pub max_lum: f32,
    pub ks: f32,
    inv_one_minus_ks: f32,
    pub source_peak: f32,
    pub normalizer: f32,
    pub inv_target_peak: f32,
    pub target_peak: f32,
}

impl Rec2408Params {
    pub fn new(source_range: [f32; 2], target_range: [f32; 2]) -> Self {
        let pq_mastering_min = pq_encode_nits(source_range[0]);
        let pq_mastering_max = pq_encode_nits(source_range[1]);
        let pq_mastering_range = pq_mastering_max - pq_mastering_min;
        let inv_pq_mastering_range = 1.0 / pq_mastering_range;

        let min_lum = (pq_encode_nits(target_range[0]) - pq_mastering_min) * inv_pq_mastering_range;
        let max_lum = (pq_encode_nits(target_range[1]) - pq_mastering_min) * inv_pq_mastering_range;
        let ks = 1.5 * max_lum - 0.5;

        Self {
            pq_mastering_min,
            pq_mastering_range,
            inv_pq_mastering_range,
            min_lum,
            max_lum,
            ks,
            inv_one_minus_ks: 1.0 / (1.0 - ks).max(1e-6),
            source_peak: source_range[1],
            normalizer: source_range[1] / target_range[1],
            inv_target_peak: 1.0 / target_range[1],
            target_peak: target_range[1],
        }
    }

    /// Hermite spline knee curve (BT.2390 §5.4).
    #[inline]
    pub fn hermite_spline(&self, b: f32) -> f32 {
        let t = (b - self.ks) * self.inv_one_minus_ks;
        let t2 = t * t;
        let t3 = t2 * t;
        (2.0 * t3 - 3.0 * t2 + 1.0) * self.ks
            + (t3 - 2.0 * t2 + t) * (1.0 - self.ks)
            + (-2.0 * t3 + 3.0 * t2) * self.max_lum
    }
}

/// PQ inverse EOTF: encode absolute luminance (nits) to PQ signal [0, 1].
///
/// Equivalent to libjxl's `TF_PQ_Base::EncodedFromDisplay` with `display_intensity_target=1.0`.
pub(super) fn pq_encode_nits(luminance_nits: f32) -> f32 {
    let mut val = [luminance_nits / 10000.0];
    crate::color::tf::linear_to_pq_precise(10000.0, &mut val);
    val[0]
}

/// PQ EOTF: decode PQ signal [0, 1] to absolute luminance (nits).
///
/// Equivalent to libjxl's `TF_PQ_Base::DisplayFromEncoded` with `display_intensity_target=1.0`.
pub(super) fn pq_decode_nits(encoded: f32) -> f32 {
    let mut val = [encoded];
    crate::color::tf::pq_to_linear_precise(10000.0, &mut val);
    val[0] * 10000.0
}

pub(super) fn process_row(
    stage: &ToneMappingStage,
    xsize: usize,
    row_r: &mut [f32],
    row_g: &mut [f32],
    row_b: &mut [f32],
) {
    let params = stage.rec2408();
    let [lr, lg, lb] = stage.luminances;

    for i in 0..xsize {
        let r = row_r[i];
        let g = row_g[i];
        let b = row_b[i];

        // Step 1: Compute luminance in nits.
        let luminance = params.source_peak * (lr * r + lg * g + lb * b);

        // Step 2: PQ-encode luminance, normalize to [0, 1] within mastering range.
        let normalized_pq = ((pq_encode_nits(luminance) - params.pq_mastering_min)
            * params.inv_pq_mastering_range)
            .min(1.0);

        // Step 3: Apply knee curve — identity below ks, Hermite spline above.
        let e2 = if normalized_pq < params.ks {
            normalized_pq
        } else {
            params.hermite_spline(normalized_pq)
        };

        // Step 4: Black level lift via 4th-power rolloff.
        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let e3 = params.min_lum * (one_minus_e2_2 * one_minus_e2_2) + e2;

        // Step 5: Denormalize PQ and decode back to nits.
        let e4 = e3 * params.pq_mastering_range + params.pq_mastering_min;
        let new_luminance = pq_decode_nits(e4).clamp(0.0, params.target_peak);

        // Step 6: Scale all channels (preserving chromaticity).
        const MIN_LUMINANCE: f32 = 1e-6;
        if luminance <= MIN_LUMINANCE {
            let cap = new_luminance * params.inv_target_peak;
            row_r[i] = cap;
            row_g[i] = cap;
            row_b[i] = cap;
        } else {
            let multiplier = (new_luminance / luminance) * params.normalizer;
            row_r[i] = r * multiplier;
            row_g[i] = g * multiplier;
            row_b[i] = b * multiplier;
        }

        // Step 7: Gamut mapping (GamutMapScalar).
        gamut_map(&mut row_r[i], &mut row_g[i], &mut row_b[i], lr, lg, lb);
    }
}

/// Desaturate out-of-gamut pixels while preserving luminance.
///
/// Matches libjxl's `GamutMapScalar` with `preserve_saturation = 0.1`.
#[inline]
fn gamut_map(r: &mut f32, g: &mut f32, b: &mut f32, lr: f32, lg: f32, lb: f32) {
    let luminance = lr * *r + lg * *g + lb * *b;

    let mut gray_mix_saturation = 0.0_f32;
    let mut gray_mix_luminance = 0.0_f32;

    for &val in &[*r, *g, *b] {
        let val_minus_gray = val - luminance;
        let inv_val_minus_gray = if val_minus_gray == 0.0 {
            1.0
        } else {
            1.0 / val_minus_gray
        };
        let val_over_val_minus_gray = val * inv_val_minus_gray;

        if val_minus_gray < 0.0 {
            gray_mix_saturation = gray_mix_saturation.max(val_over_val_minus_gray);
        }

        gray_mix_luminance = gray_mix_luminance.max(if val_minus_gray <= 0.0 {
            gray_mix_saturation
        } else {
            val_over_val_minus_gray - inv_val_minus_gray
        });
    }

    let gray_mix = (PRESERVE_SATURATION * (gray_mix_saturation - gray_mix_luminance)
        + gray_mix_luminance)
        .clamp(0.0, 1.0);

    *r = gray_mix * (luminance - *r) + *r;
    *g = gray_mix * (luminance - *g) + *g;
    *b = gray_mix * (luminance - *b) + *b;

    let max_clr = r.max(*g).max(*b).max(1.0);
    let normalizer = 1.0 / max_clr;
    *r *= normalizer;
    *g *= normalizer;
    *b *= normalizer;
}
