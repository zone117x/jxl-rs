// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::color::tf::{
    hlg_display_to_scene, hlg_scene_to_display, linear_to_pq_precise, pq_to_linear_precise,
};
use crate::render::RenderPipelineInPlaceStage;

/// HDR tone mapping render pipeline stage.
///
/// Maps linear RGB pixel values from a source intensity range to a desired
/// target intensity range. For PQ content, uses BT.2408 tone mapping curve.
/// For HLG content, re-applies the OOTF for the desired display luminance.
///
/// After this stage, pixel values are normalized so that 1.0 corresponds to
/// `desired_intensity_target` nits (instead of the file's original intensity_target).
pub struct ToneMappingStage {
    first_channel: usize,
    method: ToneMapMethod,
}

enum ToneMapMethod {
    /// BT.2408 tone mapping for PQ content.
    Pq(Rec2408ToneMapper),
    /// HLG OOTF re-application: undo source display adaptation, apply desired.
    Hlg {
        source_intensity_target: f32,
        desired_intensity_target: f32,
        luminances: [f32; 3],
    },
}

impl ToneMappingStage {
    /// Create a tone mapping stage for PQ content using BT.2408.
    pub fn new_pq(
        first_channel: usize,
        source_intensity_target: f32,
        desired_intensity_target: f32,
        min_nits: f32,
        luminances: [f32; 3],
    ) -> Self {
        Self {
            first_channel,
            method: ToneMapMethod::Pq(Rec2408ToneMapper::new(
                (min_nits, source_intensity_target),
                (min_nits, desired_intensity_target),
                luminances,
            )),
        }
    }

    /// Create a tone mapping stage for HLG content using OOTF re-application.
    pub fn new_hlg(
        first_channel: usize,
        source_intensity_target: f32,
        desired_intensity_target: f32,
        luminances: [f32; 3],
    ) -> Self {
        Self {
            first_channel,
            method: ToneMapMethod::Hlg {
                source_intensity_target,
                desired_intensity_target,
                luminances,
            },
        }
    }
}

impl std::fmt::Display for ToneMappingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.first_channel;
        match &self.method {
            ToneMapMethod::Pq(mapper) => write!(
                f,
                "PQ tone map ({:.0} -> {:.0} nits) for channels [{},{},{}]",
                mapper.source_range.1,
                mapper.target_range.1,
                c,
                c + 1,
                c + 2
            ),
            ToneMapMethod::Hlg {
                source_intensity_target,
                desired_intensity_target,
                ..
            } => write!(
                f,
                "HLG OOTF ({:.0} -> {:.0} nits) for channels [{},{},{}]",
                source_intensity_target,
                desired_intensity_target,
                c,
                c + 1,
                c + 2
            ),
        }
    }
}

impl RenderPipelineInPlaceStage for ToneMappingStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        (self.first_channel..self.first_channel + 3).contains(&c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let [row_r, row_g, row_b] = row else {
            panic!(
                "incorrect number of channels; expected 3, found {}",
                row.len()
            );
        };

        match &self.method {
            ToneMapMethod::Pq(tone_mapper) => {
                for i in 0..xsize {
                    let mut rgb = [row_r[i], row_g[i], row_b[i]];
                    tone_mapper.tone_map(&mut rgb);
                    row_r[i] = rgb[0];
                    row_g[i] = rgb[1];
                    row_b[i] = rgb[2];
                }
            }
            ToneMapMethod::Hlg {
                source_intensity_target,
                desired_intensity_target,
                luminances,
            } => {
                // Undo source display adaptation (display→scene)
                let rows_undo = [
                    &mut row_r[..xsize],
                    &mut row_g[..xsize],
                    &mut row_b[..xsize],
                ];
                hlg_display_to_scene(*source_intensity_target, *luminances, rows_undo);

                // Apply desired display adaptation (scene→display)
                let rows_apply = [
                    &mut row_r[..xsize],
                    &mut row_g[..xsize],
                    &mut row_b[..xsize],
                ];
                hlg_scene_to_display(*desired_intensity_target, *luminances, rows_apply);
            }
        }
    }
}

/// BT.2408 HDR to SDR tone mapper.
/// Maps PQ content from source range (e.g., 0-10000 nits) to target range (e.g., 0-250 nits).
struct Rec2408ToneMapper {
    source_range: (f32, f32), // (min, max) in nits
    target_range: (f32, f32),
    luminances: [f32; 3], // RGB luminance coefficients (Y values)

    // Precomputed values
    pq_mastering_min: f32,
    #[allow(dead_code)] // Stored for potential future use / debugging
    pq_mastering_max: f32,
    pq_mastering_range: f32,
    inv_pq_mastering_range: f32,
    min_lum: f32,
    max_lum: f32,
    ks: f32,
    inv_one_minus_ks: f32,
    normalizer: f32,
    inv_target_peak: f32,
}

impl Rec2408ToneMapper {
    fn new(source_range: (f32, f32), target_range: (f32, f32), luminances: [f32; 3]) -> Self {
        let pq_mastering_min = Self::linear_to_pq(source_range.0);
        let pq_mastering_max = Self::linear_to_pq(source_range.1);
        let pq_mastering_range = pq_mastering_max - pq_mastering_min;
        let inv_pq_mastering_range = 1.0 / pq_mastering_range;

        let min_lum =
            (Self::linear_to_pq(target_range.0) - pq_mastering_min) * inv_pq_mastering_range;
        let max_lum =
            (Self::linear_to_pq(target_range.1) - pq_mastering_min) * inv_pq_mastering_range;
        let ks = 1.5 * max_lum - 0.5;

        Self {
            source_range,
            target_range,
            luminances,
            pq_mastering_min,
            pq_mastering_max,
            pq_mastering_range,
            inv_pq_mastering_range,
            min_lum,
            max_lum,
            ks,
            inv_one_minus_ks: 1.0 / (1.0 - ks).max(1e-6),
            normalizer: source_range.1 / target_range.1,
            inv_target_peak: 1.0 / target_range.1,
        }
    }

    /// PQ inverse EOTF - converts luminance (nits) to PQ encoded value.
    fn linear_to_pq(luminance: f32) -> f32 {
        let mut val = [luminance / 10000.0];
        linear_to_pq_precise(10000.0, &mut val);
        val[0]
    }

    /// PQ EOTF - converts PQ encoded value to luminance (nits).
    fn pq_to_linear(encoded: f32) -> f32 {
        let mut val = [encoded];
        pq_to_linear_precise(10000.0, &mut val);
        val[0] * 10000.0
    }

    fn t(&self, a: f32) -> f32 {
        (a - self.ks) * self.inv_one_minus_ks
    }

    fn p(&self, b: f32) -> f32 {
        let t_b = self.t(b);
        let t_b_2 = t_b * t_b;
        let t_b_3 = t_b_2 * t_b;
        (2.0 * t_b_3 - 3.0 * t_b_2 + 1.0) * self.ks
            + (t_b_3 - 2.0 * t_b_2 + t_b) * (1.0 - self.ks)
            + (-2.0 * t_b_3 + 3.0 * t_b_2) * self.max_lum
    }

    /// Apply tone mapping to RGB values (in-place).
    /// Input: RGB normalized so 1.0 = source_range.1 nits.
    /// Output: RGB normalized so 1.0 = target_range.1 nits.
    fn tone_map(&self, rgb: &mut [f32; 3]) {
        let luminance = self.source_range.1
            * (self.luminances[0] * rgb[0]
                + self.luminances[1] * rgb[1]
                + self.luminances[2] * rgb[2]);

        let normalized_pq = ((Self::linear_to_pq(luminance) - self.pq_mastering_min)
            * self.inv_pq_mastering_range)
            .min(1.0);

        let e2 = if normalized_pq < self.ks {
            normalized_pq
        } else {
            self.p(normalized_pq)
        };

        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let one_minus_e2_4 = one_minus_e2_2 * one_minus_e2_2;
        let e3 = self.min_lum * one_minus_e2_4 + e2;
        let e4 = e3 * self.pq_mastering_range + self.pq_mastering_min;
        let d4 = Self::pq_to_linear(e4);
        let new_luminance = d4.clamp(0.0, self.target_range.1);

        let min_luminance = 1e-6;
        let use_cap = luminance <= min_luminance;
        let ratio = new_luminance / luminance.max(min_luminance);
        let cap = new_luminance * self.inv_target_peak;
        let multiplier = ratio * self.normalizer;

        for c in rgb.iter_mut() {
            *c = if use_cap { cap } else { *c * multiplier };
        }
    }
}
