// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::color::tf;
use crate::render::RenderPipelineInPlaceStage;

/// Tone maps HDR linear RGB to a lower intensity target.
///
/// Input: planar f32 linear RGB where 1.0 = `source_intensity_target` nits.
/// Output: planar f32 linear RGB where 1.0 = `desired_intensity_target` nits.
///
/// Uses a Reinhard-style tone curve in the PQ (perceptually uniform) domain:
/// - Content at or below the desired intensity target is preserved exactly
/// - Content above is smoothly compressed with a C1-continuous knee
/// - RGB channel ratios (hue/saturation) are preserved by scaling all channels
///   uniformly based on the luminance compression ratio
#[derive(Debug)]
pub struct ToneMappingStage {
    first_channel: usize,
    source_intensity_target: f32,
    desired_intensity_target: f32,
    luminances: [f32; 3],
    /// PQ signal value corresponding to the desired intensity target (knee point).
    pq_knee: f32,
    /// Reinhard softness parameter: controls how quickly highlights are compressed.
    /// Equals half the PQ range between knee and source peak.
    reinhard_scale: f32,
}

impl ToneMappingStage {
    pub fn new(
        first_channel: usize,
        source_intensity_target: f32,
        desired_intensity_target: f32,
        luminances: [f32; 3],
    ) -> Self {
        // Precompute PQ knee point (PQ signal of desired_intensity_target).
        // linear_to_pq with source IT maps linear 1.0 = source_it nits.
        // desired_it / source_it in linear = desired_it nits absolute.
        let mut pq_knee_buf = [desired_intensity_target / source_intensity_target];
        tf::linear_to_pq(source_intensity_target, &mut pq_knee_buf);
        let pq_knee = pq_knee_buf[0];

        // Precompute PQ source peak (PQ signal of source_intensity_target).
        let mut pq_peak_buf = [1.0_f32];
        tf::linear_to_pq(source_intensity_target, &mut pq_peak_buf);
        let pq_peak = pq_peak_buf[0];

        // Reinhard softness: half the PQ range above the knee.
        // This gives a smooth rolloff where the derivative at the knee = 1.0 (C1 continuous)
        // and asymptotically approaches pq_knee + 2*reinhard_scale.
        let reinhard_scale = (pq_peak - pq_knee) * 0.5;

        Self {
            first_channel,
            source_intensity_target,
            desired_intensity_target,
            luminances,
            pq_knee,
            reinhard_scale,
        }
    }
}

impl std::fmt::Display for ToneMappingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tone mapping {} -> {} nits on channels [{},{},{}]",
            self.source_intensity_target,
            self.desired_intensity_target,
            self.first_channel,
            self.first_channel + 1,
            self.first_channel + 2,
        )
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

        let src_it = self.source_intensity_target;
        let dst_it = self.desired_intensity_target;
        let [lr, lg, lb] = self.luminances;
        let pq_knee = self.pq_knee;
        let scale_param = self.reinhard_scale;

        for i in 0..xsize {
            let r = row_r[i];
            let g = row_g[i];
            let b = row_b[i];

            // Compute linear luminance
            let y_lin = lr * r + lg * g + lb * b;

            if y_lin <= 0.0 {
                continue; // Black stays black
            }

            // Convert luminance to PQ domain using source intensity target.
            let mut y_pq_buf = [y_lin];
            tf::linear_to_pq(src_it, &mut y_pq_buf);
            let y_pq = y_pq_buf[0];

            // Apply tone curve in PQ domain:
            // - Below knee (SDR range): pass through unchanged
            // - Above knee (HDR highlights): Reinhard compression
            //   f(x) = knee + excess * scale / (excess + scale)
            //   where excess = x - knee
            //   Properties: f(knee) = knee, f'(knee) = 1.0 (C1), f(∞) → knee + scale
            let y_pq_out = if y_pq <= pq_knee {
                y_pq
            } else {
                let excess = y_pq - pq_knee;
                pq_knee + excess * scale_param / (excess + scale_param)
            };

            // Convert mapped luminance back to linear using desired intensity target.
            let mut y_mapped_buf = [y_pq_out];
            tf::pq_to_linear(dst_it, &mut y_mapped_buf);
            let y_mapped_lin = y_mapped_buf[0];

            // Scale RGB by luminance compression ratio (preserves hue/saturation)
            let ratio = y_mapped_lin / y_lin;

            row_r[i] = r * ratio;
            row_g[i] = g * ratio;
            row_b[i] = b * ratio;
        }
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::util::test::assert_all_almost_abs_eq;

    const LUMINANCE_BT2020: [f32; 3] = [0.2627, 0.678, 0.0593];

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || ToneMappingStage::new(0, 10000.0, 203.0, LUMINANCE_BT2020),
            (500, 500),
            3,
        )
    }

    #[test]
    fn sdr_white_preserved() -> Result<()> {
        // 203 nits is SDR reference white (ITU-R BT.2408).
        // In a 10000-nit PQ image, 203 nits = 0.0203 linear (since 1.0 = 10000 nits).
        // After tone mapping to 203 nits target, SDR white should map to ~1.0 linear.
        let source_it = 10000.0_f32;
        let desired_it = 203.0_f32;
        let sdr_white_linear = desired_it / source_it;

        let input_r = Image::new_with_value((1, 1), sdr_white_linear)?;
        let input_g = Image::new_with_value((1, 1), sdr_white_linear)?;
        let input_b = Image::new_with_value((1, 1), sdr_white_linear)?;

        let stage = ToneMappingStage::new(0, source_it, desired_it, LUMINANCE_BT2020);
        let output =
            make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

        // SDR white is exactly at the knee point, so it passes through unchanged.
        // PQ approximation introduces small error.
        let mapped = output[0].row(0)[0];
        assert!(
            (mapped - 1.0).abs() < 0.02,
            "SDR white (203 nits) should map to ~1.0, got {mapped}"
        );

        Ok(())
    }

    #[test]
    fn bright_highlights_compressed_not_clipped() -> Result<()> {
        // A very bright pixel (5000 nits) in a 10000-nit image should be compressed,
        // not just clipped to 1.0. It should be > 1.0 but much less than 5000/203.
        let source_it = 10000.0_f32;
        let desired_it = 203.0_f32;
        let bright_linear = 5000.0 / source_it; // 0.5 linear

        let input_r = Image::new_with_value((1, 1), bright_linear)?;
        let input_g = Image::new_with_value((1, 1), bright_linear)?;
        let input_b = Image::new_with_value((1, 1), bright_linear)?;

        let stage = ToneMappingStage::new(0, source_it, desired_it, LUMINANCE_BT2020);
        let output =
            make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

        let mapped = output[0].row(0)[0];
        // Should be above SDR white (since it's brighter than SDR white)
        assert!(
            mapped > 1.0,
            "5000-nit pixel should map above 1.0, got {mapped}"
        );
        // Should be compressed well below naive linear scaling (5000/203 ≈ 24.6)
        assert!(
            mapped < 10.0,
            "5000-nit pixel should be compressed (naive ≈ 24.6, got {mapped})"
        );

        Ok(())
    }

    #[test]
    fn black_unchanged() -> Result<()> {
        let input_r = Image::new_with_value((1, 1), 0.0)?;
        let input_g = Image::new_with_value((1, 1), 0.0)?;
        let input_b = Image::new_with_value((1, 1), 0.0)?;

        let stage = ToneMappingStage::new(0, 10000.0, 203.0, LUMINANCE_BT2020);
        let output =
            make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

        assert_all_almost_abs_eq(output[0].row(0), &[0.0], 1e-6);
        assert_all_almost_abs_eq(output[1].row(0), &[0.0], 1e-6);
        assert_all_almost_abs_eq(output[2].row(0), &[0.0], 1e-6);

        Ok(())
    }

    #[test]
    fn color_ratios_preserved() -> Result<()> {
        // A colored pixel should maintain its R:G:B ratios after tone mapping
        // (since we scale all channels by the same luminance ratio).
        let source_it = 10000.0_f32;
        let desired_it = 203.0_f32;

        let r_val = 0.1_f32;
        let g_val = 0.05_f32;
        let b_val = 0.02_f32;

        let input_r = Image::new_with_value((1, 1), r_val)?;
        let input_g = Image::new_with_value((1, 1), g_val)?;
        let input_b = Image::new_with_value((1, 1), b_val)?;

        let stage = ToneMappingStage::new(0, source_it, desired_it, LUMINANCE_BT2020);
        let output =
            make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

        let out_r = output[0].row(0)[0];
        let out_g = output[1].row(0)[0];
        let out_b = output[2].row(0)[0];

        // Check ratios are preserved
        let in_ratio_rg = r_val / g_val;
        let out_ratio_rg = out_r / out_g;
        assert!(
            (in_ratio_rg - out_ratio_rg).abs() < 1e-4,
            "R:G ratio should be preserved: in={in_ratio_rg}, out={out_ratio_rg}"
        );

        let in_ratio_rb = r_val / b_val;
        let out_ratio_rb = out_r / out_b;
        assert!(
            (in_ratio_rb - out_ratio_rb).abs() < 1e-4,
            "R:B ratio should be preserved: in={in_ratio_rb}, out={out_ratio_rb}"
        );

        Ok(())
    }

    #[test]
    fn monotonic_increasing() -> Result<()> {
        // Brighter input should always produce brighter output.
        let source_it = 10000.0_f32;
        let desired_it = 203.0_f32;
        let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
        let mut prev_output = 0.0_f32;

        for &val in &test_values {
            let input_r = Image::new_with_value((1, 1), val)?;
            let input_g = Image::new_with_value((1, 1), val)?;
            let input_b = Image::new_with_value((1, 1), val)?;

            let s = ToneMappingStage::new(0, source_it, desired_it, LUMINANCE_BT2020);
            let output =
                make_and_run_simple_pipeline(s, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

            let mapped = output[0].row(0)[0];
            assert!(
                mapped > prev_output,
                "Tone map should be monotonic: input {val} → {mapped}, but prev was {prev_output}"
            );
            prev_output = mapped;
        }

        Ok(())
    }
}
