// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::RenderPipelineInPlaceStage;

/// Tone maps HDR linear RGB to a lower intensity target using the ITU-R
/// BT.2446 Method A tone curve applied in linear RGB space.
///
/// Note: the full BT.2446-1 spec applies the curve in the Y'CbCr' domain
/// (gamma-encode each channel, convert to YCbCr, curve Y', scale CbCr).
/// This implementation instead extracts luminance from linear RGB, applies
/// the curve, and scales all channels by the compression ratio. The
/// differences are: (1) luminance is computed in linear space rather than
/// perceptual space (gamma is non-linear so the weighted sum differs), and
/// (2) chromaticity is preserved exactly, whereas the spec's CbCr scaling
/// can desaturate very bright highlights. In practice the results are similar
/// for most natural content; differences are most visible on highly saturated
/// bright highlights (neon signs, colored lights).
///
/// Input: planar f32 linear RGB where 1.0 = `source_intensity_target` nits.
/// Output: planar f32 linear RGB where 1.0 = `desired_intensity_target` nits.
///
/// The BT.2446 Method A tone curve is also implemented by libplacebo
/// (<https://github.com/haasn/libplacebo/blob/master/src/tone_mapping.c>).
///
/// The curve operates in gamma-2.4 / logarithmic domain:
/// 1. Gamma-encode luminance (BT.1886 OETF)
/// 2. Logarithmic compression normalized by source peak
/// 3. Piecewise knee curve (fixed BT.2446a coefficients)
/// 4. Inverse logarithmic expansion normalized by target peak
/// 5. Linearize (BT.1886 EOTF)
///
/// RGB channel ratios (hue/saturation) are preserved by scaling all channels
/// uniformly based on the luminance compression ratio.
#[derive(Debug)]
pub struct ToneMappingStage {
    first_channel: usize,
    source_intensity_target: f32,
    desired_intensity_target: f32,
    luminances: [f32; 3],
    /// ρ_HDR: perceptual peak of the HDR source, `1 + 32 * (source_it / 10000)^(1/2.4)`.
    rho_hdr: f32,
    /// ρ_SDR: perceptual peak of the SDR target, `1 + 32 * (desired_it / 10000)^(1/2.4)`.
    rho_sdr: f32,
    /// ln(ρ_HDR), precomputed for the log compression step.
    ln_rho_hdr: f32,
}

impl ToneMappingStage {
    pub fn new(
        first_channel: usize,
        source_intensity_target: f32,
        desired_intensity_target: f32,
        luminances: [f32; 3],
    ) -> Self {
        let rho_hdr = 1.0 + 32.0 * (source_intensity_target / 10000.0).powf(1.0 / 2.4);
        let rho_sdr = 1.0 + 32.0 * (desired_intensity_target / 10000.0).powf(1.0 / 2.4);
        let ln_rho_hdr = rho_hdr.ln();

        Self {
            first_channel,
            source_intensity_target,
            desired_intensity_target,
            luminances,
            rho_hdr,
            rho_sdr,
            ln_rho_hdr,
        }
    }

    /// Applies the BT.2446 Method A tone curve to a normalized luminance value.
    ///
    /// Input: linear luminance in [0, 1] where 1.0 = source_intensity_target nits.
    /// Output: linear luminance where 1.0 ≈ desired_intensity_target nits.
    #[inline]
    fn bt2446a_map(&self, y: f32) -> f32 {
        // Step 1: BT.1886 OETF (gamma encode)
        let mut x = y.powf(1.0 / 2.4);

        // Step 2: Logarithmic HDR compression → [0, 1]
        x = (1.0 + (self.rho_hdr - 1.0) * x).ln() / self.ln_rho_hdr;

        // Step 3: Piecewise knee curve (BT.2446a fixed coefficients)
        x = if x <= 0.7399 {
            1.0770 * x
        } else if x < 0.9909 {
            (-1.1510 * x + 2.7811) * x - 0.6302
        } else {
            0.5 * x + 0.5
        };

        // Step 4: Inverse logarithmic expansion
        x = (self.rho_sdr.powf(x) - 1.0) / (self.rho_sdr - 1.0);

        // Step 5: BT.1886 EOTF (linearize)
        x.powf(2.4)
    }
}

impl std::fmt::Display for ToneMappingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BT.2446a tone mapping {} -> {} nits on channels [{},{},{}]",
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

        let [lr, lg, lb] = self.luminances;

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
            let y_mapped = self.bt2446a_map(y_lin);

            // Scale RGB by luminance compression ratio (preserves hue/saturation)
            let ratio = y_mapped / y_lin;

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
    fn peak_maps_to_peak() -> Result<()> {
        // Source peak (1.0 linear = 10000 nits) should map to approximately
        // the output peak (1.0 in output space = 203 nits).
        let source_it = 10000.0_f32;
        let desired_it = 203.0_f32;

        let input_r = Image::new_with_value((1, 1), 1.0)?;
        let input_g = Image::new_with_value((1, 1), 1.0)?;
        let input_b = Image::new_with_value((1, 1), 1.0)?;

        let stage = ToneMappingStage::new(0, source_it, desired_it, LUMINANCE_BT2020);
        let output =
            make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

        let mapped = output[0].row(0)[0];
        // BT.2446a maps source peak to output peak.
        assert!(
            (mapped - 1.0).abs() < 0.02,
            "Source peak should map to ~1.0 output, got {mapped}"
        );

        Ok(())
    }

    #[test]
    fn highlights_compressed() -> Result<()> {
        // A very bright pixel (5000 nits) in a 10000-nit image should be
        // compressed: above midtones but well below naive linear scaling.
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
        // Should be well below the peak but not zero
        assert!(
            mapped > 0.1,
            "5000-nit pixel should map above 0.1, got {mapped}"
        );
        assert!(
            mapped < 1.0,
            "5000-nit pixel should be compressed below peak, got {mapped}"
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

    #[test]
    fn compression_reduces_absolute_nits() -> Result<()> {
        // BT.2446a compresses the dynamic range. A dim value (10 nits) in a
        // 10000-nit image should map to fewer nits in the output, even though
        // the normalized value may increase (different output scale).
        let source_it = 10000.0_f32;
        let desired_it = 203.0_f32;

        let stage = ToneMappingStage::new(0, source_it, desired_it, LUMINANCE_BT2020);

        let dim_nits = 10.0_f32;
        let dim_linear = dim_nits / source_it; // 0.001
        let mapped = stage.bt2446a_map(dim_linear);
        let mapped_nits = mapped * desired_it;

        // 10 nits should compress to fewer nits in the output
        assert!(
            mapped_nits < dim_nits,
            "10 nits should compress: {dim_nits} nits -> {mapped_nits} nits"
        );
        // But should still be positive
        assert!(
            mapped_nits > 0.0,
            "Output should be positive, got {mapped_nits} nits"
        );

        Ok(())
    }
}
