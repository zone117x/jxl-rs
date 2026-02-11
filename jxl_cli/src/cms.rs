// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Color Management System implementation using lcms2.

use jxl::api::{JxlCms, JxlCmsTransformer, JxlColorProfile, JxlTransferFunction};
use jxl::error::{Error, Result};
use lcms2::{
    AllowCache, ColorSpaceSignatureExt, Intent, PixelFormat, Profile, ThreadContext, Transform,
};

/// CMS implementation using Little CMS (lcms2).
pub struct Lcms2Cms;

impl JxlCms for Lcms2Cms {
    fn initialize_transforms(
        &self,
        n: usize,
        _max_pixels_per_transform: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        _intensity_target: f32,
    ) -> Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send>>)> {
        // Convert profiles to ICC
        let input_icc = input
            .try_as_icc()
            .ok_or_else(|| Error::CmsError("Cannot create ICC for input profile".into()))?;
        let output_icc = output
            .try_as_icc()
            .ok_or_else(|| Error::CmsError("Cannot create ICC for output profile".into()))?;

        // Parse profiles once to determine channel counts
        let temp_input_profile = Profile::new_icc(input_icc.as_slice())
            .map_err(|e| Error::CmsError(format!("lcms2 failed to parse input ICC: {e}")))?;
        let temp_output_profile = Profile::new_icc(output_icc.as_slice())
            .map_err(|e| Error::CmsError(format!("lcms2 failed to parse output ICC: {e}")))?;

        let input_channels = temp_input_profile.color_space().channels() as usize;
        let output_channels = temp_output_profile.color_space().channels() as usize;

        let input_format = channels_to_pixel_format(input_channels);
        let output_format = channels_to_pixel_format(output_channels);

        // Create transforms using ThreadContext for thread safety (implements Send).
        // Use u8 pixel type with PixelFormat describing the actual f32 data layout.
        let mut transforms: Vec<Box<dyn JxlCmsTransformer + Send>> = Vec::with_capacity(n);

        for _ in 0..n {
            let context = ThreadContext::new();

            // Create profiles with the thread context
            let input_profile = Profile::new_icc_context(&context, input_icc.as_slice())
                .map_err(|e| Error::CmsError(format!("lcms2 failed to parse input ICC: {e}")))?;
            let output_profile = Profile::new_icc_context(&context, output_icc.as_slice())
                .map_err(|e| Error::CmsError(format!("lcms2 failed to parse output ICC: {e}")))?;

            let transform: Transform<u8, u8, ThreadContext, AllowCache> = Transform::new_context(
                context,
                &input_profile,
                input_format,
                &output_profile,
                output_format,
                Intent::RelativeColorimetric,
            )
            .map_err(|e| Error::CmsError(format!("lcms2 failed to create transform: {e}")))?;

            transforms.push(Box::new(Lcms2Transformer {
                transform,
                input_channels,
                output_channels,
            }));
        }

        Ok((output_channels, transforms))
    }
}

/// Maps channel count to lcms2 PixelFormat for f32 data.
fn channels_to_pixel_format(channels: usize) -> PixelFormat {
    match channels {
        1 => PixelFormat::GRAY_FLT,
        3 => PixelFormat::RGB_FLT,
        4 => PixelFormat::CMYK_FLT,
        _ => PixelFormat::RGB_FLT, // Default to RGB
    }
}

/// Transformer implementation using lcms2 with ThreadContext for thread safety.
struct Lcms2Transformer {
    transform: Transform<u8, u8, ThreadContext, AllowCache>,
    input_channels: usize,
    output_channels: usize,
}

impl JxlCmsTransformer for Lcms2Transformer {
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let num_pixels = input.len() / self.input_channels;

        // Verify output buffer size
        let expected_output_len = num_pixels * self.output_channels;
        if output.len() < expected_output_len {
            return Err(Error::CmsError(format!(
                "Output buffer too small: expected {expected_output_len}, got {}",
                output.len()
            )));
        }

        // Convert f32 slices to byte slices using bytemuck for safe casting
        let input_bytes: &[u8] = bytemuck::cast_slice(input);
        let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(output);

        self.transform.transform_pixels(input_bytes, output_bytes);

        Ok(())
    }

    fn do_transform_inplace(&mut self, inout: &mut [f32]) -> Result<()> {
        // For in-place transform, input and output channel counts must match
        if self.input_channels != self.output_channels {
            return Err(Error::CmsError(
                "In-place transform requires matching channel counts".into(),
            ));
        }

        // Convert f32 slice to byte slice
        let inout_bytes: &mut [u8] = bytemuck::cast_slice_mut(inout);

        self.transform.transform_in_place(inout_bytes);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tone-mapping CMS: applies tone mapping then delegates to lcms2
// ---------------------------------------------------------------------------

#[cfg(feature = "tone-mapping")]
mod tone_mapping_cms {
    use super::*;
    use crate::tone_mapping::{
        Bt2446aParams, DEFAULT_SDR_INTENSITY_TARGET, Rec2408Params, ToneMapMethod,
        tone_map_bt2446a, tone_map_bt2446a_linear, tone_map_bt2446a_perceptual, tone_map_rec2408,
    };
    use jxl::api::{JxlColorEncoding, JxlPrimaries, JxlWhitePoint};

    /// CMS that applies tone mapping before delegating to lcms2 for color
    /// space conversion.  Supports all [`ToneMapMethod`] variants.
    pub struct ToneMappingLcms2Cms {
        /// Target display luminance in cd/m² (nits). Defaults to 203.
        pub desired_intensity_target: f32,
        /// Tone mapping algorithm.
        pub method: ToneMapMethod,
    }

    impl Default for ToneMappingLcms2Cms {
        fn default() -> Self {
            Self {
                desired_intensity_target: DEFAULT_SDR_INTENSITY_TARGET,
                method: ToneMapMethod::default(),
            }
        }
    }

    /// Per-method precomputed config stored in each transformer.
    #[derive(Clone, Copy)]
    enum ToneMapConfig {
        Bt2446a {
            params: Bt2446aParams,
            luminances: [f32; 3],
        },
        Bt2446aLinear {
            params: Bt2446aParams,
            luminances: [f32; 3],
        },
        Bt2446aPerceptual {
            params: Bt2446aParams,
            source_intensity_target: f32,
        },
        Rec2408 {
            params: Rec2408Params,
            luminances: [f32; 3],
        },
    }

    impl JxlCms for ToneMappingLcms2Cms {
        fn initialize_transforms(
            &self,
            n: usize,
            max_pixels_per_transform: usize,
            input: JxlColorProfile,
            output: JxlColorProfile,
            intensity_target: f32,
        ) -> Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send>>)> {
            let luminances = luminances_from_profile(&input);

            let config = if intensity_target > self.desired_intensity_target
                && self.desired_intensity_target > 0.0
            {
                let bt2446a =
                    || Bt2446aParams::new(intensity_target, self.desired_intensity_target);

                Some(match self.method {
                    ToneMapMethod::Bt2446a => ToneMapConfig::Bt2446a {
                        params: bt2446a(),
                        luminances,
                    },
                    ToneMapMethod::Bt2446aLinear => ToneMapConfig::Bt2446aLinear {
                        params: bt2446a(),
                        luminances,
                    },
                    ToneMapMethod::Bt2446aPerceptual => ToneMapConfig::Bt2446aPerceptual {
                        params: bt2446a(),
                        source_intensity_target: intensity_target,
                    },
                    ToneMapMethod::Rec2408 => ToneMapConfig::Rec2408 {
                        params: Rec2408Params::new(
                            [0.0, intensity_target],
                            [0.0, self.desired_intensity_target],
                        ),
                        luminances,
                    },
                    ToneMapMethod::CmsOnly => unreachable!("CmsOnly uses plain Lcms2Cms"),
                })
            } else {
                None
            };

            // For non-XYB images with PQ transfer function, pixel data arrives
            // PQ-encoded. Tone mapping expects linear input, so we need to decode
            // PQ→linear first and tell lcms2 the input is linear (not PQ).
            let pq_intensity_target = config.is_some()
                && input
                    .transfer_function()
                    .is_some_and(|tf| matches!(tf, JxlTransferFunction::PQ));
            let pq_intensity_target = if pq_intensity_target {
                Some(intensity_target)
            } else {
                None
            };

            let cms_input = if pq_intensity_target.is_some() {
                input.with_linear_tf().unwrap_or(input)
            } else {
                input
            };

            // Delegate to lcms2 for color space conversion.
            let (output_channels, lcms2_transforms) = Lcms2Cms.initialize_transforms(
                n,
                max_pixels_per_transform,
                cms_input,
                output,
                intensity_target,
            )?;

            let transforms: Vec<Box<dyn JxlCmsTransformer + Send>> = lcms2_transforms
                .into_iter()
                .map(|inner| -> Box<dyn JxlCmsTransformer + Send> {
                    Box::new(ToneMappingLcms2Transformer {
                        inner,
                        config,
                        pq_intensity_target,
                    })
                })
                .collect();

            Ok((output_channels, transforms))
        }
    }

    /// Transformer that applies tone mapping then delegates to lcms2.
    struct ToneMappingLcms2Transformer {
        inner: Box<dyn JxlCmsTransformer + Send>,
        config: Option<ToneMapConfig>,
        /// If set, input data is PQ-encoded and needs decoding to linear first.
        pq_intensity_target: Option<f32>,
    }

    impl JxlCmsTransformer for ToneMappingLcms2Transformer {
        fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
            if self.config.is_some() || self.pq_intensity_target.is_some() {
                output[..input.len()].copy_from_slice(input);
                if let Some(it) = self.pq_intensity_target {
                    jxl::color::tf::pq_to_linear_precise(it, &mut output[..input.len()]);
                }
                if let Some(config) = self.config {
                    tone_map_interleaved(config, &mut output[..input.len()]);
                }
                self.inner.do_transform_inplace(output)
            } else {
                self.inner.do_transform(input, output)
            }
        }

        fn do_transform_inplace(&mut self, inout: &mut [f32]) -> Result<()> {
            if let Some(it) = self.pq_intensity_target {
                jxl::color::tf::pq_to_linear_precise(it, inout);
            }
            if let Some(config) = self.config {
                tone_map_interleaved(config, inout);
            }
            self.inner.do_transform_inplace(inout)
        }
    }

    fn tone_map_interleaved(config: ToneMapConfig, data: &mut [f32]) {
        match config {
            ToneMapConfig::Bt2446a { params, luminances } => {
                tone_map_bt2446a(&params, luminances, data);
            }
            ToneMapConfig::Bt2446aLinear { params, luminances } => {
                tone_map_bt2446a_linear(&params, luminances, data);
            }
            ToneMapConfig::Bt2446aPerceptual {
                params,
                source_intensity_target,
            } => {
                tone_map_bt2446a_perceptual(&params, source_intensity_target, data);
            }
            ToneMapConfig::Rec2408 { params, luminances } => {
                tone_map_rec2408(&params, luminances, data);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Luminance derivation from color profile primaries
    // -----------------------------------------------------------------------

    /// Standard luminance coefficients for known primaries.
    fn luminances_from_primaries(primaries: &JxlPrimaries) -> [f32; 3] {
        match primaries {
            JxlPrimaries::BT2100 => [0.2627, 0.6780, 0.0593],
            JxlPrimaries::SRGB => [0.2126, 0.7152, 0.0722],
            JxlPrimaries::P3 => [0.2290, 0.6917, 0.0793],
            JxlPrimaries::Chromaticities {
                rx,
                ry,
                gx,
                gy,
                bx,
                by,
            } => luminances_from_chromaticities(
                *rx, *ry, *gx, *gy, *bx, *by, // Default to D65 white point
                0.3127, 0.3290,
            ),
        }
    }

    /// Derive luminance coefficients from the input profile.
    /// Falls back to BT.2020 for ICC profiles (the most common HDR primaries).
    fn luminances_from_profile(profile: &JxlColorProfile) -> [f32; 3] {
        match profile {
            JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
                primaries,
                white_point,
                ..
            }) => {
                if let JxlPrimaries::Chromaticities {
                    rx,
                    ry,
                    gx,
                    gy,
                    bx,
                    by,
                } = primaries
                {
                    let (wx, wy) = white_point_chromaticity(white_point);
                    luminances_from_chromaticities(*rx, *ry, *gx, *gy, *bx, *by, wx, wy)
                } else {
                    luminances_from_primaries(primaries)
                }
            }
            // ICC profiles or non-RGB: default to BT.2020 luminances
            _ => [0.2627, 0.6780, 0.0593],
        }
    }

    fn white_point_chromaticity(wp: &JxlWhitePoint) -> (f32, f32) {
        match wp {
            JxlWhitePoint::D65 => (0.3127, 0.3290),
            JxlWhitePoint::E => (1.0 / 3.0, 1.0 / 3.0),
            JxlWhitePoint::DCI => (0.314, 0.351),
            JxlWhitePoint::Chromaticity { wx, wy } => (*wx, *wy),
        }
    }

    /// Compute luminance coefficients from arbitrary chromaticity coordinates.
    ///
    /// Solves for the Y row of the RGB-to-XYZ matrix given:
    ///   - Red, green, blue chromaticity pairs (rx,ry), (gx,gy), (bx,by)
    ///   - White point chromaticity (wx, wy)
    ///
    /// The Y row gives [Yr, Yg, Yb] which are the luminance coefficients.
    #[allow(clippy::too_many_arguments)]
    fn luminances_from_chromaticities(
        rx: f32,
        ry: f32,
        gx: f32,
        gy: f32,
        bx: f32,
        by: f32,
        wx: f32,
        wy: f32,
    ) -> [f32; 3] {
        let rz = 1.0 - rx - ry;
        let gz = 1.0 - gx - gy;
        let bz = 1.0 - bx - by;

        let w_x = wx / wy;
        let w_y = 1.0;
        let w_z = (1.0 - wx - wy) / wy;

        let m00 = rx / ry;
        let m01 = gx / gy;
        let m02 = bx / by;
        let m10 = 1.0f32;
        let m11 = 1.0f32;
        let m12 = 1.0f32;
        let m20 = rz / ry;
        let m21 = gz / gy;
        let m22 = bz / by;

        let det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20);

        if det.abs() < 1e-10 {
            return [0.2627, 0.6780, 0.0593];
        }

        let inv_det = 1.0 / det;

        let sr = (w_x * (m11 * m22 - m12 * m21) - m01 * (w_y * m22 - m12 * w_z)
            + m02 * (w_y * m21 - m11 * w_z))
            * inv_det;
        let sg = (m00 * (w_y * m22 - m12 * w_z) - w_x * (m10 * m22 - m12 * m20)
            + m02 * (m10 * w_z - w_y * m20))
            * inv_det;
        let sb = (m00 * (m11 * w_z - w_y * m21) - m01 * (m10 * w_z - w_y * m20)
            + w_x * (m10 * m21 - m11 * m20))
            * inv_det;

        let sum = sr + sg + sb;
        if sum.abs() < 1e-10 {
            return [0.2627, 0.6780, 0.0593];
        }

        [sr / sum, sg / sum, sb / sum]
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_bt2020_luminances() {
            let lum = luminances_from_primaries(&JxlPrimaries::BT2100);
            assert!((lum[0] - 0.2627).abs() < 1e-4);
            assert!((lum[1] - 0.6780).abs() < 1e-4);
            assert!((lum[2] - 0.0593).abs() < 1e-4);
        }

        #[test]
        fn test_srgb_luminances_from_chromaticities() {
            let lum =
                luminances_from_chromaticities(0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290);
            assert!((lum[0] - 0.2126).abs() < 0.002, "R: {}", lum[0]);
            assert!((lum[1] - 0.7152).abs() < 0.002, "G: {}", lum[1]);
            assert!((lum[2] - 0.0722).abs() < 0.002, "B: {}", lum[2]);
        }

        #[test]
        fn test_bt2446a_black_unchanged() {
            let params = Bt2446aParams::new(10000.0, 203.0);
            let lum = [0.2627, 0.6780, 0.0593];
            let mut data = [0.0f32, 0.0, 0.0];
            tone_map_bt2446a(&params, lum, &mut data);
            assert_eq!(data, [0.0, 0.0, 0.0]);
        }

        #[test]
        fn test_bt2446a_boosts_midtones() {
            let params = Bt2446aParams::new(10000.0, 203.0);
            let lum = [0.2627, 0.6780, 0.0593];
            let mut data = [0.5f32, 0.5, 0.5];
            let original = data;
            tone_map_bt2446a(&params, lum, &mut data);
            assert!(
                data[0] > original[0],
                "should boost: {} vs {}",
                data[0],
                original[0]
            );
            assert!(data[0] <= 1.001, "should not exceed peak: {}", data[0]);
        }

        #[test]
        fn test_bt2446a_linear_black_unchanged() {
            let params = Bt2446aParams::new(10000.0, 203.0);
            let lum = [0.2627, 0.6780, 0.0593];
            let mut data = [0.0f32, 0.0, 0.0];
            tone_map_bt2446a_linear(&params, lum, &mut data);
            assert_eq!(data, [0.0, 0.0, 0.0]);
        }

        #[test]
        fn test_rec2408_black_unchanged() {
            let params = Rec2408Params::new([0.0, 10000.0], [0.0, 203.0]);
            let lum = [0.2627, 0.6780, 0.0593];
            let mut data = [0.0f32, 0.0, 0.0];
            tone_map_rec2408(&params, lum, &mut data);
            assert!(data[0].abs() < 1e-4, "R: {}", data[0]);
            assert!(data[1].abs() < 1e-4, "G: {}", data[1]);
            assert!(data[2].abs() < 1e-4, "B: {}", data[2]);
        }

        #[test]
        fn test_rec2408_compresses_highlights() {
            let params = Rec2408Params::new([0.0, 10000.0], [0.0, 203.0]);
            let lum = [0.2627, 0.6780, 0.0593];
            let mut data = [0.8f32, 0.8, 0.8];
            tone_map_rec2408(&params, lum, &mut data);
            // Rec2408 normalizes so 1.0 = target peak, so output can be > input
            // but should be finite and non-negative
            assert!(data[0].is_finite() && data[0] >= 0.0, "R: {}", data[0]);
        }

        #[test]
        fn test_perceptual_black_unchanged() {
            let params = Bt2446aParams::new(10000.0, 203.0);
            let mut data = [0.0f32, 0.0, 0.0];
            tone_map_bt2446a_perceptual(&params, 10000.0, &mut data);
            // PQ encode/decode round-trip introduces tiny residuals (~1e-6)
            assert!(data[0].abs() < 1e-5, "R: {}", data[0]);
            assert!(data[1].abs() < 1e-5, "G: {}", data[1]);
            assert!(data[2].abs() < 1e-5, "B: {}", data[2]);
        }
    }
}

#[cfg(feature = "tone-mapping")]
pub use tone_mapping_cms::ToneMappingLcms2Cms;

#[cfg(test)]
mod tests {
    use super::*;
    use jxl::api::{JxlColorEncoding, JxlPrimaries, JxlTransferFunction, JxlWhitePoint};
    use jxl::headers::color_encoding::RenderingIntent;

    fn srgb_profile() -> JxlColorProfile {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::SRGB,
            transfer_function: JxlTransferFunction::SRGB,
            rendering_intent: RenderingIntent::Relative,
        })
    }

    fn linear_srgb_profile() -> JxlColorProfile {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::SRGB,
            transfer_function: JxlTransferFunction::Linear,
            rendering_intent: RenderingIntent::Relative,
        })
    }

    #[test]
    fn test_create_transform() {
        let cms = Lcms2Cms;
        let result =
            cms.initialize_transforms(1, 1024, srgb_profile(), linear_srgb_profile(), 255.0);
        assert!(result.is_ok());
        let (output_channels, transforms) = result.unwrap();
        assert_eq!(output_channels, 3);
        assert_eq!(transforms.len(), 1);
    }

    #[test]
    fn test_transform_identity() {
        let cms = Lcms2Cms;
        let (_, mut transforms) = cms
            .initialize_transforms(1, 1024, srgb_profile(), srgb_profile(), 255.0)
            .unwrap();

        let input = [0.5f32, 0.5, 0.5]; // Gray
        let mut output = [0.0f32; 3];

        transforms[0].do_transform(&input, &mut output).unwrap();

        // Should be approximately the same (identity transform)
        for i in 0..3 {
            assert!(
                (input[i] - output[i]).abs() < 0.001,
                "Mismatch at {i}: {} vs {}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_transform_srgb_to_linear() {
        let cms = Lcms2Cms;
        let (_, mut transforms) = cms
            .initialize_transforms(1, 1024, srgb_profile(), linear_srgb_profile(), 255.0)
            .unwrap();

        // sRGB mid-gray (0.5) should map to approximately 0.214 in linear
        let input = [0.5f32, 0.5, 0.5];
        let mut output = [0.0f32; 3];

        transforms[0].do_transform(&input, &mut output).unwrap();

        // Linear value for sRGB 0.5 is approximately 0.214
        for (i, element) in output.iter().enumerate() {
            assert!(
                (element - 0.214).abs() < 0.01,
                "Output {i} = {}, expected ~0.214",
                element
            );
        }
    }
}
