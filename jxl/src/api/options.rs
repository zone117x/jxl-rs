// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlCms;

/// Standard SDR reference white per ITU-R BT.2408 (cd/m² / nits).
pub const DEFAULT_SDR_INTENSITY_TARGET: f32 = 203.0;

/// Tone mapping algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JxlToneMappingMethod {
    /// BT.2446a in Y'CbCr' domain per ITU-R BT.2446-1.
    /// Gamma-encodes, converts to YCbCr, applies curve to Y', scales CbCr, converts back.
    Bt2446a,
    /// BT.2446a curve applied to linear RGB luminance. Fast approximation —
    /// same curve but luminance is computed in linear domain instead of Y'CbCr'.
    Bt2446aLinear,
    /// BT.2446a curve in IPTPQc4 perceptual space (libplacebo-style).
    /// Best color preservation for saturated HDR content.
    #[default]
    Bt2446aPerceptual,
    /// Rec. 2408 / BT.2390-style tone mapping matching libjxl's Rec2408ToneMapperBase.
    /// Operates in PQ domain with Hermite spline knee, followed by gamut mapping.
    /// Output is re-normalized so 1.0 = target peak (unlike BT.2446a variants).
    Rec2408,
}

impl JxlToneMappingMethod {
    /// Returns the default target display luminance (nits) for this method.
    ///
    /// - `Rec2408`: 255 nits, matching libjxl's render pipeline default.
    /// - BT.2446a variants: 203 nits (ITU-R BT.2408 SDR reference white).
    pub fn default_intensity_target(self) -> f32 {
        match self {
            Self::Rec2408 => 255.0,
            Self::Bt2446a | Self::Bt2446aLinear | Self::Bt2446aPerceptual => {
                DEFAULT_SDR_INTENSITY_TARGET
            }
        }
    }
}

/// Options for HDR→SDR tone mapping.
#[derive(Debug, Clone, Copy)]
pub struct JxlToneMappingOptions {
    /// Target display luminance in cd/m² (nits).
    /// `None` defaults per method via [`JxlToneMappingMethod::default_intensity_target`].
    pub desired_intensity_target: Option<f32>,
    /// Tone mapping algorithm to use.
    pub method: JxlToneMappingMethod,
}

pub enum JxlProgressiveMode {
    /// Renders all pixels in every call to Process.
    Eager,
    /// Renders pixels once passes are completed.
    Pass,
    /// Renders pixels only once the final frame is ready.
    FullFrame,
}

#[non_exhaustive]
pub struct JxlDecoderOptions {
    pub adjust_orientation: bool,
    pub render_spot_colors: bool,
    pub coalescing: bool,
    /// HDR→SDR tone mapping options. `None` disables tone mapping (default).
    pub tone_mapping: Option<JxlToneMappingOptions>,
    pub skip_preview: bool,
    pub progressive_mode: JxlProgressiveMode,
    pub cms: Option<Box<dyn JxlCms>>,
    /// Fail decoding images with more than this number of pixels, or with frames with
    /// more than this number of pixels. The limit counts the product of pixels and
    /// channels, so for example an image with 1 extra channel of size 1024x1024 has 4
    /// million pixels.
    pub pixel_limit: Option<usize>,
    /// Use high precision mode for decoding.
    /// When false (default), uses lower precision settings that match libjxl's default.
    /// When true, uses higher precision at the cost of performance.
    ///
    /// This affects multiple decoder decisions including spline rendering precision
    /// and potentially intermediate buffer storage (e.g., using f32 vs f16).
    pub high_precision: bool,
    /// If true, multiply RGB by alpha before writing to output buffer.
    /// This produces premultiplied alpha output, which is useful for compositing.
    /// Default: false (output straight alpha)
    pub premultiply_output: bool,
}

impl Default for JxlDecoderOptions {
    fn default() -> Self {
        Self {
            adjust_orientation: true,
            render_spot_colors: true,
            coalescing: true,
            skip_preview: true,
            tone_mapping: None,
            progressive_mode: JxlProgressiveMode::Pass,
            cms: None,
            pixel_limit: None,
            high_precision: false,
            premultiply_output: false,
        }
    }
}
