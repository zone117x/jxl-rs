// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! HDR→SDR tone mapping algorithms and supporting math.

pub mod bt2446a;
pub mod bt2446a_linear;
pub mod bt2446a_perceptual;
pub mod common;
pub mod rec2408;

pub use bt2446a::tone_map_bt2446a;
pub use bt2446a_linear::{bt2446a_map, tone_map_bt2446a_linear};
pub use bt2446a_perceptual::{
    IPT_TO_LMS_PQ, LMS_PQ_TO_IPT, LMS_TO_RGB, RGB_TO_LMS, mat_mul, tone_map_bt2446a_perceptual,
};
pub use common::{Bt2446aParams, bt2446a_knee};
pub use rec2408::{Rec2408Params, gamut_map, pq_decode_nits, pq_encode_nits, tone_map_rec2408};

use clap::ValueEnum;

/// Standard SDR reference white per ITU-R BT.2408 (cd/m² / nits).
pub const DEFAULT_SDR_INTENSITY_TARGET: f32 = 203.0;

/// Tone mapping algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum ToneMapMethod {
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
    /// No tone mapping — just convert to sRGB via lcms2.
    /// Useful for comparing raw CMS output against tone-mapped results.
    CmsOnly,
}

impl ToneMapMethod {
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
            Self::CmsOnly => DEFAULT_SDR_INTENSITY_TARGET,
        }
    }
}

#[cfg(test)]
mod test;
