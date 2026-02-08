// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

mod bt2446a;
mod bt2446a_linear;
mod bt2446a_perceptual;
mod common;

use crate::api::JxlToneMappingMethod;
use crate::render::RenderPipelineInPlaceStage;

/// Tone maps HDR linear RGB to a lower intensity target.
///
/// Input/output: planar f32 linear RGB where 1.0 = `source_intensity_target` nits.
/// The tone curve redistributes perceptual contrast for SDR viewing while
/// preserving the scene-referred normalization (1.0 = source peak).
///
/// The method field selects between different BT.2446a tone mapping algorithms.
/// All methods preserve RGB channel ratios (hue/saturation) by scaling
/// channels uniformly based on a luminance compression ratio.
#[derive(Debug)]
pub struct ToneMappingStage {
    first_channel: usize,
    method: JxlToneMappingMethod,
    source_intensity_target: f32,
    desired_intensity_target: f32,
    luminances: [f32; 3],
    bt2446a: common::Bt2446aParams,
}

impl ToneMappingStage {
    pub fn new(
        first_channel: usize,
        source_intensity_target: f32,
        desired_intensity_target: f32,
        luminances: [f32; 3],
        method: JxlToneMappingMethod,
    ) -> Self {
        let bt2446a = common::Bt2446aParams::new(source_intensity_target, desired_intensity_target);

        Self {
            first_channel,
            method,
            source_intensity_target,
            desired_intensity_target,
            luminances,
            bt2446a,
        }
    }
}

impl std::fmt::Display for ToneMappingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} tone mapping {} -> {} nits on channels [{},{},{}]",
            self.method,
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

        match self.method {
            JxlToneMappingMethod::Bt2446aLinear => {
                bt2446a_linear::process_row(self, xsize, row_r, row_g, row_b);
            }
            JxlToneMappingMethod::Bt2446a => {
                bt2446a::process_row(self, xsize, row_r, row_g, row_b);
            }
            JxlToneMappingMethod::Bt2446aPerceptual => {
                bt2446a_perceptual::process_row(self, xsize, row_r, row_g, row_b);
            }
        }
    }
}

#[cfg(test)]
mod test;
