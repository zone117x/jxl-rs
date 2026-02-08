// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// Precomputed BT.2446 Method A parameters, shared across methods that use
/// the same log-compress → knee → inverse-log curve.
#[derive(Debug, Clone, Copy)]
pub(super) struct Bt2446aParams {
    /// ρ_HDR: perceptual peak of the HDR source, `1 + 32 * (source_it / 10000)^(1/2.4)`.
    pub rho_hdr: f32,
    /// ρ_SDR: perceptual peak of the SDR target, `1 + 32 * (desired_it / 10000)^(1/2.4)`.
    pub rho_sdr: f32,
    /// ln(ρ_HDR), precomputed for the log compression step.
    pub ln_rho_hdr: f32,
}

impl Bt2446aParams {
    pub fn new(source_intensity_target: f32, desired_intensity_target: f32) -> Self {
        let rho_hdr = 1.0 + 32.0 * (source_intensity_target / 10000.0).powf(1.0 / 2.4);
        let rho_sdr = 1.0 + 32.0 * (desired_intensity_target / 10000.0).powf(1.0 / 2.4);
        let ln_rho_hdr = rho_hdr.ln();
        Self {
            rho_hdr,
            rho_sdr,
            ln_rho_hdr,
        }
    }
}

/// BT.2446a knee curve: log-compress → piecewise knee → inverse-log.
///
/// Input/output are in gamma-encoded (perceptual) domain — no gamma
/// encode/decode is performed. Use this when the input is already
/// gamma-encoded (e.g. Y' from Y'CbCr', or I from IPTPQc4).
#[inline]
pub(super) fn bt2446a_knee(params: &Bt2446aParams, y_prime: f32) -> f32 {
    // Logarithmic HDR compression → [0, 1]
    let mut x = (1.0 + (params.rho_hdr - 1.0) * y_prime).ln() / params.ln_rho_hdr;

    // Piecewise knee curve (BT.2446a fixed coefficients)
    x = if x <= 0.7399 {
        1.0770 * x
    } else if x < 0.9909 {
        (-1.1510 * x + 2.7811) * x - 0.6302
    } else {
        0.5 * x + 0.5
    };

    // Inverse logarithmic expansion
    (params.rho_sdr.powf(x) - 1.0) / (params.rho_sdr - 1.0)
}

/// Full BT.2446a tone curve for linear luminance values.
///
/// Wraps `bt2446a_knee` with gamma encode (step 1) and linearize (step 5):
/// 1. BT.1886 OETF (gamma encode): `y^(1/2.4)`
/// 2–4. `bt2446a_knee` (log compress → knee → inverse log)
/// 5. BT.1886 EOTF (linearize): `x^2.4`
#[inline]
pub(super) fn bt2446a_map(params: &Bt2446aParams, y: f32) -> f32 {
    let y_prime = y.powf(1.0 / 2.4);
    bt2446a_knee(params, y_prime).powf(2.4)
}
