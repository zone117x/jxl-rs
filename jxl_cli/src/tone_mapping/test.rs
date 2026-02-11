// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;

const LUMINANCE_BT2020: [f32; 3] = [0.2627, 0.678, 0.0593];

/// Helper: apply tone mapping to a single pixel via the interleaved function.
fn apply(f: impl Fn(&mut [f32]), pixel: &mut [f32; 3]) {
    f(pixel.as_mut_slice());
}

// ============================================================================
// Bt2446aLinear tests
// ============================================================================

#[test]
fn bt2446a_linear_peak_maps_to_peak() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let mut pixel = [1.0, 1.0, 1.0];
    apply(
        |d| tone_map_bt2446a_linear(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    assert!(
        (pixel[0] - 1.0).abs() < 0.02,
        "Source peak should map to ~1.0 output, got {}",
        pixel[0]
    );
}

#[test]
fn bt2446a_linear_highlights_compressed() {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let bright_linear = 5000.0 / source_it;
    let params = Bt2446aParams::new(source_it, desired_it);

    let mut pixel = [bright_linear, bright_linear, bright_linear];
    apply(
        |d| tone_map_bt2446a_linear(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    assert!(
        pixel[0] > 0.1,
        "5000-nit pixel should map above 0.1, got {}",
        pixel[0]
    );
    assert!(
        pixel[0] < 1.0,
        "5000-nit pixel should be compressed below peak, got {}",
        pixel[0]
    );
}

#[test]
fn bt2446a_linear_black_unchanged() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let mut pixel = [0.0, 0.0, 0.0];
    apply(
        |d| tone_map_bt2446a_linear(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    assert!(pixel[0].abs() < 1e-6);
    assert!(pixel[1].abs() < 1e-6);
    assert!(pixel[2].abs() < 1e-6);
}

#[test]
fn bt2446a_linear_color_ratios_preserved() {
    let params = Bt2446aParams::new(10000.0, 203.0);

    let r_val = 0.1_f32;
    let g_val = 0.05_f32;
    let b_val = 0.02_f32;

    let mut pixel = [r_val, g_val, b_val];
    apply(
        |d| tone_map_bt2446a_linear(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    let in_ratio_rg = r_val / g_val;
    let out_ratio_rg = pixel[0] / pixel[1];
    assert!(
        (in_ratio_rg - out_ratio_rg).abs() < 1e-4,
        "R:G ratio should be preserved: in={in_ratio_rg}, out={out_ratio_rg}"
    );

    let in_ratio_rb = r_val / b_val;
    let out_ratio_rb = pixel[0] / pixel[2];
    assert!(
        (in_ratio_rb - out_ratio_rb).abs() < 1e-4,
        "R:B ratio should be preserved: in={in_ratio_rb}, out={out_ratio_rb}"
    );
}

#[test]
fn bt2446a_linear_monotonic_increasing() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = 0.0_f32;

    for &val in &test_values {
        let mut pixel = [val, val, val];
        apply(
            |d| tone_map_bt2446a_linear(&params, LUMINANCE_BT2020, d),
            &mut pixel,
        );

        assert!(
            pixel[0] > prev_output,
            "Tone map should be monotonic: input {val} → {}, but prev was {prev_output}",
            pixel[0]
        );
        prev_output = pixel[0];
    }
}

#[test]
fn bt2446a_linear_compression_reduces_absolute_nits() {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let params = Bt2446aParams::new(source_it, desired_it);

    let dim_nits = 10.0_f32;
    let dim_linear = dim_nits / source_it;
    let mapped = bt2446a_map(&params, dim_linear);
    let mapped_nits = mapped * desired_it;

    assert!(
        mapped_nits < dim_nits,
        "10 nits should compress: {dim_nits} nits -> {mapped_nits} nits"
    );
    assert!(
        mapped_nits > 0.0,
        "Output should be positive, got {mapped_nits} nits"
    );
}

// ============================================================================
// Bt2446a (Y'CbCr') tests
// ============================================================================

#[test]
fn bt2446a_black_unchanged() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let mut pixel = [0.0, 0.0, 0.0];
    apply(
        |d| tone_map_bt2446a(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    assert!(pixel[0].abs() < 1e-6);
    assert!(pixel[1].abs() < 1e-6);
    assert!(pixel[2].abs() < 1e-6);
}

#[test]
fn bt2446a_monotonic_increasing() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = 0.0_f32;

    for &val in &test_values {
        let mut pixel = [val, val, val];
        apply(
            |d| tone_map_bt2446a(&params, LUMINANCE_BT2020, d),
            &mut pixel,
        );

        assert!(
            pixel[0] > prev_output,
            "Tone map should be monotonic: input {val} → {}, but prev was {prev_output}",
            pixel[0]
        );
        prev_output = pixel[0];
    }
}

#[test]
fn bt2446a_peak_maps_near_peak() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let mut pixel = [1.0, 1.0, 1.0];
    apply(
        |d| tone_map_bt2446a(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    assert!(
        (pixel[0] - 1.0).abs() < 0.05,
        "Source peak should map near ~1.0 output, got {}",
        pixel[0]
    );
}

/// Proves that the direct gamma-domain scaling `C'_out = C' · ratio` is
/// algebraically equivalent to the full Y'CbCr decomposition → scale → reconstruct
/// path from the BT.2446a spec, for a range of saturated and neutral colors.
#[test]
fn bt2446a_matches_ycbcr_roundtrip() {
    let [lr, lg, lb] = LUMINANCE_BT2020;
    let params = Bt2446aParams::new(10000.0, 203.0);

    // Test a variety of colors: neutrals, primaries, secondaries, and mixed.
    let test_colors: &[[f32; 3]] = &[
        [1.0, 1.0, 1.0],     // white
        [0.5, 0.5, 0.5],     // mid gray
        [0.01, 0.01, 0.01],  // near black
        [1.0, 0.0, 0.0],     // pure red
        [0.0, 1.0, 0.0],     // pure green
        [0.0, 0.0, 1.0],     // pure blue
        [1.0, 1.0, 0.0],     // yellow
        [0.0, 1.0, 1.0],     // cyan
        [1.0, 0.0, 1.0],     // magenta
        [0.8, 0.2, 0.05],    // saturated warm
        [0.05, 0.3, 0.9],    // saturated cool
        [0.001, 0.5, 0.001], // near-monochromatic green
    ];

    for &[r, g, b] in test_colors {
        let r_prime = r.max(0.0_f32).powf(1.0 / 2.4);
        let g_prime = g.max(0.0_f32).powf(1.0 / 2.4);
        let b_prime = b.max(0.0_f32).powf(1.0 / 2.4);

        let y_prime = lr * r_prime + lg * g_prime + lb * b_prime;
        if y_prime <= 0.0 {
            continue;
        }

        let y_prime_mapped = bt2446a_knee(&params, y_prime);
        let ratio = y_prime_mapped / y_prime;

        // --- Optimized path: direct scaling ---
        let opt_r = (r_prime * ratio).max(0.0).powf(2.4);
        let opt_g = (g_prime * ratio).max(0.0).powf(2.4);
        let opt_b = (b_prime * ratio).max(0.0).powf(2.4);

        // --- Full Y'CbCr round-trip (the spec's decomposition) ---
        let cb = (b_prime - y_prime) / (2.0 * (1.0 - lb));
        let cr = (r_prime - y_prime) / (2.0 * (1.0 - lr));
        let cb_out = cb * ratio;
        let cr_out = cr * ratio;
        let ycbcr_r = (y_prime_mapped + 2.0 * (1.0 - lr) * cr_out)
            .max(0.0)
            .powf(2.4);
        let ycbcr_g = (y_prime_mapped
            - 2.0 * (1.0 - lb) * (lb / lg) * cb_out
            - 2.0 * (1.0 - lr) * (lr / lg) * cr_out)
            .max(0.0)
            .powf(2.4);
        let ycbcr_b = (y_prime_mapped + 2.0 * (1.0 - lb) * cb_out)
            .max(0.0)
            .powf(2.4);

        // The two paths must be identical (within f32 rounding).
        let eps = 1e-6;
        assert!(
            (opt_r - ycbcr_r).abs() < eps,
            "R mismatch for [{r},{g},{b}]: optimized={opt_r}, ycbcr={ycbcr_r}"
        );
        assert!(
            (opt_g - ycbcr_g).abs() < eps,
            "G mismatch for [{r},{g},{b}]: optimized={opt_g}, ycbcr={ycbcr_g}"
        );
        assert!(
            (opt_b - ycbcr_b).abs() < eps,
            "B mismatch for [{r},{g},{b}]: optimized={opt_b}, ycbcr={ycbcr_b}"
        );
    }
}

// ============================================================================
// Bt2446aPerceptual (IPTPQc4) tests
// ============================================================================

#[test]
fn bt2446a_perceptual_black_unchanged() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let mut pixel = [0.0, 0.0, 0.0];
    apply(
        |d| tone_map_bt2446a_perceptual(&params, 10000.0, d),
        &mut pixel,
    );

    // PQ encode/decode round-trip introduces tiny residuals (~1e-6)
    assert!(pixel[0].abs() < 1e-5, "R: {}", pixel[0]);
    assert!(pixel[1].abs() < 1e-5, "G: {}", pixel[1]);
    assert!(pixel[2].abs() < 1e-5, "B: {}", pixel[2]);
}

#[test]
fn bt2446a_perceptual_monotonic_increasing() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = 0.0_f32;

    for &val in &test_values {
        let mut pixel = [val, val, val];
        apply(
            |d| tone_map_bt2446a_perceptual(&params, 10000.0, d),
            &mut pixel,
        );

        assert!(
            pixel[0] > prev_output,
            "Tone map should be monotonic: input {val} → {}, but prev was {prev_output}",
            pixel[0]
        );
        prev_output = pixel[0];
    }
}

#[test]
fn bt2446a_perceptual_highlights_compressed() {
    let params = Bt2446aParams::new(10000.0, 203.0);
    let bright_linear = 5000.0 / 10000.0;

    let mut pixel = [bright_linear, bright_linear, bright_linear];
    apply(
        |d| tone_map_bt2446a_perceptual(&params, 10000.0, d),
        &mut pixel,
    );

    assert!(
        pixel[0] > 0.1,
        "5000-nit pixel should map above 0.1, got {}",
        pixel[0]
    );
    assert!(
        pixel[0] < 1.0,
        "5000-nit pixel should be compressed below peak, got {}",
        pixel[0]
    );
}

// ============================================================================
// Rec2408 tests
// ============================================================================

#[test]
fn rec2408_black_unchanged() {
    let params = Rec2408Params::new([0.0, 10000.0], [0.0, 203.0]);
    let mut pixel = [0.0, 0.0, 0.0];
    apply(
        |d| tone_map_rec2408(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    // Near-black should map to near-black (black level lift may add a tiny offset).
    assert!(
        pixel[0].abs() < 0.01,
        "R should be near zero, got {}",
        pixel[0]
    );
    assert!(
        pixel[1].abs() < 0.01,
        "G should be near zero, got {}",
        pixel[1]
    );
    assert!(
        pixel[2].abs() < 0.01,
        "B should be near zero, got {}",
        pixel[2]
    );
}

#[test]
fn rec2408_peak_maps_near_peak() {
    let params = Rec2408Params::new([0.0, 10000.0], [0.0, 203.0]);
    let mut pixel = [1.0, 1.0, 1.0];
    apply(
        |d| tone_map_rec2408(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    // Rec2408 re-normalizes so 1.0 = target peak.
    // Source peak should map near ~1.0 (after normalizer and gamut map clamping).
    assert!(
        (pixel[0] - 1.0).abs() < 0.05,
        "Source peak should map near ~1.0 output, got {}",
        pixel[0]
    );
}

#[test]
fn rec2408_monotonic_increasing() {
    let params = Rec2408Params::new([0.0, 10000.0], [0.0, 203.0]);
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = -1.0_f32;

    for &val in &test_values {
        let mut pixel = [val, val, val];
        apply(
            |d| tone_map_rec2408(&params, LUMINANCE_BT2020, d),
            &mut pixel,
        );

        assert!(
            pixel[0] > prev_output,
            "Tone map should be monotonic: input {val} → {}, but prev was {prev_output}",
            pixel[0]
        );
        prev_output = pixel[0];
    }
}

#[test]
fn rec2408_highlights_compressed() {
    let params = Rec2408Params::new([0.0, 10000.0], [0.0, 203.0]);
    let bright_linear = 5000.0 / 10000.0;

    let mut pixel = [bright_linear, bright_linear, bright_linear];
    apply(
        |d| tone_map_rec2408(&params, LUMINANCE_BT2020, d),
        &mut pixel,
    );

    assert!(
        pixel[0] > 0.1,
        "5000-nit pixel should map above 0.1, got {}",
        pixel[0]
    );
    assert!(
        pixel[0] < 1.0,
        "5000-nit pixel should be compressed below peak, got {}",
        pixel[0]
    );
}

#[test]
fn rec2408_gamut_map_clamps() {
    let params = Rec2408Params::new([0.0, 10000.0], [0.0, 203.0]);

    // After Rec2408 + gamut map, all output channels should be in [0, 1].
    let test_colors: &[[f32; 3]] = &[
        [1.0, 0.0, 0.0],   // pure red
        [0.0, 0.0, 1.0],   // pure blue
        [0.9, 0.01, 0.01], // near-monochromatic red
        [0.01, 0.01, 0.9], // near-monochromatic blue
        [1.0, 1.0, 1.0],   // white
        [0.5, 0.3, 0.1],   // typical warm color
    ];

    for &[rv, gv, bv] in test_colors {
        let mut pixel = [rv, gv, bv];
        apply(
            |d| tone_map_rec2408(&params, LUMINANCE_BT2020, d),
            &mut pixel,
        );

        assert!(
            (0.0..=1.0).contains(&pixel[0]),
            "R out of [0,1] for input [{rv},{gv},{bv}]: got {}",
            pixel[0]
        );
        assert!(
            (0.0..=1.0).contains(&pixel[1]),
            "G out of [0,1] for input [{rv},{gv},{bv}]: got {}",
            pixel[1]
        );
        assert!(
            (0.0..=1.0).contains(&pixel[2]),
            "B out of [0,1] for input [{rv},{gv},{bv}]: got {}",
            pixel[2]
        );
    }
}

/// Validates that the Rec2408 tone mapping produces the same results as
/// hand-computed reference values using the same math, for neutral gray.
#[test]
fn rec2408_matches_reference() {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let [lr, lg, lb] = LUMINANCE_BT2020;

    let params = Rec2408Params::new([0.0, source_it], [0.0, desired_it]);

    // For neutral gray (R=G=B), gamut map is a no-op, so we can validate
    // the tone mapping math directly.
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];

    for &val in &test_values {
        // Hand-compute the expected output.
        let luminance = source_it * (lr * val + lg * val + lb * val);
        let pq = pq_encode_nits(luminance);
        let normalized_pq =
            ((pq - params.pq_mastering_min) * params.inv_pq_mastering_range).min(1.0);

        let e2 = if normalized_pq < params.ks {
            normalized_pq
        } else {
            params.hermite_spline(normalized_pq)
        };

        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let e3 = params.min_lum * (one_minus_e2_2 * one_minus_e2_2) + e2;
        let e4 = e3 * params.pq_mastering_range + params.pq_mastering_min;
        let new_luminance = pq_decode_nits(e4).clamp(0.0, params.target_peak);

        let multiplier = (new_luminance / luminance) * params.normalizer;
        let expected = val * multiplier;

        // Run through the actual function.
        let mut pixel = [val, val, val];
        apply(
            |d| tone_map_rec2408(&params, LUMINANCE_BT2020, d),
            &mut pixel,
        );

        let eps = 1e-4;
        assert!(
            (pixel[0] - expected).abs() < eps,
            "Mismatch at val={val}: actual={}, expected={expected}",
            pixel[0]
        );
    }
}

// ============================================================================
// PQ round-trip tests
// ============================================================================

#[test]
fn pq_round_trip() {
    let test_nits = [0.0, 0.001, 1.0, 100.0, 203.0, 1000.0, 4000.0, 10000.0];

    for &nits in &test_nits {
        let encoded = pq_encode_nits(nits);
        let decoded = pq_decode_nits(encoded);
        let eps = nits * 1e-4 + 1e-6; // relative + absolute epsilon
        assert!(
            (decoded - nits).abs() < eps,
            "PQ round-trip failed for {nits} nits: encoded={encoded}, decoded={decoded}"
        );
    }
}

// ============================================================================
// Matrix inverse tests
// ============================================================================

#[test]
fn rgb_to_lms_inverse_roundtrip() {
    let test_vectors: &[[f32; 3]] = &[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.3, 0.8],
    ];

    for &v in test_vectors {
        let lms = mat_mul(&RGB_TO_LMS, v);
        let rgb = mat_mul(&LMS_TO_RGB, lms);

        let eps = 1e-4;
        assert!(
            (rgb[0] - v[0]).abs() < eps
                && (rgb[1] - v[1]).abs() < eps
                && (rgb[2] - v[2]).abs() < eps,
            "RGB→LMS→RGB roundtrip failed for {v:?}: got {rgb:?}"
        );
    }
}

#[test]
fn lms_pq_to_ipt_inverse_roundtrip() {
    let test_vectors: &[[f32; 3]] = &[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.3, 0.8],
    ];

    for &v in test_vectors {
        let ipt = mat_mul(&LMS_PQ_TO_IPT, v);
        let lms = mat_mul(&IPT_TO_LMS_PQ, ipt);

        let eps = 1e-4;
        assert!(
            (lms[0] - v[0]).abs() < eps
                && (lms[1] - v[1]).abs() < eps
                && (lms[2] - v[2]).abs() < eps,
            "LMS_PQ→IPT→LMS_PQ roundtrip failed for {v:?}: got {lms:?}"
        );
    }
}
