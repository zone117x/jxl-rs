// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use test_log::test;

use super::*;
use crate::api::JxlToneMappingMethod;
use crate::error::Result;
use crate::image::Image;
use crate::render::test::make_and_run_simple_pipeline;
use crate::util::test::assert_all_almost_abs_eq;

const LUMINANCE_BT2020: [f32; 3] = [0.2627, 0.678, 0.0593];

fn make_stage(source_it: f32, desired_it: f32, method: JxlToneMappingMethod) -> ToneMappingStage {
    ToneMappingStage::new(0, source_it, desired_it, LUMINANCE_BT2020, method)
}

// ============================================================================
// Bt2446aLinear tests
// ============================================================================

#[test]
fn bt2446a_linear_consistency() -> Result<()> {
    crate::render::test::test_stage_consistency(
        || make_stage(10000.0, 203.0, JxlToneMappingMethod::Bt2446aLinear),
        (500, 500),
        3,
    )
}

#[test]
fn bt2446a_linear_peak_maps_to_peak() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;

    let input_r = Image::new_with_value((1, 1), 1.0)?;
    let input_g = Image::new_with_value((1, 1), 1.0)?;
    let input_b = Image::new_with_value((1, 1), 1.0)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446aLinear);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
    assert!(
        (mapped - 1.0).abs() < 0.02,
        "Source peak should map to ~1.0 output, got {mapped}"
    );

    Ok(())
}

#[test]
fn bt2446a_linear_highlights_compressed() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let bright_linear = 5000.0 / source_it;

    let input_r = Image::new_with_value((1, 1), bright_linear)?;
    let input_g = Image::new_with_value((1, 1), bright_linear)?;
    let input_b = Image::new_with_value((1, 1), bright_linear)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446aLinear);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
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
fn bt2446a_linear_black_unchanged() -> Result<()> {
    let input_r = Image::new_with_value((1, 1), 0.0)?;
    let input_g = Image::new_with_value((1, 1), 0.0)?;
    let input_b = Image::new_with_value((1, 1), 0.0)?;

    let stage = make_stage(10000.0, 203.0, JxlToneMappingMethod::Bt2446aLinear);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    assert_all_almost_abs_eq(output[0].row(0), &[0.0], 1e-6);
    assert_all_almost_abs_eq(output[1].row(0), &[0.0], 1e-6);
    assert_all_almost_abs_eq(output[2].row(0), &[0.0], 1e-6);

    Ok(())
}

#[test]
fn bt2446a_linear_color_ratios_preserved() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;

    let r_val = 0.1_f32;
    let g_val = 0.05_f32;
    let b_val = 0.02_f32;

    let input_r = Image::new_with_value((1, 1), r_val)?;
    let input_g = Image::new_with_value((1, 1), g_val)?;
    let input_b = Image::new_with_value((1, 1), b_val)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446aLinear);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let out_r = output[0].row(0)[0];
    let out_g = output[1].row(0)[0];
    let out_b = output[2].row(0)[0];

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
fn bt2446a_linear_monotonic_increasing() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = 0.0_f32;

    for &val in &test_values {
        let input_r = Image::new_with_value((1, 1), val)?;
        let input_g = Image::new_with_value((1, 1), val)?;
        let input_b = Image::new_with_value((1, 1), val)?;

        let s = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446aLinear);
        let output = make_and_run_simple_pipeline(s, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
fn bt2446a_linear_compression_reduces_absolute_nits() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446aLinear);

    let dim_nits = 10.0_f32;
    let dim_linear = dim_nits / source_it;
    let mapped = bt2446a_linear::bt2446a_map(stage.bt2446a(), dim_linear);
    let mapped_nits = mapped * desired_it;

    assert!(
        mapped_nits < dim_nits,
        "10 nits should compress: {dim_nits} nits -> {mapped_nits} nits"
    );
    assert!(
        mapped_nits > 0.0,
        "Output should be positive, got {mapped_nits} nits"
    );

    Ok(())
}

// ============================================================================
// Bt2446a (Y'CbCr') tests
// ============================================================================

#[test]
fn bt2446a_consistency() -> Result<()> {
    crate::render::test::test_stage_consistency(
        || make_stage(10000.0, 203.0, JxlToneMappingMethod::Bt2446a),
        (500, 500),
        3,
    )
}

#[test]
fn bt2446a_black_unchanged() -> Result<()> {
    let input_r = Image::new_with_value((1, 1), 0.0)?;
    let input_g = Image::new_with_value((1, 1), 0.0)?;
    let input_b = Image::new_with_value((1, 1), 0.0)?;

    let stage = make_stage(10000.0, 203.0, JxlToneMappingMethod::Bt2446a);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    assert_all_almost_abs_eq(output[0].row(0), &[0.0], 1e-6);
    assert_all_almost_abs_eq(output[1].row(0), &[0.0], 1e-6);
    assert_all_almost_abs_eq(output[2].row(0), &[0.0], 1e-6);

    Ok(())
}

#[test]
fn bt2446a_monotonic_increasing() -> Result<()> {
    // For neutral gray (R=G=B), brighter input should produce brighter output.
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = 0.0_f32;

    for &val in &test_values {
        let input_r = Image::new_with_value((1, 1), val)?;
        let input_g = Image::new_with_value((1, 1), val)?;
        let input_b = Image::new_with_value((1, 1), val)?;

        let s = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446a);
        let output = make_and_run_simple_pipeline(s, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
fn bt2446a_peak_maps_near_peak() -> Result<()> {
    // Source peak (1.0 linear = 10000 nits) should map near output peak.
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;

    let input_r = Image::new_with_value((1, 1), 1.0)?;
    let input_g = Image::new_with_value((1, 1), 1.0)?;
    let input_b = Image::new_with_value((1, 1), 1.0)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446a);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
    assert!(
        (mapped - 1.0).abs() < 0.05,
        "Source peak should map near ~1.0 output, got {mapped}"
    );

    Ok(())
}

/// Proves that the direct gamma-domain scaling `C'_out = C' · ratio` is
/// algebraically equivalent to the full Y'CbCr decomposition → scale → reconstruct
/// path from the BT.2446a spec, for a range of saturated and neutral colors.
#[test]
fn bt2446a_matches_ycbcr_roundtrip() {
    let [lr, lg, lb] = LUMINANCE_BT2020;
    let params = common::Bt2446aParams::new(10000.0, 203.0);

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

        let y_prime_mapped = common::bt2446a_knee(&params, y_prime);
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
fn bt2446a_perceptual_consistency() -> Result<()> {
    crate::render::test::test_stage_consistency(
        || make_stage(10000.0, 203.0, JxlToneMappingMethod::Bt2446aPerceptual),
        (500, 500),
        3,
    )
}

#[test]
fn bt2446a_perceptual_black_unchanged() -> Result<()> {
    let input_r = Image::new_with_value((1, 1), 0.0)?;
    let input_g = Image::new_with_value((1, 1), 0.0)?;
    let input_b = Image::new_with_value((1, 1), 0.0)?;

    let stage = make_stage(10000.0, 203.0, JxlToneMappingMethod::Bt2446aPerceptual);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    assert_all_almost_abs_eq(output[0].row(0), &[0.0], 1e-6);
    assert_all_almost_abs_eq(output[1].row(0), &[0.0], 1e-6);
    assert_all_almost_abs_eq(output[2].row(0), &[0.0], 1e-6);

    Ok(())
}

#[test]
fn bt2446a_perceptual_monotonic_increasing() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = 0.0_f32;

    for &val in &test_values {
        let input_r = Image::new_with_value((1, 1), val)?;
        let input_g = Image::new_with_value((1, 1), val)?;
        let input_b = Image::new_with_value((1, 1), val)?;

        let s = make_stage(
            source_it,
            desired_it,
            JxlToneMappingMethod::Bt2446aPerceptual,
        );
        let output = make_and_run_simple_pipeline(s, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
fn bt2446a_perceptual_highlights_compressed() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let bright_linear = 5000.0 / source_it;

    let input_r = Image::new_with_value((1, 1), bright_linear)?;
    let input_g = Image::new_with_value((1, 1), bright_linear)?;
    let input_b = Image::new_with_value((1, 1), bright_linear)?;

    let stage = make_stage(
        source_it,
        desired_it,
        JxlToneMappingMethod::Bt2446aPerceptual,
    );
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
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

// ============================================================================
// Rec2408 tests
// ============================================================================

#[test]
fn rec2408_consistency() -> Result<()> {
    crate::render::test::test_stage_consistency(
        || make_stage(10000.0, 203.0, JxlToneMappingMethod::Rec2408),
        (500, 500),
        3,
    )
}

#[test]
fn rec2408_black_unchanged() -> Result<()> {
    let input_r = Image::new_with_value((1, 1), 0.0)?;
    let input_g = Image::new_with_value((1, 1), 0.0)?;
    let input_b = Image::new_with_value((1, 1), 0.0)?;

    let stage = make_stage(10000.0, 203.0, JxlToneMappingMethod::Rec2408);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    // Near-black should map to near-black (black level lift may add a tiny offset).
    assert!(
        output[0].row(0)[0].abs() < 0.01,
        "R should be near zero, got {}",
        output[0].row(0)[0]
    );
    assert!(
        output[1].row(0)[0].abs() < 0.01,
        "G should be near zero, got {}",
        output[1].row(0)[0]
    );
    assert!(
        output[2].row(0)[0].abs() < 0.01,
        "B should be near zero, got {}",
        output[2].row(0)[0]
    );

    Ok(())
}

#[test]
fn rec2408_peak_maps_near_peak() -> Result<()> {
    // Source peak (1.0 at 10000 nits) should map to ~1.0 in target normalization.
    let input_r = Image::new_with_value((1, 1), 1.0)?;
    let input_g = Image::new_with_value((1, 1), 1.0)?;
    let input_b = Image::new_with_value((1, 1), 1.0)?;

    let stage = make_stage(10000.0, 203.0, JxlToneMappingMethod::Rec2408);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
    assert!(
        (mapped - 1.0).abs() < 0.05,
        "Source peak should map near ~1.0 in target normalization, got {mapped}"
    );

    Ok(())
}

#[test]
fn rec2408_monotonic_increasing() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = -1.0_f32;

    for &val in &test_values {
        let input_r = Image::new_with_value((1, 1), val)?;
        let input_g = Image::new_with_value((1, 1), val)?;
        let input_b = Image::new_with_value((1, 1), val)?;

        let s = make_stage(source_it, desired_it, JxlToneMappingMethod::Rec2408);
        let output = make_and_run_simple_pipeline(s, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
fn rec2408_highlights_compressed() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let bright_linear = 5000.0 / source_it;

    let input_r = Image::new_with_value((1, 1), bright_linear)?;
    let input_g = Image::new_with_value((1, 1), bright_linear)?;
    let input_b = Image::new_with_value((1, 1), bright_linear)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Rec2408);
    let output = make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
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
fn rec2408_gamut_map_clamps() -> Result<()> {
    // After Rec2408 + gamut map, all output channels should be in [0, 1].
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;

    // Test a variety of inputs including very saturated colors.
    let test_colors: &[[f32; 3]] = &[
        [1.0, 0.0, 0.0],   // pure red
        [0.0, 0.0, 1.0],   // pure blue
        [0.9, 0.01, 0.01], // near-monochromatic red
        [0.01, 0.01, 0.9], // near-monochromatic blue
        [1.0, 1.0, 1.0],   // white
        [0.5, 0.3, 0.1],   // typical warm color
    ];

    for &[rv, gv, bv] in test_colors {
        let input_r = Image::new_with_value((1, 1), rv)?;
        let input_g = Image::new_with_value((1, 1), gv)?;
        let input_b = Image::new_with_value((1, 1), bv)?;

        let s = make_stage(source_it, desired_it, JxlToneMappingMethod::Rec2408);
        let output = make_and_run_simple_pipeline(s, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

        let out_r = output[0].row(0)[0];
        let out_g = output[1].row(0)[0];
        let out_b = output[2].row(0)[0];

        assert!(
            (0.0..=1.0).contains(&out_r),
            "R out of [0,1] for input [{rv},{gv},{bv}]: got {out_r}"
        );
        assert!(
            (0.0..=1.0).contains(&out_g),
            "G out of [0,1] for input [{rv},{gv},{bv}]: got {out_g}"
        );
        assert!(
            (0.0..=1.0).contains(&out_b),
            "B out of [0,1] for input [{rv},{gv},{bv}]: got {out_b}"
        );
    }

    Ok(())
}

/// Validates that the Rec2408 pipeline stage produces the same results as
/// hand-computed reference values using the same math, for neutral gray.
#[test]
fn rec2408_matches_reference() {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let [lr, lg, lb] = LUMINANCE_BT2020;

    let params = rec2408::Rec2408Params::new([0.0, source_it], [0.0, desired_it]);

    // For neutral gray (R=G=B), gamut map is a no-op, so we can validate
    // the tone mapping math directly.
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];

    for &val in &test_values {
        // Hand-compute the expected output.
        let luminance = source_it * (lr * val + lg * val + lb * val);
        let pq = rec2408::pq_encode_nits(luminance);
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
        let new_luminance = rec2408::pq_decode_nits(e4).clamp(0.0, params.target_peak);

        let multiplier = (new_luminance / luminance) * params.normalizer;
        let expected = val * multiplier;

        // Run through the actual stage.
        let mut row_r = vec![val];
        let mut row_g = vec![val];
        let mut row_b = vec![val];
        let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Rec2408);
        rec2408::process_row(&stage, 1, &mut row_r, &mut row_g, &mut row_b);

        let eps = 1e-4;
        assert!(
            (row_r[0] - expected).abs() < eps,
            "Mismatch at val={val}: stage={}, expected={expected}",
            row_r[0]
        );
    }
}
