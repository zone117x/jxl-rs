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
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
fn bt2446a_linear_compression_reduces_absolute_nits() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446aLinear);

    let dim_nits = 10.0_f32;
    let dim_linear = dim_nits / source_it;
    let mapped = common::bt2446a_map(&stage.bt2446a, dim_linear);
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
// Reinhard tests
// ============================================================================

#[test]
fn reinhard_consistency() -> Result<()> {
    crate::render::test::test_stage_consistency(
        || make_stage(10000.0, 203.0, JxlToneMappingMethod::Reinhard),
        (500, 500),
        3,
    )
}

#[test]
fn reinhard_sdr_white_preserved() -> Result<()> {
    // 203 nits (SDR reference white) should map to ~1.0 in output space.
    // Reinhard preserves content at or below the knee point exactly.
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let sdr_white_linear = desired_it / source_it;

    let input_r = Image::new_with_value((1, 1), sdr_white_linear)?;
    let input_g = Image::new_with_value((1, 1), sdr_white_linear)?;
    let input_b = Image::new_with_value((1, 1), sdr_white_linear)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Reinhard);
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
    assert!(
        (mapped - 1.0).abs() < 0.02,
        "SDR white (203 nits) should map to ~1.0, got {mapped}"
    );

    Ok(())
}

#[test]
fn reinhard_highlights_compressed_not_clipped() -> Result<()> {
    // 5000 nits should be compressed above SDR white but well below naive scaling.
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let bright_linear = 5000.0 / source_it;

    let input_r = Image::new_with_value((1, 1), bright_linear)?;
    let input_g = Image::new_with_value((1, 1), bright_linear)?;
    let input_b = Image::new_with_value((1, 1), bright_linear)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Reinhard);
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
    assert!(
        mapped > 1.0,
        "5000-nit pixel should map above 1.0, got {mapped}"
    );
    assert!(
        mapped < 10.0,
        "5000-nit pixel should be compressed (naive ≈ 24.6, got {mapped})"
    );

    Ok(())
}

#[test]
fn reinhard_black_unchanged() -> Result<()> {
    let input_r = Image::new_with_value((1, 1), 0.0)?;
    let input_g = Image::new_with_value((1, 1), 0.0)?;
    let input_b = Image::new_with_value((1, 1), 0.0)?;

    let stage = make_stage(10000.0, 203.0, JxlToneMappingMethod::Reinhard);
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    assert_all_almost_abs_eq(output[0].row(0), &[0.0], 1e-6);
    assert_all_almost_abs_eq(output[1].row(0), &[0.0], 1e-6);
    assert_all_almost_abs_eq(output[2].row(0), &[0.0], 1e-6);

    Ok(())
}

#[test]
fn reinhard_monotonic_increasing() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let test_values = [0.001, 0.01, 0.0203, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0];
    let mut prev_output = 0.0_f32;

    for &val in &test_values {
        let input_r = Image::new_with_value((1, 1), val)?;
        let input_g = Image::new_with_value((1, 1), val)?;
        let input_b = Image::new_with_value((1, 1), val)?;

        let s = make_stage(source_it, desired_it, JxlToneMappingMethod::Reinhard);
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
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
fn bt2446a_peak_maps_near_peak() -> Result<()> {
    // Source peak (1.0 linear = 10000 nits) should map near output peak.
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;

    let input_r = Image::new_with_value((1, 1), 1.0)?;
    let input_g = Image::new_with_value((1, 1), 1.0)?;
    let input_b = Image::new_with_value((1, 1), 1.0)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446a);
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

    let mapped = output[0].row(0)[0];
    assert!(
        (mapped - 1.0).abs() < 0.05,
        "Source peak should map near ~1.0 output, got {mapped}"
    );

    Ok(())
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
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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

        let s = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446aPerceptual);
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
fn bt2446a_perceptual_highlights_compressed() -> Result<()> {
    let source_it = 10000.0_f32;
    let desired_it = 203.0_f32;
    let bright_linear = 5000.0 / source_it;

    let input_r = Image::new_with_value((1, 1), bright_linear)?;
    let input_g = Image::new_with_value((1, 1), bright_linear)?;
    let input_b = Image::new_with_value((1, 1), bright_linear)?;

    let stage = make_stage(source_it, desired_it, JxlToneMappingMethod::Bt2446aPerceptual);
    let output =
        make_and_run_simple_pipeline(stage, &[input_r, input_g, input_b], (1, 1), 0, 256)?;

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
