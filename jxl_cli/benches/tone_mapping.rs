// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use jxl::api::JxlDecoderOptions;
use jxl_cli::cms::Lcms2Cms;
use jxl_cli::dec::{OutputDataType, decode_frames, decode_header};
use jxl_cli::tone_mapping::{
    Bt2446aParams, Rec2408Params, tone_map_bt2446a, tone_map_bt2446a_linear,
    tone_map_bt2446a_perceptual, tone_map_rec2408,
};
use std::path::{Path, PathBuf};

const LUMINANCE_BT2020: [f32; 3] = [0.2627, 0.678, 0.0593];

/// Load HDR pixel data from a JXL file.
///
/// Returns interleaved RGB f32 data and the source intensity target (nits).
/// Set `TONE_MAP_FILE` to specify a custom HDR JXL file; defaults to the
/// repo's `hdr_pq_test.jxl`.
fn load_hdr_data() -> (Vec<f32>, f32) {
    let path = std::env::var("TONE_MAP_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("jxl/resources/test/hdr_pq_test.jxl")
        });

    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    // Read the source intensity target from the header.
    let mut header_input = bytes.as_slice();
    let header = decode_header(&mut header_input, JxlDecoderOptions::default())
        .unwrap_or_else(|e| panic!("Failed to decode header of {}: {e}", path.display()));
    let source_it = header.basic_info().tone_mapping.intensity_target;

    // Decode to F32 linear, no tone mapping.
    let mut options = JxlDecoderOptions::default();
    options.cms = Some(Box::new(Lcms2Cms));

    let mut input = bytes.as_slice();
    let (output, _) = decode_frames(
        &mut input,
        options,
        None,
        Some(OutputDataType::F32),
        &[OutputDataType::F32],
        true,  // interleave_alpha
        true,  // linear_output
        None,  // render_interval
        false, // allow_partial_files
        None,  // native color profile
    )
    .unwrap_or_else(|e| panic!("Failed to decode {}: {e}", path.display()));

    let (width, height) = output.size;
    let frame = &output.frames[0];
    let cpp = frame.color_type.samples_per_pixel();

    // Extract interleaved RGB (3 channels only, skip alpha if present).
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let row_bytes = frame.channels[0].row(y);
        for x in 0..width {
            for c in 0..3 {
                let offset = (x * cpp + c) * 4;
                let bytes: [u8; 4] = row_bytes[offset..offset + 4].try_into().unwrap();
                data.push(f32::from_ne_bytes(bytes));
            }
        }
    }

    eprintln!(
        "Loaded {} ({}x{}, {:.0} nit source, {} pixels)",
        path.display(),
        width,
        height,
        source_it,
        width * height
    );

    (data, source_it)
}

fn tone_mapping_benches(c: &mut Criterion) {
    let (base_data, source_it) = load_hdr_data();
    let pixel_count = (base_data.len() / 3) as u64;
    let desired_it = 203.0;

    let bt2446a_params = Bt2446aParams::new(source_it, desired_it);
    let rec2408_params = Rec2408Params::new([0.0, source_it], [0.0, desired_it]);

    let mut group = c.benchmark_group("tone_mapping");
    group.throughput(criterion::Throughput::Elements(pixel_count));

    group.bench_with_input(
        BenchmarkId::from_parameter("baseline_noop"),
        &base_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut d| {
                    std::hint::black_box(&mut d);
                },
                criterion::BatchSize::LargeInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("bt2446a"),
        &base_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut d| tone_map_bt2446a(&bt2446a_params, LUMINANCE_BT2020, &mut d),
                criterion::BatchSize::LargeInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("bt2446a_linear"),
        &base_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut d| tone_map_bt2446a_linear(&bt2446a_params, LUMINANCE_BT2020, &mut d),
                criterion::BatchSize::LargeInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("bt2446a_perceptual"),
        &base_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut d| tone_map_bt2446a_perceptual(&bt2446a_params, source_it, &mut d),
                criterion::BatchSize::LargeInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("rec2408"),
        &base_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut d| tone_map_rec2408(&rec2408_params, LUMINANCE_BT2020, &mut d),
                criterion::BatchSize::LargeInput,
            )
        },
    );

    group.finish();
}

criterion_group!(
    name = tone_mapping;
    config = Criterion::default().sample_size(50);
    targets = tone_mapping_benches
);
criterion_main!(tone_mapping);
