use llmzip_core::cloud::{CloudCredentials, S3MultipartStreamer};
use llmzip_core::dequant::{
    export_safetensors_shards, load_and_dequantize_gguf, DequantDtype,
};
use llmzip_core::hasher::sha256_hex;
use llmzip_core::layout::{
    build_deterministic_layout, QuantPreset, GGUF_MAGIC_VERIFIED,
};
use llmzip_core::preflight::{build_virtual_tensor_space, parse_and_validate_header_len, ShardDescriptor};
use llmzip_core::quant::gguf_simd::{decode_and_validate_upstream_to_f32, quantize_f32_slice};
use llmzip_core::streamer::{EngineConfig, HttpResponseMeta, StreamingQuantizeEngine};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Progress event emitted during streaming quantization.
#[pyclass(name = "ProgressEvent")]
#[derive(Clone)]
pub struct PyProgressEvent {
    #[pyo3(get)]
    pub bytes_processed: u64,
    #[pyo3(get)]
    pub total_bytes: u64,
    #[pyo3(get)]
    pub speed_mbps: f64,
    #[pyo3(get)]
    pub current_layer: Option<usize>,
    #[pyo3(get)]
    pub s3_parts_completed: Option<u32>,
}

#[pymethods]
impl PyProgressEvent {
    fn __repr__(&self) -> String {
        format!(
            "ProgressEvent(bytes_processed={}, total_bytes={}, speed_mbps={:.2}, current_layer={:?}, s3_parts={:?})",
            self.bytes_processed, self.total_bytes, self.speed_mbps, self.current_layer, self.s3_parts_completed
        )
    }
}

/// Returns the engine version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Inspects a model shard and previews planned quantization layouts without holding the model in RAM.
#[pyfunction]
fn inspect(py: Python<'_>, path_or_url: &str) -> PyResult<PyObject> {
    let p = Path::new(path_or_url);
    if !p.exists() {
        return Err(PyValueError::new_err(format!(
            "Input file '{}' not found",
            path_or_url
        )));
    }

    let raw = fs::read(p).map_err(|e| PyValueError::new_err(format!("Failed to read: {}", e)))?;
    if raw.len() < 8 {
        return Err(PyValueError::new_err("File smaller than 8 bytes"));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&raw[0..8]);
    let shard_name = p.file_name().and_then(|s| s.to_str()).unwrap_or("model.safetensors");
    let header_len = parse_and_validate_header_len(shard_name, len_bytes, raw.len() as u64)
        .map_err(|e| PyValueError::new_err(format!("Invalid header: {}", e)))?;
    let header_slice = raw[8..8 + (header_len as usize)].to_vec();
    let expected_sha = sha256_hex(&raw);

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: shard_name.to_string(),
        total_file_size: raw.len() as u64,
        header_len,
        expected_sha256_hex: expected_sha,
    };

    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8)
        .map_err(|e| PyValueError::new_err(format!("VTAS error: {}", e)))?;

    let plan_q4km = build_deterministic_layout(&vtas, QuantPreset::GgufQ4KM, "llama")
        .map_err(|e| PyValueError::new_err(format!("Layout error: {}", e)))?;
    let plan_q8 = build_deterministic_layout(&vtas, QuantPreset::GgufQ8_0, "llama")
        .map_err(|e| PyValueError::new_err(format!("Layout error: {}", e)))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("total_file_size", raw.len() as u64)?;
    dict.set_item("header_len", header_len)?;
    dict.set_item("tensor_count", vtas.tensors.len())?;

    let pred_sizes = PyDict::new_bound(py);
    pred_sizes.set_item("q4_k_m", plan_q4km.total_output_file_size)?;
    pred_sizes.set_item("q8_0", plan_q8.total_output_file_size)?;
    dict.set_item("predicted_sizes", pred_sizes)?;

    let tensor_list = PyList::empty_bound(py);
    for t in &vtas.tensors {
        let t_dict = PyDict::new_bound(py);
        t_dict.set_item("name", &t.hf_name)?;
        t_dict.set_item("shape", &t.shape)?;
        t_dict.set_item("layer_idx", t.layer_idx)?;
        tensor_list.append(t_dict)?;
    }
    dict.set_item("tensors", tensor_list)?;

    Ok(dict.into())
}

fn parse_py_preset(s: &str) -> PyResult<QuantPreset> {
    match s.to_ascii_lowercase().as_str() {
        "q4_k_m" | "q4km" => Ok(QuantPreset::GgufQ4KM),
        "q8_0" | "q8" => Ok(QuantPreset::GgufQ8_0),
        "awq" => Ok(QuantPreset::AwqW4G128),
        "nvfp4" => Ok(QuantPreset::Nvfp4),
        "mxfp4" => Ok(QuantPreset::Mxfp4),
        "fp8" => Ok(QuantPreset::Fp8E4M3),
        other => Err(PyValueError::new_err(format!(
            "Unsupported quantization preset '{}'",
            other
        ))),
    }
}

/// Zero-Temp-File streaming quantization pipeline. Releases the Python GIL during heavy processing.
#[pyfunction]
#[pyo3(signature = (upstream_url, out_destination, preset=None, s3_part_size_mb=None, progress_callback=None))]
fn quantize_stream(
    py: Python<'_>,
    upstream_url: &str,
    out_destination: &str,
    preset: Option<&str>,
    s3_part_size_mb: Option<usize>,
    progress_callback: Option<PyObject>,
) -> PyResult<PyObject> {
    let preset_enum = parse_py_preset(preset.unwrap_or("q4_k_m"))?;
    let input_path = Path::new(upstream_url);
    if !input_path.exists() {
        return Err(PyValueError::new_err(format!(
            "Input '{}' not found",
            upstream_url
        )));
    }

    let is_s3 = out_destination.starts_with("s3://");
    let s3_part_bytes = s3_part_size_mb.unwrap_or(16) * 1024 * 1024;

    let raw = fs::read(input_path).map_err(|e| PyValueError::new_err(format!("Failed to read: {}", e)))?;
    if raw.len() < 8 {
        return Err(PyValueError::new_err("File smaller than 8 bytes"));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&raw[0..8]);
    let shard_name = input_path.file_name().and_then(|s| s.to_str()).unwrap_or("model.safetensors");
    let header_len = parse_and_validate_header_len(shard_name, len_bytes, raw.len() as u64)
        .map_err(|e| PyValueError::new_err(format!("Invalid header: {}", e)))?;
    let header_slice = raw[8..8 + (header_len as usize)].to_vec();
    let expected_sha = sha256_hex(&raw);

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: shard_name.to_string(),
        total_file_size: raw.len() as u64,
        header_len,
        expected_sha256_hex: expected_sha,
    };

    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8)
        .map_err(|e| PyValueError::new_err(format!("VTAS error: {}", e)))?;

    let plan = build_deterministic_layout(&vtas, preset_enum, "llama")
        .map_err(|e| PyValueError::new_err(format!("Layout error: {}", e)))?;

    let total_expected_bytes = plan.total_output_file_size;
    let start_time = Instant::now();

    let committed_count: usize;

    if is_s3 {
        let creds = CloudCredentials::from_env();
        let mut streamer = S3MultipartStreamer::new(out_destination, creds.as_ref(), Some(s3_part_bytes))
            .map_err(|e| PyValueError::new_err(format!("S3 error: {}", e)))?;

        let mut header_bytes = plan.unverified_header_bytes.clone();
        header_bytes[0..4].copy_from_slice(&GGUF_MAGIC_VERIFIED);
        if (header_bytes.len() as u64) < plan.data_section_base_offset {
            header_bytes.resize(plan.data_section_base_offset as usize, 0u8);
        }
        streamer
            .write_chunk(&header_bytes)
            .map_err(|e| PyValueError::new_err(format!("S3 write header failed: {}", e)))?;

        let mut current_offset = plan.data_section_base_offset;
        let mut count = 0;

        for tensor in &vtas.tensors {
            let target = &plan.write_targets[tensor.tensor_id];
            if target.file_write_offset > current_offset {
                let pad_len = (target.file_write_offset - current_offset) as usize;
                let pad = vec![0u8; pad_len];
                streamer
                    .write_chunk(&pad)
                    .map_err(|e| PyValueError::new_err(format!("S3 pad failed: {}", e)))?;
                current_offset += pad_len as u64;
            }

            let slice_start = tensor.stream_abs_start as usize;
            let slice_end = tensor.stream_abs_end as usize;
            let raw_tensor_slice = &raw[slice_start..slice_end];

            // Release GIL during compute
            let q_bytes = py
                .allow_threads(|| -> Result<Vec<u8>, String> {
                    let f32_elements = decode_and_validate_upstream_to_f32(
                        &tensor.hf_name,
                        raw_tensor_slice,
                        tensor.dtype,
                        false,
                    )
                    .map_err(|e| format!("{}", e))?;
                    let qb = quantize_f32_slice(&f32_elements, target.assigned_quant);
                    Ok(qb)
                })
                .map_err(PyValueError::new_err)?;

            let parts_now = streamer.parts_count();

            streamer
                .write_chunk(&q_bytes)
                .map_err(|e| PyValueError::new_err(format!("S3 write chunk failed: {}", e)))?;
            current_offset += q_bytes.len() as u64;
            count += 1;

            if let Some(ref cb) = progress_callback {
                let elapsed_sec = start_time.elapsed().as_secs_f64().max(0.001);
                let speed_mbps = (current_offset as f64 / 1_000_000.0) / elapsed_sec;
                let event = PyProgressEvent {
                    bytes_processed: current_offset,
                    total_bytes: total_expected_bytes,
                    speed_mbps,
                    current_layer: tensor.layer_idx,
                    s3_parts_completed: Some(parts_now),
                };
                let _ = cb.call1(py, (event,));
            }
        }

        streamer
            .complete()
            .map_err(|e| PyValueError::new_err(format!("S3 complete failed: {}", e)))?;
        committed_count = count;
    } else {
        let out_p = Path::new(out_destination);
        let cfg = EngineConfig::default();

        let mut engine = StreamingQuantizeEngine::new_or_resume(vtas, plan.clone(), out_p, cfg)
            .map_err(|e| PyValueError::new_err(format!("Engine error: {}", e)))?;

        let meta = HttpResponseMeta {
            status_code: 200,
            content_range_start: None,
        };

        // Release GIL during file streaming
        let committed = py.allow_threads(|| {
            engine.stream_shard_slice(0, meta, &raw, 1024 * 1024, None)
        }).map_err(|e| PyValueError::new_err(format!("Streaming error: {}", e)))?;

        committed_count = committed;
    }

    let res = PyDict::new_bound(py);
    res.set_item("committed_tensors", committed_count)?;
    res.set_item("total_bytes", total_expected_bytes)?;
    res.set_item("destination", out_destination)?;
    res.set_item("elapsed_seconds", start_time.elapsed().as_secs_f64())?;

    Ok(res.into())
}

/// Dequantizes a GGUF / AWQ model into HuggingFace-compatible sharded SafeTensors.
#[pyfunction]
#[pyo3(signature = (input_path, out_dir, dtype=None, max_shard_size_gb=None))]
fn dequantize(
    py: Python<'_>,
    input_path: &str,
    out_dir: &str,
    dtype: Option<&str>,
    max_shard_size_gb: Option<f64>,
) -> PyResult<PyObject> {
    let in_p = Path::new(input_path);
    let out_p = Path::new(out_dir);

    let deq_dtype = DequantDtype::parse(dtype.unwrap_or("bf16"))
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let max_bytes = max_shard_size_gb.map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as u64);

    let written = py.allow_threads(|| -> Result<Vec<String>, String> {
        let tensors = load_and_dequantize_gguf(in_p, deq_dtype).map_err(|e| format!("{}", e))?;
        let paths = export_safetensors_shards(out_p, tensors, max_bytes).map_err(|e| format!("{}", e))?;
        Ok(paths.into_iter().map(|p| p.display().to_string()).collect())
    }).map_err(PyValueError::new_err)?;

    let py_list = PyList::new_bound(py, &written);
    Ok(py_list.into())
}

#[pymodule]
fn llmzip(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgressEvent>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_stream, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize, m)?)?;
    Ok(())
}
