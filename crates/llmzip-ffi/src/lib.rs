use llmzip_core::hasher::sha256_hex;
use llmzip_core::layout::{
    build_deterministic_layout, QuantPreset, GGUF_MAGIC_UNVERIFIED, GGUF_MAGIC_VERIFIED,
};
use llmzip_core::preflight::{build_virtual_tensor_space, parse_and_validate_header_len, ShardDescriptor};
use llmzip_core::streamer::{EngineConfig, HttpResponseMeta, StreamingQuantizeEngine};
use llmzip_core::wal::MicroWal;
use std::ffi::CStr;
use std::fs;
use std::io::Read;
use std::os::raw::c_char;
use std::path::Path;

/// C ABI status codes for embedding LLMZIP as a native Model Loader SDK.
pub const LLMZIP_OK: i32 = 0;
pub const LLMZIP_VERIFY_VALID: i32 = 1;
pub const LLMZIP_VERIFY_UNVERIFIED: i32 = 0;
pub const LLMZIP_ERR_NULL_PTR: i32 = -1;
pub const LLMZIP_ERR_EXECUTION: i32 = -2;
pub const LLMZIP_ERR_INVALID_MAGIC: i32 = -3;

static VERSION_C_STR: &[u8] = b"0.2.0\0";

/// Returns the static version string for LLMZIP (e.g. "0.1.0").
#[no_mangle]
pub extern "C" fn llmzip_version() -> *const c_char {
    VERSION_C_STR.as_ptr() as *const c_char
}

/// C ABI entry point: Quantizes a local or streamed `.safetensors` file directly into a pre-allocated `.gguf` / AWQ file.
/// `preset_id`: `0` = GGUF Q4_K_M, `1` = GGUF Q8_0, `2` = AWQ INT4 Group-128, `3` = NVFP4, `4` = MXFP4, `5` = FP8 E4M3.
///
/// # Safety
///
/// Pointers `input_safetensors_path` and `output_gguf_path` must be valid, null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn llmzip_stream_quantize_file(
    input_safetensors_path: *const c_char,
    output_gguf_path: *const c_char,
    preset_id: i32,
    sanitize_non_finite: bool,
) -> i32 {
    if input_safetensors_path.is_null() || output_gguf_path.is_null() {
        return LLMZIP_ERR_NULL_PTR;
    }

    let in_str = match CStr::from_ptr(input_safetensors_path).to_str() {
        Ok(s) => s,
        Err(_) => return LLMZIP_ERR_EXECUTION,
    };
    let out_str = match CStr::from_ptr(output_gguf_path).to_str() {
        Ok(s) => s,
        Err(_) => return LLMZIP_ERR_EXECUTION,
    };

    let preset = match preset_id {
        0 => QuantPreset::GgufQ4KM,
        1 => QuantPreset::GgufQ8_0,
        2 => QuantPreset::AwqW4G128,
        3 => QuantPreset::Nvfp4,
        4 => QuantPreset::Mxfp4,
        5 => QuantPreset::Fp8E4M3,
        _ => QuantPreset::GgufQ4KM,
    };

    let run = || -> Result<(), Box<dyn std::error::Error>> {
        let in_path = Path::new(in_str);
        let out_path = Path::new(out_str);
        let shard_name = in_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model.safetensors");
        let raw = fs::read(in_path)?;
        if raw.len() < 8 {
            return Err("Input file too small".into());
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&raw[0..8]);
        let header_len = parse_and_validate_header_len(shard_name, len_bytes, raw.len() as u64)?;
        let header_slice = raw[8..8 + (header_len as usize)].to_vec();
        let expected_sha = sha256_hex(&raw);

        let desc = ShardDescriptor {
            shard_idx: 0,
            shard_name: shard_name.to_string(),
            total_file_size: raw.len() as u64,
            header_len,
            expected_sha256_hex: expected_sha,
        };

        let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8)?;
        let plan = build_deterministic_layout(&vtas, preset, "llama")?;
        let cfg = EngineConfig {
            sanitize_non_finite,
            ..Default::default()
        };

        let mut engine = StreamingQuantizeEngine::new_or_resume(vtas, plan, out_path, cfg)?;
        let meta = HttpResponseMeta {
            status_code: 200,
            content_range_start: None,
        };
        engine.stream_shard_slice(0, meta, &raw, 4096, None)?;
        if !engine.is_committed() {
            return Err("Output not committed".into());
        }
        Ok(())
    };

    match run() {
        Ok(()) => LLMZIP_OK,
        Err(_) => LLMZIP_ERR_EXECUTION,
    }
}

/// C ABI entry point: Verifies if a GGUF file has passed the cryptographic commit lock.
/// Returns:
/// - `LLMZIP_VERIFY_VALID` (1) if file has valid verified 'GGUF' magic.
/// - `LLMZIP_VERIFY_UNVERIFIED` (0) if file is still locked with 'UNVF' unverified magic.
/// - `LLMZIP_ERR_NULL_PTR` (-1) if pointer is null.
/// - `LLMZIP_ERR_EXECUTION` (-2) if file cannot be opened or read.
/// - `LLMZIP_ERR_INVALID_MAGIC` (-3) if file magic is corrupted, poisoned, or unknown.
///
/// # Safety
///
/// `output_gguf_path` must be a valid, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn llmzip_verify_file(output_gguf_path: *const c_char) -> i32 {
    if output_gguf_path.is_null() {
        return LLMZIP_ERR_NULL_PTR;
    }
    let path_str = match CStr::from_ptr(output_gguf_path).to_str() {
        Ok(s) => s,
        Err(_) => return LLMZIP_ERR_EXECUTION,
    };
    let mut file = match fs::File::open(Path::new(path_str)) {
        Ok(f) => f,
        Err(_) => return LLMZIP_ERR_EXECUTION,
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return LLMZIP_ERR_EXECUTION;
    }

    if magic == GGUF_MAGIC_VERIFIED {
        LLMZIP_VERIFY_VALID
    } else if magic == GGUF_MAGIC_UNVERIFIED {
        LLMZIP_VERIFY_UNVERIFIED
    } else {
        LLMZIP_ERR_INVALID_MAGIC
    }
}

/// C ABI entry point: Reads recovery progress metrics from a `.wal` file.
///
/// # Safety
///
/// Pointers `wal_path`, `out_completed_tensors`, `out_total_tensors`, and `out_shard_idx` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn llmzip_wal_get_progress(
    wal_path: *const c_char,
    out_completed_tensors: *mut u32,
    out_total_tensors: *mut u32,
    out_shard_idx: *mut u32,
) -> i32 {
    if wal_path.is_null()
        || out_completed_tensors.is_null()
        || out_total_tensors.is_null()
        || out_shard_idx.is_null()
    {
        return LLMZIP_ERR_NULL_PTR;
    }
    let path_str = match CStr::from_ptr(wal_path).to_str() {
        Ok(s) => s,
        Err(_) => return LLMZIP_ERR_EXECUTION,
    };
    let wal = match MicroWal::load_from_path(Path::new(path_str)) {
        Ok(w) => w,
        Err(_) => return LLMZIP_ERR_EXECUTION,
    };

    *out_completed_tensors = wal.completed_count() as u32;
    *out_total_tensors = wal.total_tensors;
    *out_shard_idx = wal.current_shard_idx;

    LLMZIP_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_version_and_verify() {
        let ver_ptr = llmzip_version();
        assert!(!ver_ptr.is_null());
        let ver = unsafe { CStr::from_ptr(ver_ptr).to_str().unwrap() };
        assert_eq!(ver, "0.2.0");

        unsafe {
            assert_eq!(llmzip_verify_file(std::ptr::null()), LLMZIP_ERR_NULL_PTR);
        }
    }

    #[test]
    fn test_ffi_c_abi_null_and_roundtrip() {
        unsafe {
            assert_eq!(
                llmzip_stream_quantize_file(std::ptr::null(), std::ptr::null(), 0, false),
                LLMZIP_ERR_NULL_PTR
            );
        }

        // Build a synthetic .safetensors on disk and call C ABI `llmzip_stream_quantize_file`
        let header_json = br#"{"model.norm.weight":{"dtype":"F32","shape":[32],"data_offsets":[0,128]}}"#;
        let header_len = header_json.len() as u64;
        let mut raw = Vec::new();
        raw.extend_from_slice(&header_len.to_le_bytes());
        raw.extend_from_slice(header_json);
        for _ in 0..32 {
            raw.extend_from_slice(&1.0f32.to_le_bytes());
        }

        let mut in_path = std::env::temp_dir();
        in_path.push("llmzip_ffi_in.safetensors");
        let mut out_path = std::env::temp_dir();
        out_path.push("llmzip_ffi_out.gguf");

        fs::write(&in_path, &raw).unwrap();
        let _ = fs::remove_file(&out_path);

        let c_in = CString::new(in_path.to_str().unwrap()).unwrap();
        let c_out = CString::new(out_path.to_str().unwrap()).unwrap();

        let status = unsafe { llmzip_stream_quantize_file(c_in.as_ptr(), c_out.as_ptr(), 0, false) };
        assert_eq!(status, LLMZIP_OK);

        // Verify with llmzip_verify_file
        let verify_res = unsafe { llmzip_verify_file(c_out.as_ptr()) };
        assert_eq!(verify_res, LLMZIP_VERIFY_VALID);

        let out_bytes = fs::read(&out_path).unwrap();
        assert_eq!(&out_bytes[0..4], b"GGUF");

        let _ = fs::remove_file(&in_path);
        let _ = fs::remove_file(&out_path);
    }
}
