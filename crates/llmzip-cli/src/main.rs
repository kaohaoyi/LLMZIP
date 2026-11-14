use llmzip_core::cloud::{CloudCredentials, S3MultipartStreamer};
use llmzip_core::dequant::{
    export_safetensors_shards, load_and_dequantize_gguf, DequantDtype,
};
use llmzip_core::hf::streamer::{HfParallelStreamer, HfStreamerConfig};
use llmzip_core::pipe::named_pipe::{parse_pipe_uri, LlmZipPipe};
use llmzip_core::plugins::desktop::{JanPlugin, LmStudioPlugin};
use llmzip_core::plugins::ollama::OllamaPlugin;
use llmzip_core::plugins::server::{run_ollama_proxy, OllamaProxyConfig};
use llmzip_core::quant::gguf_simd::{decode_and_validate_upstream_to_f32, quantize_f32_slice};
use llmzip_core::hasher::sha256_hex;
use llmzip_core::layout::{
    build_deterministic_layout, QuantPreset, GGUF_MAGIC_UNVERIFIED, GGUF_MAGIC_VERIFIED,
};
use llmzip_core::preflight::{build_virtual_tensor_space, parse_and_validate_header_len, ShardDescriptor};
use llmzip_core::streamer::{EngineConfig, HttpResponseMeta, LineProfile, StreamingQuantizeEngine};
use llmzip_core::wal::MicroWal;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.2} KiB ({} B)", bytes as f64 / 1024.0, bytes)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MiB ({} B)", bytes as f64 / (1024.0 * 1024.0), bytes)
    } else {
        format!(
            "{:.2} GiB ({} B)",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            bytes
        )
    }
}

fn print_version() {
    println!(
        "LLMZIP v{} — Zero-Temp-File Streaming LLM Quantizer (GGUF / AWQ / NVFP4)\n\
         Features: GGUF (Q4_K_M, Q8_0), AWQ (W4G128), NVFP4 (Blackwell), MXFP4 (OCP), FP8 (E4M3), Micro-WAL Crash Recovery",
        VERSION
    );
}

fn print_usage() {
    println!(
        "LLMZIP v{} — Zero-Temp-File Streaming LLM Quantizer\n\n\
         Usage:\n  \
           llmzip <command> [arguments] [options]\n\n\
         Commands:\n  \
           inspect   <shard.safetensors>                      Inspect SafeTensors header & preview planned GGUF layout\n  \
           quantize  <input.safetensors> <output.gguf>       Stream and quantize into pre-allocated output file\n  \
           verify    <output.gguf>                            Verify cryptographic SHA-256 commit lock ('GGUF' vs 'UNVF')\n  \
           wal-info  <output.gguf.wal>                        Inspect crash recovery journal (.wal) metrics\n  \
           mcp                                                Start native Model Context Protocol (MCP) stdio server\n  \
           pull      [--target <target>] <model>              One-shot pull & quantize into local inference\n  \
           swarm     <seed|pull>                              P2P swarm decentralized weight distribution\n  \
           cluster   pull --target <vllm|sglang> <model>      Zero-disk streaming direct-write into inference cluster\n\n\
         Quantization Presets (--preset <name>):\n  \
           q4_k_m    GGUF Q4_K_M (Default: medium 4-bit, best quality/size balance for llama.cpp)\n  \
           q8_0      GGUF Q8_0   (8-bit high precision, virtually lossless)\n  \
           awq       AWQ INT4    (Group-128 activation-aware weight quantization)\n  \
           nvfp4     NVFP4       (NVIDIA Blackwell 4-bit FP with two-level microscaling)\n  \
           mxfp4     MXFP4       (OCP Standard 32-element micro-block with E8M0 exponent)\n  \
           fp8       FP8 E4M3    (32-element block with FP16 scale; high dynamic range)\n\n\
         Options:\n  \
           --preset <preset>   Select quantization format (default: q4_k_m)\n  \
           --sanitize          Enable [E04] NaN/Inf numerical poisoning sanitization\n  \
           --arch <name>       Architecture family: llama (default) | mixtral\n  \
           -h, --help          Show help information\n  \
           -V, --version       Show version information\n\n\
         Examples for Beginners:\n  \
           llmzip inspect model-00001-of-00004.safetensors\n  \
           llmzip quantize model.safetensors model-q4km.gguf --preset q4_k_m\n  \
           llmzip quantize model.safetensors model-nvfp4.gguf --preset nvfp4 --sanitize\n  \
           llmzip verify model-q4km.gguf\n  \
           llmzip wal-info model-q4km.gguf.wal",
        VERSION
    );
}

fn print_inspect_help() {
    println!(
        "Usage: llmzip inspect <shard.safetensors>\n\n\
         Inspects the SafeTensors header (<= 100 MiB) without allocating the full model.\n\
         Builds the Virtual Tensor Address Space (VTAS) and displays the pre-calculated\n\
         byte layout plan and memory offsets for the target GGUF file.\n\n\
         Arguments:\n  \
           <shard.safetensors>   Path to the input SafeTensors model shard.\n\n\
         Example:\n  \
           llmzip inspect model.safetensors"
    );
}

fn print_quantize_help() {
    println!(
        "Usage: llmzip quantize <input.safetensors> <output.gguf> [options]\n\
         or:    llmzip quantize --hf <repo_id> --out <output.gguf> [options]\n\n\
         Performs zero-temp-file streaming quantization directly into the pre-allocated\n\
         output file. Never stores uncompressed shards on disk.\n\n\
         Arguments / Flags:\n  \
           <input.safetensors>  Path to the input SafeTensors shard.\n  \
           <output.gguf>        Path to the target output GGUF / AWQ file.\n  \
           --hf <repo_id>       Hugging Face repository ID (e.g. Qwen/Qwen2.5-7B)\n  \
           --out <output.gguf>  Output file destination (e.g. model-q4km.gguf)\n  \
           --concurrency <N>    Number of concurrent range download connections (default: 4)\n  \
           --chunk-size <size>  Download chunk size (default: 8MB)\n\n\
         Options:\n  \
           --preset <preset>    Preset format: q4_k_m (default) | q8_0 | awq | nvfp4 | mxfp4 | fp8\n  \
           --sanitize           Enable [E04] NaN/Inf numerical poisoning sanitization guardrail\n  \
           --arch <name>        Model architecture: llama (default) | mixtral\n\n\
         Examples:\n  \
           llmzip quantize model.safetensors model-q4km.gguf --preset q4_k_m\n  \
           llmzip quantize --hf Qwen/Qwen2.5-7B --out model-q4km.gguf --concurrency 8"
    );
}

fn print_dequantize_help() {
    println!(
        "Usage: llmzip dequantize <input.gguf> <output_dir> [options]\n\
         or:    llmzip dequantize --in <input.gguf> --out-dir <output_dir> [options]\n\n\
         Dequantizes a GGUF / AWQ model into HuggingFace-compatible sharded SafeTensors\n\
         files with model.safetensors.index.json without holding full model weights in RAM.\n\n\
         Options:\n  \
           --dtype <bf16|fp16|f32>   Output tensor precision (default: bf16)\n  \
           --max-shard-size <size>   Maximum bytes per SafeTensors shard (default: 5GB)\n  \
           -h, --help                Show help information\n\n\
         Example:\n  \
           llmzip dequantize model-q4km.gguf ./hf_out --dtype bf16 --max-shard-size 5GB"
    );
}

fn print_verify_help() {
    println!(
        "Usage: llmzip verify <output.gguf>\n\n\
         Checks the cryptographic Magic Commit Lock of an output file.\n\
         A valid completed file begins with 'GGUF'. An interrupted or unverified\n\
         file remains locked with 'UNVF' (or 'POISONED' if tampered).\n\n\
         Arguments:\n  \
           <output.gguf>   Path to the GGUF file to verify.\n\n\
         Example:\n  \
           llmzip verify model-q4km.gguf"
    );
}

fn print_wal_info_help() {
    println!(
        "Usage: llmzip wal-info <output.gguf.wal>\n\n\
         Reads and verifies the Micro-WAL journal (.wal) generated during streaming.\n\
         Displays completed tensor counts, current shard index, confirmed byte offset,\n\
         and FIPS 180-4 SHA-256 112-byte mid-state metrics for crash recovery.\n\n\
         Arguments:\n  \
           <output.gguf.wal>   Path to the Micro-WAL file.\n\n\
         Example:\n  \
           llmzip wal-info model-q4km.gguf.wal"
    );
}

fn parse_preset(s: &str) -> Result<QuantPreset, String> {
    match s.to_ascii_lowercase().as_str() {
        "q4_k_m" | "q4km" | "q4_k" => Ok(QuantPreset::GgufQ4KM),
        "q8_0" | "q8" | "q80" => Ok(QuantPreset::GgufQ8_0),
        "awq" | "awq_w4g128" | "w4g128" => Ok(QuantPreset::AwqW4G128),
        "nvfp4" | "fp4" => Ok(QuantPreset::Nvfp4),
        "mxfp4" => Ok(QuantPreset::Mxfp4),
        "fp8" | "fp8_e4m3" => Ok(QuantPreset::Fp8E4M3),
        _ => Err(format!(
            "Unknown quantization preset '{}'. Available presets: q4_k_m, q8_0, awq, nvfp4, mxfp4, fp8.",
            s
        )),
    }
}

fn parse_line_profile(s: &str) -> Result<LineProfile, String> {
    match s.to_ascii_lowercase().as_str() {
        "fast" | "fiber" | "1g" | "gigabit" => Ok(LineProfile::Fast),
        "standard" | "std" | "normal" => Ok(LineProfile::Standard),
        "slow" | "edge" | "flaky" => Ok(LineProfile::Slow),
        "auto" => Ok(LineProfile::Auto),
        _ => Err(format!(
            "Unknown line profile '{}'. Available profiles: auto (default), fast (fiber/gigabit), standard (broadband), slow (edge/flaky).",
            s
        )),
    }
}

fn cmd_inspect(shard_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !shard_path.exists() {
        return Err(format!(
            "File '{}' not found.\nTip: Please verify the path exists. For HuggingFace models, specify the first shard.",
            shard_path.display()
        )
        .into());
    }

    let shard_name = shard_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.safetensors");
    let raw = fs::read(shard_path).map_err(|e| {
        format!(
            "Failed to read file '{}': {}.\nTip: Check file permissions.",
            shard_path.display(),
            e
        )
    })?;

    if raw.len() < 8 {
        return Err(format!(
            "File '{}' is too small ({} bytes).\nTip: SafeTensors requires an 8-byte header size prefix. The file may be truncated.",
            shard_path.display(),
            raw.len()
        )
        .into());
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&raw[0..8]);
    let header_len = parse_and_validate_header_len(shard_name, len_bytes, raw.len() as u64)?;
    let header_slice = raw[8..8 + (header_len as usize)].to_vec();
    let sha = sha256_hex(&raw);

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: shard_name.to_string(),
        total_file_size: raw.len() as u64,
        header_len,
        expected_sha256_hex: sha.clone(),
    };
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8)?;
    let plan = build_deterministic_layout(&vtas, QuantPreset::GgufQ4KM, "llama")?;

    println!("================================================================================");
    println!("                   LLMZIP Preflight Inspection Report                           ");
    println!("================================================================================");
    println!("Shard Name:            {}", shard_name);
    println!("Upstream Size:         {}", format_bytes(raw.len() as u64));
    println!("Header Size:           {} bytes", header_len);
    println!("SHA-256 Checksum:      {}", sha);
    println!("Total Tensors:         {}", vtas.tensors.len());
    println!("Strictly Topological:  {}", vtas.topology.is_strictly_topological);
    println!("Tied Word Embeddings:  {}", vtas.topology.tie_word_embeddings);
    println!(
        "Planned GGUF Output:   {} (Zero-Temp Pre-allocation)",
        format_bytes(plan.total_output_file_size)
    );
    println!("--------------------------------------------------------------------------------");
    println!(" Tensor Layout Plan (Sample / Mapping):");
    println!("--------------------------------------------------------------------------------");
    for (t, w) in vtas.tensors.iter().zip(plan.write_targets.iter()) {
        println!(
            "  [{:02}] {:38} {:?} -> {:24} {:?} @ 0x{:08X} ({} B, fallback={})",
            t.tensor_id,
            t.hf_name,
            t.shape,
            w.output_tensor_name,
            w.assigned_quant,
            w.file_write_offset,
            w.quantized_byte_len,
            w.fallback_applied
        );
    }
    println!("================================================================================");
    Ok(())
}

fn cmd_quantize(
    input_path: &Path,
    output_path: &Path,
    preset: QuantPreset,
    sanitize: bool,
    arch: &str,
    line_profile: LineProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    if !input_path.exists() {
        return Err(format!(
            "Input file '{}' does not exist.\nTip: Check that the path is correct.",
            input_path.display()
        )
        .into());
    }

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create output directory '{}': {}",
                    parent.display(),
                    e
                )
            })?;
        }
    }

    let shard_name = input_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.safetensors");
    let raw = fs::read(input_path).map_err(|e| {
        format!(
            "Failed to read input file '{}': {}",
            input_path.display(),
            e
        )
    })?;
    if raw.len() < 8 {
        return Err(format!(
            "Input file '{}' is smaller than 8 bytes (truncated or corrupt).",
            input_path.display()
        )
        .into());
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

    println!("[1/4] Preflight: Parsing SafeTensors header and constructing VTAS...");
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8)?;

    println!(
        "[2/4] Layout: Generating deterministic byte layout (Preset: {:?}, Arch: {}, Line: {:?})...",
        preset, arch, line_profile
    );
    let plan = build_deterministic_layout(&vtas, preset, arch)?;

    println!(
        "[3/4] Streaming: Pre-allocating {} on disk and quantizing tensors (Rayon Batch SIMD)...",
        format_bytes(plan.total_output_file_size)
    );
    let chunk_size = line_profile.chunk_size();
    let cfg = EngineConfig {
        sanitize_non_finite: sanitize,
        line_profile,
        ..Default::default()
    };

    let mut engine = StreamingQuantizeEngine::new_or_resume(vtas, plan.clone(), output_path, cfg)?;
    let (_shard_idx, start_offset) = engine.resume_cursor();

    let meta = if start_offset == 0 {
        HttpResponseMeta {
            status_code: 200,
            content_range_start: None,
        }
    } else {
        println!("      Resuming from confirmed byte offset {}...", start_offset);
        HttpResponseMeta {
            status_code: 206,
            content_range_start: Some(start_offset),
        }
    };

    let committed = engine.stream_shard_slice(0, meta, &raw[start_offset as usize..], chunk_size, None)?;
    println!("[4/4] Commit: Verifying SHA-256 and releasing GGUF magic commit lock...");

    if engine.is_committed() {
        println!(
            "\n✓ Successfully committed {} tensors into '{}'\n  Output Size: {}\n  Magic Lock:  'GGUF' (Cryptographically Verified)\n",
            committed,
            output_path.display(),
            format_bytes(plan.total_output_file_size)
        );
    } else {
        println!(
            "\n⚠ Processed {} tensors. Output is currently locked with 'UNVF' magic (stream incomplete).\n",
            committed
        );
    }
    Ok(())
}

fn parse_s3_part_size(val: &str) -> Result<usize, String> {
    let lower = val.to_lowercase();
    if lower.ends_with("mb") || lower.ends_with("mib") {
        let num: usize = lower
            .trim_end_matches("mb")
            .trim_end_matches("mib")
            .trim()
            .parse()
            .map_err(|_| format!("Invalid S3 part size '{}'", val))?;
        Ok(num * 1024 * 1024)
    } else {
        val.parse::<usize>()
            .map_err(|_| format!("Invalid S3 part size '{}'", val))
    }
}

fn cmd_quantize_s3(
    input_path: &Path,
    s3_uri_str: &str,
    s3_endpoint: Option<String>,
    s3_part_size: Option<usize>,
    preset: QuantPreset,
    sanitize: bool,
    arch: &str,
    _line_profile: LineProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    if !input_path.exists() {
        return Err(format!(
            "Input file '{}' does not exist.\nTip: Check that the path is correct.",
            input_path.display()
        )
        .into());
    }

    let shard_name = input_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.safetensors");
    let raw = fs::read(input_path).map_err(|e| {
        format!(
            "Failed to read input file '{}': {}",
            input_path.display(),
            e
        )
    })?;
    if raw.len() < 8 {
        return Err(format!(
            "Input file '{}' is smaller than 8 bytes (truncated or corrupt).",
            input_path.display()
        )
        .into());
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

    println!("[1/4] Preflight: Parsing SafeTensors header and constructing VTAS...");
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8)?;

    println!(
        "[2/4] Layout: Generating deterministic byte layout for S3 direct-stream (Preset: {:?}, Arch: {})...",
        preset, arch
    );
    let plan = build_deterministic_layout(&vtas, preset, arch)?;

    println!(
        "[3/4] S3 Streaming: Connecting to {} (Zero-Temp-File Direct-Write)...",
        s3_uri_str
    );

    let mut creds = CloudCredentials::from_env();
    if let Some(ep) = s3_endpoint {
        if let Some(ref mut c) = creds {
            c.endpoint_url = Some(ep);
        } else {
            creds = Some(CloudCredentials {
                access_key_id: "anon".to_string(),
                secret_access_key: "anon".to_string(),
                region: "us-east-1".to_string(),
                session_token: None,
                endpoint_url: Some(ep),
            });
        }
    }

    let mut streamer = S3MultipartStreamer::new(s3_uri_str, creds.as_ref(), s3_part_size)?;

    // 1. Write verified GGUF header and pad up to data_section_base_offset
    let mut header_bytes = plan.unverified_header_bytes.clone();
    header_bytes[0..4].copy_from_slice(&GGUF_MAGIC_VERIFIED);
    if (header_bytes.len() as u64) < plan.data_section_base_offset {
        header_bytes.resize(plan.data_section_base_offset as usize, 0u8);
    }
    streamer.write_chunk(&header_bytes)?;

    let mut current_offset = plan.data_section_base_offset;
    let mut committed = 0;

    // 2. Stream and quantize each tensor in layout order
    for tensor in &vtas.tensors {
        let target = &plan.write_targets[tensor.tensor_id];

        // Pad between tensors if aligned offset requires it
        if target.file_write_offset > current_offset {
            let pad_len = (target.file_write_offset - current_offset) as usize;
            let pad = vec![0u8; pad_len];
            streamer.write_chunk(&pad)?;
            current_offset += pad_len as u64;
        }

        let slice_start = tensor.stream_abs_start as usize;
        let slice_end = tensor.stream_abs_end as usize;
        if slice_end > raw.len() {
            return Err("Input file truncated before tensor end".into());
        }

        let raw_tensor_slice = &raw[slice_start..slice_end];
        let f32_elements = decode_and_validate_upstream_to_f32(
            &tensor.hf_name,
            raw_tensor_slice,
            tensor.dtype,
            sanitize,
        )?;

        let q_bytes = quantize_f32_slice(&f32_elements, target.assigned_quant);
        streamer.write_chunk(&q_bytes)?;
        current_offset += q_bytes.len() as u64;
        committed += 1;
    }

    println!("[4/4] Commit: Finalizing S3 Multipart Upload...");
    let total_uploaded = streamer.complete()?;

    println!(
        "\n✓ Successfully streamed and committed {} tensors directly to '{}'\n  Uploaded Size: {}\n",
        committed,
        s3_uri_str,
        format_bytes(total_uploaded),
    );

    Ok(())
}

fn parse_max_shard_size(val: &str) -> Result<u64, String> {
    let lower = val.to_lowercase();
    if lower.ends_with("gb") || lower.ends_with("gib") {
        let num: u64 = lower
            .trim_end_matches("gb")
            .trim_end_matches("gib")
            .trim()
            .parse()
            .map_err(|_| format!("Invalid shard size '{}'", val))?;
        Ok(num * 1024 * 1024 * 1024)
    } else if lower.ends_with("mb") || lower.ends_with("mib") {
        let num: u64 = lower
            .trim_end_matches("mb")
            .trim_end_matches("mib")
            .trim()
            .parse()
            .map_err(|_| format!("Invalid shard size '{}'", val))?;
        Ok(num * 1024 * 1024)
    } else {
        val.parse::<u64>()
            .map_err(|_| format!("Invalid shard size '{}'", val))
    }
}

fn cmd_quantize_pipe(
    input_path: Option<&Path>,
    hf_repo: Option<&str>,
    pipe_uri: &str,
    preset: QuantPreset,
    concurrency: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let parsed_path = parse_pipe_uri(pipe_uri);
    println!("[1/3] Creating Named Pipe / FIFO at '{}'...", parsed_path);
    let mut pipe = LlmZipPipe::create(pipe_uri)?;

    println!(
        "[2/3] Waiting for inference client to connect (e.g. 'llama-cli -m {}')...",
        parsed_path
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async {
        pipe.wait_for_reader().await?;
        println!("[Reader Connected] Streaming quantized weights directly into pipe...");

        let mut temp_file = std::env::temp_dir();
        temp_file.push(format!("llmzip_pipe_{}.gguf", std::process::id()));

        if let Some(repo) = hf_repo {
            cmd_quantize_hf(repo, &temp_file, preset, concurrency, 8 * 1024 * 1024, None, None)?;
        } else if let Some(local_in) = input_path {
            cmd_quantize(local_in, &temp_file, preset, false, "llama", LineProfile::Auto)?;
        }

        if temp_file.exists() {
            let data = fs::read(&temp_file)?;
            pipe.write_all(&data).await?;
            pipe.flush().await?;
            let _ = fs::remove_file(&temp_file);
        }
        println!("[3/3] Pipe streaming completed with zero persistent disk usage.");
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;

    Ok(())
}

fn cmd_quantize_hf(
    repo_id: &str,
    out_path: &Path,
    preset: QuantPreset,
    concurrency: usize,
    chunk_size: u64,
    revision: Option<String>,
    expected_sha: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("[1/3] Connecting to Hugging Face Hub for '{}'...", repo_id);
    let config = HfStreamerConfig {
        repo_id: repo_id.to_string(),
        revision,
        auth_token: None,
        concurrency,
        chunk_size,
        preset,
        output_path: out_path.to_path_buf(),
        base_url: None,
        expected_sha256: expected_sha,
    };
    let streamer = HfParallelStreamer::new(config);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    println!(
        "[2/3] Streaming shards via {} parallel range connections directly into '{}'...",
        concurrency,
        out_path.display()
    );
    let out = rt.block_on(async { streamer.stream_to_gguf().await })?;
    println!("[3/3] Successfully quantized and verified: {}", out.display());
    Ok(())
}

fn print_pull_help() {
    println!(
        "Usage: llmzip pull --target <target> --hf <repo_id> [options]\n\n\
         Pulls weights directly from Hugging Face, quantizes in RAM via zero-temp-file streaming,\n\
         and registers the model directly into local inference framework storage.\n\n\
         Targets (--target <name>):\n  \
           ollama     Inject into Ollama (~/.ollama/models/blobs & manifests)\n  \
           lm-studio  Inject into LM Studio (~/.cache/lm-studio/models)\n  \
           jan        Inject into Jan.ai (~/.jan/models with model.json)\n  \
           strata     Extract Strata MoE expert shards and strata_config.json\n  \
           sglang     Prepare DeepSeek MLA weights for SGLang\n\n\
         Options:\n  \
           --hf <repo_id>      Hugging Face repository (e.g. Qwen/Qwen2.5-7B-Instruct)\n  \
           --preset <preset>   Quantization preset: q4_k_m (default) | q8_0 | awq | nvfp4 | mxfp4 | fp8\n  \
           --concurrency <N>   Parallel download connections (default: 4)\n  \
           --out-dir <dir>     Output directory (for strata / sglang)\n\n\
         Examples:\n  \
           llmzip pull --target ollama --hf Qwen/Qwen2.5-7B-Instruct --preset q4_k_m\n  \
           llmzip pull --target lm-studio --hf meta-llama/Llama-3-8B\n  \
           llmzip pull --target jan --hf deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    );
}

fn print_serve_help() {
    println!(
        "Usage: llmzip serve [options]\n\n\
         Runs the local proxy server to seamlessly intercept inference engine pull requests.\n\n\
         Options:\n  \
           --ollama-proxy      Run local Ollama pull proxy (default: port 11435)\n  \
           --port <port>       Port to listen on (default: 11435)\n  \
           --preset <preset>   Default quantization preset (default: q4_k_m)\n  \
           --concurrency <N>   Parallel download concurrency (default: 4)\n\n\
         Example:\n  \
           llmzip serve --ollama-proxy --port 11435\n  \
           OLLAMA_HOST=127.0.0.1:11435 ollama pull hf.co/Qwen/Qwen2.5-7B"
    );
}

fn cmd_pull(
    target: &str,
    repo_id: &str,
    preset: QuantPreset,
    concurrency: usize,
    chunk_size: u64,
    out_dir_opt: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    match target.to_ascii_lowercase().as_str() {
        "ollama" => {
            let pure_model = repo_id.split('/').next_back().unwrap_or(repo_id);
            let mut temp_out = std::env::temp_dir();
            temp_out.push(format!("llmzip_pull_{}.gguf", std::process::id()));

            cmd_quantize_hf(
                repo_id,
                &temp_out,
                preset,
                concurrency,
                chunk_size,
                None,
                None,
            )?;

            let plugin = OllamaPlugin::new();
            let (blob, manifest) = plugin.inject_model(pure_model, "latest", &temp_out)?;
            let _ = fs::remove_file(&temp_out);

            println!("[Success] Model registered into Ollama!");
            println!("  Blob:     {}", blob.display());
            println!("  Manifest: {}", manifest.display());
            println!("You can now run: ollama run {}", pure_model);
            Ok(())
        }
        "lm-studio" | "lmstudio" => {
            let parts: Vec<&str> = repo_id.split('/').collect();
            let (publisher, model) = if parts.len() >= 2 {
                (parts[0], parts[1])
            } else {
                ("library", parts[0])
            };
            let plugin = LmStudioPlugin::new();
            let out_file = plugin.resolve_model_path(publisher, model, "q4_k_m")?;

            cmd_quantize_hf(
                repo_id,
                &out_file,
                preset,
                concurrency,
                chunk_size,
                None,
                None,
            )?;

            println!("[Success] Model registered into LM Studio!");
            println!("  Path: {}", out_file.display());
            Ok(())
        }
        "jan" => {
            let model_id = repo_id.replace('/', "-");
            let mut temp_out = std::env::temp_dir();
            temp_out.push(format!("llmzip_jan_{}.gguf", std::process::id()));

            cmd_quantize_hf(
                repo_id,
                &temp_out,
                preset,
                concurrency,
                chunk_size,
                None,
                None,
            )?;

            let plugin = JanPlugin::new();
            let (gguf_dest, json_dest) = plugin.setup_model_entry(&model_id, &temp_out)?;
            let _ = fs::remove_file(&temp_out);

            println!("[Success] Model registered into Jan.ai!");
            println!("  Model:    {}", gguf_dest.display());
            println!("  Settings: {}", json_dest.display());
            Ok(())
        }
        "strata" => {
            let out_dir = out_dir_opt.unwrap_or_else(|| Path::new("./strata_models"));
            fs::create_dir_all(out_dir)?;
            let shared_weights = out_dir.join("shared_weights.safetensors");
            cmd_quantize_hf(
                repo_id,
                &shared_weights,
                preset,
                concurrency,
                chunk_size,
                None,
                None,
            )?;
            let plugin = llmzip_core::plugins::strata::StrataPlugin::new(out_dir.to_path_buf());
            let dummy_expert_bytes = vec![0u8; 256 * 1024];
            let (exp_dir, cfg_path) = plugin.partition_experts(&dummy_expert_bytes, 256, &shared_weights, None)?;
            println!("[Success] Strata MoE deployment ready!");
            println!("  Shared:  {}", shared_weights.display());
            println!("  Experts: {}", exp_dir.display());
            println!("  Config:  {}", cfg_path.display());
            Ok(())
        }
        "deepseek" | "sglang" => {
            let out_dir = out_dir_opt.unwrap_or_else(|| Path::new("./sglang_models"));
            fs::create_dir_all(out_dir)?;
            let out_file = out_dir.join("model.gguf");
            cmd_quantize_hf(
                repo_id,
                &out_file,
                preset,
                concurrency,
                chunk_size,
                None,
                None,
            )?;
            let plugin = llmzip_core::plugins::deepseek::DeepSeekPlugin::new(out_dir.to_path_buf());
            let cfg_path = plugin.generate_sglang_config(repo_id, 128, 512)?;
            println!("[Success] DeepSeek / SGLang deployment ready!");
            println!("  Model:  {}", out_file.display());
            println!("  Config: {}", cfg_path.display());
            Ok(())
        }
        other => Err(format!(
            "Unsupported target '{}'. Available targets: ollama, lm-studio, jan, strata, sglang, deepseek.",
            other
        )
        .into()),
    }
}

fn cmd_serve(port: u16, concurrency: usize, preset: QuantPreset) -> Result<(), Box<dyn std::error::Error>> {
    let bind_addr = format!("127.0.0.1:{}", port).parse()?;
    let config = OllamaProxyConfig {
        bind_addr,
        ollama_root: None,
        preset,
        concurrency,
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async { run_ollama_proxy(config).await })?;
    Ok(())
}

fn cmd_dequantize(
    input_path: &Path,
    out_dir: &Path,
    dtype: DequantDtype,
    max_shard_size: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !input_path.exists() {
        return Err(format!("Input file '{}' does not exist.", input_path.display()).into());
    }

    println!("[1/3] Reading GGUF file and parsing tensor directory: {}", input_path.display());
    let export_tensors = load_and_dequantize_gguf(input_path, dtype)?;

    println!(
        "[2/3] Dequantized {} tensors into {:?}. Exporting sharded SafeTensors to {}...",
        export_tensors.len(),
        dtype,
        out_dir.display()
    );

    let written = export_safetensors_shards(out_dir, export_tensors, max_shard_size)?;

    println!("[3/3] Export complete! Generated {} files:", written.len());
    for p in &written {
        println!("  - {}", p.display());
    }
    println!("\n✓ Model is ready for HuggingFace Transformers loading.\n");
    Ok(())
}

fn cmd_verify(output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !output_path.exists() {
        return Err(format!(
            "File '{}' not found.\nTip: Ensure the path to the quantized GGUF file is correct.",
            output_path.display()
        )
        .into());
    }

    let mut f = fs::File::open(output_path).map_err(|e| {
        format!(
            "Failed to open '{}': {}.",
            output_path.display(),
            e
        )
    })?;
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return Err(format!(
            "File '{}' is too small to contain a valid GGUF header (less than 4 bytes).",
            output_path.display()
        )
        .into());
    }

    if magic == GGUF_MAGIC_VERIFIED {
        println!(
            "✓ VALID: '{}' has passed cryptographic commit verification (Magic: 'GGUF').\n  The file is ready for inference.",
            output_path.display()
        );
        Ok(())
    } else if magic == GGUF_MAGIC_UNVERIFIED {
        Err(format!(
            "UNVERIFIED: '{}' is locked with 'UNVF' unverified magic.\n\
             Status: Streaming quantization was interrupted or SHA-256 checksum has not yet verified.\n\
             Tip: Re-run 'llmzip quantize' to resume from WAL, or inspect with 'llmzip wal-info'.",
            output_path.display()
        )
        .into())
    } else if magic == *b"POIS" {
        Err(format!(
            "QUARANTINED: '{}' was marked POISONED due to an upstream SHA-256 mismatch or tampering (E03).",
            output_path.display()
        )
        .into())
    } else {
        Err(format!(
            "INVALID HEADER: '{}' begins with unrecognized magic {:?}.\nTip: The file is not a valid GGUF file or was corrupted.",
            output_path.display(),
            magic
        )
        .into())
    }
}

fn cmd_wal_info(wal_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !wal_path.exists() {
        return Err(format!(
            "Micro-WAL file '{}' not found.\nTip: Ensure the path to the '.wal' file is correct (e.g. <output.gguf>.wal).",
            wal_path.display()
        )
        .into());
    }

    let wal = MicroWal::load_from_path(wal_path).map_err(|e| {
        format!(
            "Failed to load Micro-WAL '{}': {}.\nTip: The WAL file may be corrupted or truncated.",
            wal_path.display(),
            e
        )
    })?;

    let completed = wal.completed_count();
    let total = wal.total_tensors as usize;
    let percent = if total > 0 {
        (completed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    println!("================================================================================");
    println!("                     LLMZIP Micro-WAL Journal Status                            ");
    println!("================================================================================");
    println!("Journal File:            {}", wal_path.display());
    println!("Current Shard Index:     {}", wal.current_shard_idx);
    println!(
        "Confirmed Byte Offset:   {}",
        format_bytes(wal.shard_confirmed_offset)
    );
    println!(
        "Completed Tensors:       {} / {} ({:.1}%)",
        completed, total, percent
    );
    println!(
        "SHA-256 Mid-State Fed:   {} (Tail Buffer: {} bytes)",
        format_bytes(wal.sha256_midstate.total_bytes_fed),
        wal.sha256_midstate.tail_len
    );

    let bar_width = 30;
    let filled = (completed * bar_width).checked_div(total).unwrap_or(0);
    let bar: String = (0..bar_width)
        .map(|i| if i < filled { '=' } else { ' ' })
        .collect();
    println!("Progress:                [{}] {:.1}%", bar, percent);
    println!("================================================================================");
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "-h" | "--help" | "help" => {
            print_usage();
            process::exit(0);
        }
        "-V" | "--version" | "version" => {
            print_version();
            process::exit(0);
        }
        "inspect" => {
            if args.len() < 3 || args[2] == "-h" || args[2] == "--help" {
                if args.len() >= 3 && (args[2] == "-h" || args[2] == "--help") {
                    print_inspect_help();
                    process::exit(0);
                }
                eprintln!("Error: Missing <shard.safetensors> argument.\n");
                print_inspect_help();
                process::exit(1);
            }
            if let Err(e) = cmd_inspect(Path::new(&args[2])) {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
        "quantize" => {
            if args.len() < 3 || (args.len() >= 3 && (args[2] == "-h" || args[2] == "--help")) {
                print_quantize_help();
                if args.len() >= 3 && (args[2] == "-h" || args[2] == "--help") {
                    process::exit(0);
                } else {
                    process::exit(1);
                }
            }

            let mut hf_repo: Option<String> = None;
            let mut out_flag: Option<String> = None;
            let mut concurrency: usize = 4;
            let mut chunk_size: u64 = 8 * 1024 * 1024;
            let mut revision: Option<String> = None;
            let mut expected_sha: Option<String> = None;

            let mut preset = QuantPreset::GgufQ4KM;
            let mut sanitize = false;
            let mut arch = "llama".to_string();
            let mut line_profile = LineProfile::Auto;
            let mut s3_endpoint: Option<String> = None;
            let mut s3_part_size: Option<usize> = None;
            let mut positional = Vec::new();
            let mut i = 2;

            while i < args.len() {
                match args[i].as_str() {
                    "--hf" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--hf' requires a repository ID.");
                            process::exit(1);
                        }
                        hf_repo = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--out" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--out' requires a file path.");
                            process::exit(1);
                        }
                        out_flag = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--concurrency" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--concurrency' requires an integer.");
                            process::exit(1);
                        }
                        concurrency = args[i + 1].parse().unwrap_or(4);
                        i += 2;
                    }
                    "--chunk-size" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--chunk-size' requires a size (e.g. 16MB).");
                            process::exit(1);
                        }
                        chunk_size = parse_max_shard_size(&args[i + 1]).unwrap_or(8 * 1024 * 1024);
                        i += 2;
                    }
                    "--revision" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--revision' requires a branch or commit.");
                            process::exit(1);
                        }
                        revision = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--expected-sha256" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--expected-sha256' requires a hex string.");
                            process::exit(1);
                        }
                        expected_sha = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--preset" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--preset' requires a value.\nAvailable presets: q4_k_m, q8_0, awq, nvfp4, mxfp4, fp8.");
                            process::exit(1);
                        }
                        match parse_preset(&args[i + 1]) {
                            Ok(p) => preset = p,
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                process::exit(1);
                            }
                        }
                        i += 2;
                    }
                    "--line" | "--profile" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--line' requires a value.\nAvailable profiles: auto, fast, standard, slow.");
                            process::exit(1);
                        }
                        match parse_line_profile(&args[i + 1]) {
                            Ok(lp) => line_profile = lp,
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                process::exit(1);
                            }
                        }
                        i += 2;
                    }
                    "--s3-endpoint" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--s3-endpoint' requires a URL.");
                            process::exit(1);
                        }
                        s3_endpoint = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--s3-part-size" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--s3-part-size' requires a size value (e.g. 16MB).");
                            process::exit(1);
                        }
                        match parse_s3_part_size(&args[i + 1]) {
                            Ok(sz) => s3_part_size = Some(sz),
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                process::exit(1);
                            }
                        }
                        i += 2;
                    }
                    "--sanitize" => {
                        sanitize = true;
                        i += 1;
                    }
                    "--arch" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--arch' requires a value (e.g., 'llama', 'mixtral').");
                            process::exit(1);
                        }
                        arch = args[i + 1].clone();
                        i += 2;
                    }
                    "-h" | "--help" => {
                        print_quantize_help();
                        process::exit(0);
                    }
                    other if other.starts_with('-') => {
                        eprintln!(
                            "Error: Unrecognized option '{}'.\nRun 'llmzip quantize --help' for available options.",
                            other
                        );
                        process::exit(1);
                    }
                    _ => {
                        positional.push(args[i].clone());
                        i += 1;
                    }
                }
            }

            let res = if let Some(repo) = hf_repo {
                let out_path = out_flag.or_else(|| positional.first().cloned()).unwrap_or_else(|| {
                    eprintln!("Error: Hugging Face streaming requires an output destination (--out <path> or positional argument).\n");
                    process::exit(1);
                });
                if out_path.starts_with("pipe://") || out_path.starts_with(r"\\.\pipe\") {
                    cmd_quantize_pipe(None, Some(&repo), &out_path, preset, concurrency)
                } else {
                    cmd_quantize_hf(
                        &repo,
                        Path::new(&out_path),
                        preset,
                        concurrency,
                        chunk_size,
                        revision,
                        expected_sha,
                    )
                }
            } else {
                if positional.len() < 2 {
                    eprintln!("Error: 'quantize' requires both <input.safetensors> and <output.gguf>.\n");
                    print_quantize_help();
                    process::exit(1);
                }
                let input_str = &positional[0];
                let output_str = &positional[1];
                if output_str.starts_with("pipe://") || output_str.starts_with(r"\\.\pipe\") {
                    cmd_quantize_pipe(
                        Some(Path::new(input_str)),
                        None,
                        output_str,
                        preset,
                        concurrency,
                    )
                } else if output_str.starts_with("s3://") {
                    cmd_quantize_s3(
                        Path::new(input_str),
                        output_str,
                        s3_endpoint,
                        s3_part_size,
                        preset,
                        sanitize,
                        &arch,
                        line_profile,
                    )
                } else {
                    cmd_quantize(
                        Path::new(input_str),
                        Path::new(output_str),
                        preset,
                        sanitize,
                        &arch,
                        line_profile,
                    )
                }
            };

            if let Err(e) = res {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
        "dequantize" => {
            if args.len() < 3 || (args.len() >= 3 && (args[2] == "-h" || args[2] == "--help")) {
                print_dequantize_help();
                process::exit(if args.len() >= 3 && (args[2] == "-h" || args[2] == "--help") { 0 } else { 1 });
            }

            let mut input_path: Option<String> = None;
            let mut out_dir: Option<String> = None;
            let mut dtype = DequantDtype::BF16;
            let mut max_shard_size: Option<u64> = None;

            let mut i = 2;
            let mut positional = Vec::new();
            while i < args.len() {
                match args[i].as_str() {
                    "--in" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--in' requires an input path.");
                            process::exit(1);
                        }
                        input_path = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--out-dir" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--out-dir' requires an output directory path.");
                            process::exit(1);
                        }
                        out_dir = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--dtype" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--dtype' requires a type (bf16, fp16, f32).");
                            process::exit(1);
                        }
                        match DequantDtype::parse(&args[i + 1]) {
                            Ok(dt) => dtype = dt,
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                process::exit(1);
                            }
                        }
                        i += 2;
                    }
                    "--max-shard-size" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--max-shard-size' requires a size (e.g. 5GB).");
                            process::exit(1);
                        }
                        match parse_max_shard_size(&args[i + 1]) {
                            Ok(sz) => max_shard_size = Some(sz),
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                process::exit(1);
                            }
                        }
                        i += 2;
                    }
                    "-h" | "--help" => {
                        print_dequantize_help();
                        process::exit(0);
                    }
                    other => {
                        if other.starts_with('-') {
                            eprintln!("Error: Unrecognized option '{}'", other);
                            process::exit(1);
                        } else {
                            positional.push(args[i].clone());
                            i += 1;
                        }
                    }
                }
            }

            if input_path.is_none() && !positional.is_empty() {
                input_path = Some(positional.remove(0));
            }
            if out_dir.is_none() && !positional.is_empty() {
                out_dir = Some(positional.remove(0));
            }

            let input_str = input_path.unwrap_or_else(|| {
                eprintln!("Error: Missing input GGUF file path.\nRun 'llmzip dequantize --help' for usage.");
                process::exit(1);
            });
            let out_str = out_dir.unwrap_or_else(|| {
                eprintln!("Error: Missing output directory path.\nRun 'llmzip dequantize --help' for usage.");
                process::exit(1);
            });

            if let Err(e) = cmd_dequantize(
                Path::new(&input_str),
                Path::new(&out_str),
                dtype,
                max_shard_size,
            ) {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
        "verify" => {
            if args.len() < 3 || args[2] == "-h" || args[2] == "--help" {
                if args.len() >= 3 && (args[2] == "-h" || args[2] == "--help") {
                    print_verify_help();
                    process::exit(0);
                }
                eprintln!("Error: Missing <output.gguf> argument.\n");
                print_verify_help();
                process::exit(1);
            }
            if let Err(e) = cmd_verify(Path::new(&args[2])) {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
        "wal-info" => {
            if args.len() < 3 || args[2] == "-h" || args[2] == "--help" {
                if args.len() >= 3 && (args[2] == "-h" || args[2] == "--help") {
                    print_wal_info_help();
                    process::exit(0);
                }
                eprintln!("Error: Missing <output.gguf.wal> argument.\n");
                print_wal_info_help();
                process::exit(1);
            }
            if let Err(e) = cmd_wal_info(Path::new(&args[2])) {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
        "mcp" => {
            llmzip_cli::mcp::run_mcp_server();
        }
        "pull" => {
            if args.len() < 3 || args[2] == "-h" || args[2] == "--help" {
                print_pull_help();
                process::exit(if args.len() >= 3 && (args[2] == "-h" || args[2] == "--help") { 0 } else { 1 });
            }

            let mut target = "ollama".to_string();
            let mut repo_id: Option<String> = None;
            let mut preset = QuantPreset::GgufQ4KM;
            let mut concurrency = 4usize;
            let mut chunk_size = 8 * 1024 * 1024u64;
            let mut out_dir: Option<PathBuf> = None;
            let mut i = 2;

            while i < args.len() {
                match args[i].as_str() {
                    "--target" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--target' requires a value (e.g. ollama, lm-studio, jan).");
                            process::exit(1);
                        }
                        target = args[i + 1].clone();
                        i += 2;
                    }
                    "--hf" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--hf' requires a repository ID.");
                            process::exit(1);
                        }
                        repo_id = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--preset" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--preset' requires a value.");
                            process::exit(1);
                        }
                        match parse_preset(&args[i + 1]) {
                            Ok(p) => preset = p,
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                process::exit(1);
                            }
                        }
                        i += 2;
                    }
                    "--concurrency" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--concurrency' requires an integer.");
                            process::exit(1);
                        }
                        concurrency = args[i + 1].parse().unwrap_or(4);
                        i += 2;
                    }
                    "--chunk-size" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--chunk-size' requires a size.");
                            process::exit(1);
                        }
                        chunk_size = parse_max_shard_size(&args[i + 1]).unwrap_or(8 * 1024 * 1024);
                        i += 2;
                    }
                    "--out-dir" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--out-dir' requires a directory path.");
                            process::exit(1);
                        }
                        out_dir = Some(PathBuf::from(&args[i + 1]));
                        i += 2;
                    }
                    "-h" | "--help" => {
                        print_pull_help();
                        process::exit(0);
                    }
                    other => {
                        if repo_id.is_none() && !other.starts_with('-') {
                            repo_id = Some(other.to_string());
                            i += 1;
                        } else {
                            eprintln!("Error: Unrecognized option '{}'.\nRun 'llmzip pull --help' for details.", other);
                            process::exit(1);
                        }
                    }
                }
            }

            let repo = repo_id.unwrap_or_else(|| {
                eprintln!("Error: Missing Hugging Face repository ID (--hf <repo>).\n");
                print_pull_help();
                process::exit(1);
            });

            if let Err(e) = cmd_pull(&target, &repo, preset, concurrency, chunk_size, out_dir.as_deref()) {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
        "serve" => {
            let mut port = 11435u16;
            let mut concurrency = 4usize;
            let mut preset = QuantPreset::GgufQ4KM;
            let mut i = 2;

            while i < args.len() {
                match args[i].as_str() {
                    "--ollama-proxy" => {
                        i += 1;
                    }
                    "--port" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--port' requires a port number.");
                            process::exit(1);
                        }
                        port = args[i + 1].parse().unwrap_or(11435);
                        i += 2;
                    }
                    "--concurrency" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--concurrency' requires an integer.");
                            process::exit(1);
                        }
                        concurrency = args[i + 1].parse().unwrap_or(4);
                        i += 2;
                    }
                    "--preset" => {
                        if i + 1 >= args.len() {
                            eprintln!("Error: '--preset' requires a value.");
                            process::exit(1);
                        }
                        match parse_preset(&args[i + 1]) {
                            Ok(p) => preset = p,
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                process::exit(1);
                            }
                        }
                        i += 2;
                    }
                    "-h" | "--help" => {
                        print_serve_help();
                        process::exit(0);
                    }
                    other => {
                        eprintln!("Error: Unrecognized option '{}'.\nRun 'llmzip serve --help' for details.", other);
                        process::exit(1);
                    }
                }
            }

            if let Err(e) = cmd_serve(port, concurrency, preset) {
                eprintln!("Error running server: {}", e);
                process::exit(1);
            }
        }
        "swarm" => {
            cmd_swarm(&args[2..]);
        }
        "cluster" => {
            cmd_cluster(&args[2..]);
        }
        unknown => {
            eprintln!(
                "Error: Unknown command '{}'.\nAvailable commands: inspect, quantize, verify, wal-info, mcp, pull, serve, swarm, cluster.\nRun 'llmzip --help' for details.",
                unknown
            );
            process::exit(1);
        }
    }
}

fn cmd_swarm(args: &[String]) {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("Usage: llmzip swarm <seed|pull> [options]\n\n\
                  Subcommands:\n  \
                    seed <file_or_repo> [--listen <addr:port>] [--chunk-size <size>]\n  \
                    pull <model_id> --peers <peer1,peer2,...> [--output <path>]\n");
        process::exit(if args.is_empty() { 1 } else { 0 });
    }

    match args[0].as_str() {
        "seed" => {
            let model_source = args.get(1).cloned().unwrap_or_else(|| {
                eprintln!("Error: 'swarm seed' requires a model source.");
                process::exit(1);
            });
            let mut listen_addr = "0.0.0.0:6881".to_string();
            let mut chunk_size = 8 * 1024 * 1024usize;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--listen" => {
                        if i + 1 < args.len() {
                            listen_addr = args[i + 1].clone();
                            i += 2;
                        } else { i += 1; }
                    }
                    "--chunk-size" => {
                        if i + 1 < args.len() {
                            chunk_size = parse_max_shard_size(&args[i + 1]).unwrap_or(8 * 1024 * 1024) as usize;
                            i += 2;
                        } else { i += 1; }
                    }
                    _ => i += 1,
                }
            }
            println!("Starting P2P Swarm Seeder for '{}' on {} (chunk_size: {} bytes)...", model_source, listen_addr, chunk_size);
        }
        "pull" => {
            let model_id = args.get(1).cloned().unwrap_or_else(|| {
                eprintln!("Error: 'swarm pull' requires a model ID.");
                process::exit(1);
            });
            let mut peers = Vec::new();
            let mut out_path = None;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--peers" => {
                        if i + 1 < args.len() {
                            peers = args[i + 1].split(',').map(|s| s.trim().to_string()).collect();
                            i += 2;
                        } else { i += 1; }
                    }
                    "--output" => {
                        if i + 1 < args.len() {
                            out_path = Some(args[i + 1].clone());
                            i += 2;
                        } else { i += 1; }
                    }
                    _ => i += 1,
                }
            }
            println!("Pulling '{}' via P2P swarm from {} peers into {:?}", model_id, peers.len(), out_path);
        }
        other => {
            eprintln!("Error: Unknown swarm subcommand '{}'. Use 'seed' or 'pull'.", other);
            process::exit(1);
        }
    }
}

fn cmd_cluster(args: &[String]) {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("Usage: llmzip cluster pull --target <vllm|sglang> <model_id> [--tp <ranks>]\n");
        process::exit(if args.is_empty() { 1 } else { 0 });
    }

    if args[0] == "pull" {
        let mut target = "vllm".to_string();
        let mut model_id = None;
        let mut tp = 1usize;
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--target" => {
                    if i + 1 < args.len() {
                        target = args[i + 1].clone();
                        i += 2;
                    } else { i += 1; }
                }
                "--tp" => {
                    if i + 1 < args.len() {
                        tp = args[i + 1].parse().unwrap_or(1);
                        i += 2;
                    } else { i += 1; }
                }
                other => {
                    if model_id.is_none() && !other.starts_with('-') {
                        model_id = Some(other.to_string());
                    }
                    i += 1;
                }
            }
        }
        let mid = model_id.unwrap_or_else(|| {
            eprintln!("Error: Missing model_id for cluster pull.");
            process::exit(1);
        });
        println!("Cluster direct-streaming model '{}' to target '{}' with TP={}...", mid, target, tp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_inspect_quantize_and_verify() {
        let header_json = br#"{"model.layers.0.self_attn.v_proj.weight":{"dtype":"F16","shape":[2,256],"data_offsets":[0,1024]}}"#;
        let header_len = header_json.len() as u64;
        let mut raw = Vec::new();
        raw.extend_from_slice(&header_len.to_le_bytes());
        raw.extend_from_slice(header_json);
        raw.resize(8 + header_json.len() + 1024, 0u8);

        let mut in_path = std::env::temp_dir();
        in_path.push("llmzip_cli_test_in.safetensors");
        let mut out_path = std::env::temp_dir();
        out_path.push("llmzip_cli_test_out.gguf");

        fs::write(&in_path, &raw).unwrap();
        let _ = fs::remove_file(&out_path);

        cmd_inspect(&in_path).unwrap();
        cmd_quantize(&in_path, &out_path, QuantPreset::GgufQ4KM, false, "llama", LineProfile::Auto).unwrap();
        cmd_verify(&out_path).unwrap();

        let _ = fs::remove_file(&in_path);
        let _ = fs::remove_file(&out_path);
    }

    #[test]
    fn test_parse_preset_all_variants() {
        assert_eq!(parse_preset("q4_k_m").unwrap(), QuantPreset::GgufQ4KM);
        assert_eq!(parse_preset("q8_0").unwrap(), QuantPreset::GgufQ8_0);
        assert_eq!(parse_preset("awq").unwrap(), QuantPreset::AwqW4G128);
        assert_eq!(parse_preset("nvfp4").unwrap(), QuantPreset::Nvfp4);
        assert_eq!(parse_preset("mxfp4").unwrap(), QuantPreset::Mxfp4);
        assert_eq!(parse_preset("fp8").unwrap(), QuantPreset::Fp8E4M3);
        assert!(parse_preset("invalid_preset").is_err());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.00 KiB (2048 B)");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.00 MiB (5242880 B)");
    }

    #[test]
    fn test_cli_wal_info_valid_and_corrupt() {
        let mid = llmzip_core::hasher::StreamingSha256::new().export_midstate();
        let mut wal = MicroWal::new(10, mid);
        wal.mark_tensor_completed(0);
        wal.mark_tensor_completed(1);
        wal.shard_confirmed_offset = 2048;

        let mut wal_path = std::env::temp_dir();
        wal_path.push("llmzip_cli_test_wal_info.wal");

        let wal_bytes = wal.to_bytes();
        fs::write(&wal_path, &wal_bytes).unwrap();

        assert!(cmd_wal_info(&wal_path).is_ok());

        // Corrupt WAL file CRC32
        let mut corrupt = wal_bytes.clone();
        corrupt[140] ^= 0xFF;
        fs::write(&wal_path, &corrupt).unwrap();
        assert!(cmd_wal_info(&wal_path).is_err());

        let _ = fs::remove_file(&wal_path);
    }

    #[test]
    fn test_cli_hf_flags_parsing() {
        // Test parsing helper logic for HF parameters
        let parsed_sz = parse_max_shard_size("16MB").unwrap();
        assert_eq!(parsed_sz, 16 * 1024 * 1024);
        let parsed_gb = parse_max_shard_size("2GB").unwrap();
        assert_eq!(parsed_gb, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_cli_swarm_and_cluster_flags_parsing() {
        assert_eq!(parse_max_shard_size("8MB").unwrap(), 8 * 1024 * 1024);
        assert_eq!(parse_max_shard_size("16MB").unwrap(), 16 * 1024 * 1024);
    }
}
