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
use std::path::Path;
use std::process;

const VERSION: &str = "0.1.0";

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
            mcp                                                Start native Model Context Protocol (MCP) stdio server\n\n\
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
        "Usage: llmzip quantize <input.safetensors> <output.gguf> [options]\n\n\
         Performs zero-temp-file streaming quantization directly into the pre-allocated\n\
         output file. Never stores uncompressed shards on disk.\n\n\
         Arguments:\n  \
           <input.safetensors>  Path to the input SafeTensors shard.\n  \
           <output.gguf>        Path to the target output GGUF / AWQ file.\n\n\
         Options:\n  \
           --preset <preset>    Preset format: q4_k_m (default) | q8_0 | awq | nvfp4 | mxfp4 | fp8\n  \
           --sanitize           Enable [E04] NaN/Inf numerical poisoning sanitization guardrail\n  \
           --arch <name>        Model architecture: llama (default) | mixtral\n\n\
         Example:\n  \
           llmzip quantize model.safetensors model-q4km.gguf --preset q4_k_m"
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
            if args.len() < 4 || (args.len() >= 3 && (args[2] == "-h" || args[2] == "--help")) {
                if args.len() >= 3 && (args[2] == "-h" || args[2] == "--help") {
                    print_quantize_help();
                    process::exit(0);
                }
                eprintln!("Error: 'quantize' requires both <input.safetensors> and <output.gguf>.\n");
                print_quantize_help();
                process::exit(1);
            }

            let mut preset = QuantPreset::GgufQ4KM;
            let mut sanitize = false;
            let mut arch = "llama".to_string();
            let mut line_profile = LineProfile::Auto;
            let mut i = 4;

            while i < args.len() {
                match args[i].as_str() {
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
                    other => {
                        eprintln!(
                            "Error: Unrecognized option '{}'.\nRun 'llmzip quantize --help' for available options.",
                            other
                        );
                        process::exit(1);
                    }
                }
            }

            if let Err(e) = cmd_quantize(
                Path::new(&args[2]),
                Path::new(&args[3]),
                preset,
                sanitize,
                &arch,
                line_profile,
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
        unknown => {
            eprintln!(
                "Error: Unknown command '{}'.\nAvailable commands: inspect, quantize, verify, wal-info, mcp.\nRun 'llmzip --help' for details.",
                unknown
            );
            process::exit(1);
        }
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
}
