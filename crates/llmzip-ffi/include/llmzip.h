/**
 * @file llmzip.h
 * @brief LLMZIP Native C SDK Header: Zero-Temp-File Streaming LLM Quantizer & Model Loader
 *
 * Provides C-compatible ABI interfaces for streaming quantization, commit lock verification,
 * and crash-recovery journal inspection. Designed for embedding into C/C++ runtimes
 * such as llama.cpp, Ollama, vLLM, TensorRT-LLM, and custom inference engines.
 *
 * @copyright Copyright (c) 2026 LLMZIP Contributors. Licensed under MIT OR Apache-2.0.
 */

#ifndef LLMZIP_H
#define LLMZIP_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Status and Return Codes                                                   */
/* ========================================================================= */

/** Operation completed successfully. */
#define LLMZIP_OK                     0

/** Target file is verified with valid 'GGUF' magic commit lock. */
#define LLMZIP_VERIFY_VALID           1

/** Target file is unverified (still locked with 'UNVF' unverified magic). */
#define LLMZIP_VERIFY_UNVERIFIED      0

/** Null pointer argument supplied. */
#define LLMZIP_ERR_NULL_PTR          -1

/** General execution, I/O, or parsing error. */
#define LLMZIP_ERR_EXECUTION         -2

/** File header magic is corrupted, poisoned, or unrecognized. */
#define LLMZIP_ERR_INVALID_MAGIC     -3

/* ========================================================================= */
/* Quantization Presets                                                      */
/* ========================================================================= */

typedef enum llmzip_preset {
    /** GGUF Q4_K_M: Balanced 4-bit medium quantization (Q4_K + Q6_K). */
    LLMZIP_PRESET_GGUF_Q4_K_M = 0,

    /** GGUF Q8_0: 8-bit high-precision quantization. */
    LLMZIP_PRESET_GGUF_Q8_0   = 1,

    /** AWQ INT4: Group-128 Activation-aware Weight Quantization. */
    LLMZIP_PRESET_AWQ_W4G128  = 2,

    /** NVFP4: NVIDIA Blackwell two-level microscaling (E2M1 + FP8 scale). */
    LLMZIP_PRESET_NVFP4       = 3,

    /** MXFP4: OCP Standard 32-element micro-block with E8M0 power-of-two scale. */
    LLMZIP_PRESET_MXFP4       = 4,

    /** FP8 E4M3: 32-element block with FP16 scale factor. */
    LLMZIP_PRESET_FP8_E4M3    = 5
} llmzip_preset_t;

/* ========================================================================= */
/* Public API Functions                                                      */
/* ========================================================================= */

/**
 * @brief Returns the null-terminated semantic version string of the LLMZIP library (e.g. "0.1.0").
 *
 * @return Constant pointer to version string. The returned string is statically allocated.
 */
const char* llmzip_version(void);

/**
 * @brief Streams and quantizes an upstream SafeTensors file into a deterministic GGUF/AWQ file.
 *
 * Performs zero-temp-file streaming quantization directly into the pre-allocated output file.
 *
 * @param input_safetensors_path Path to the input .safetensors model shard (null-terminated UTF-8).
 * @param output_gguf_path Path to the target output .gguf file (null-terminated UTF-8).
 * @param preset_id Quantization preset ID (see @ref llmzip_preset_t).
 * @param sanitize_non_finite If true, enables [E04] NaN/Inf numerical poisoning sanitization.
 * @return LLMZIP_OK on success, or negative error code on failure.
 */
int32_t llmzip_stream_quantize_file(
    const char* input_safetensors_path,
    const char* output_gguf_path,
    int32_t preset_id,
    bool sanitize_non_finite
);

/**
 * @brief Verifies whether an output file has passed cryptographic SHA-256 commit lock.
 *
 * @param output_gguf_path Path to the output .gguf file.
 * @return LLMZIP_VERIFY_VALID (1) if file has valid 'GGUF' magic.
 *         LLMZIP_VERIFY_UNVERIFIED (0) if still locked with 'UNVF' unverified magic.
 *         LLMZIP_ERR_NULL_PTR (-1) if path is NULL.
 *         LLMZIP_ERR_EXECUTION (-2) if file cannot be read.
 *         LLMZIP_ERR_INVALID_MAGIC (-3) if header is corrupted or poisoned.
 */
int32_t llmzip_verify_file(const char* output_gguf_path);

/**
 * @brief Reads recovery progress metrics from a Micro-WAL (.wal) file.
 *
 * @param wal_path Path to the journal file (e.g., "model.gguf.wal").
 * @param[out] out_completed_tensors Pointer to store number of completed tensors.
 * @param[out] out_total_tensors Pointer to store total number of tensors planned.
 * @param[out] out_shard_idx Pointer to store the current shard index.
 * @return LLMZIP_OK on success, or negative error code on failure.
 */
int32_t llmzip_wal_get_progress(
    const char* wal_path,
    uint32_t* out_completed_tensors,
    uint32_t* out_total_tensors,
    uint32_t* out_shard_idx
);

#ifdef __cplusplus
}
#endif

#endif /* LLMZIP_H */
