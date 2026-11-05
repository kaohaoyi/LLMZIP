use crate::dequant::engine::{dequantize_to_dtype, gguf_to_hf_name, DequantDtype};
use crate::dequant::safetensors_export::ExportTensor;
use crate::error::{LlmZipError, Result};
use crate::layout::{QuantFormat, GGUF_MAGIC_UNVERIFIED, GGUF_MAGIC_VERIFIED};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug)]
pub struct ParsedGgufTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub format: QuantFormat,
    pub data_offset: u64,
    pub byte_len: u64,
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf) as usize;
    if len > 64 * 1024 * 1024 {
        return Err(LlmZipError::MetadataParse("GGUF string exceeds 64MB".to_string()));
    }
    let mut str_buf = vec![0u8; len];
    reader.read_exact(&mut str_buf)?;
    String::from_utf8(str_buf).map_err(|e| LlmZipError::MetadataParse(format!("Invalid UTF-8: {}", e)))
}

fn skip_metadata_value<R: Read + Seek>(reader: &mut R, val_type: u32) -> Result<()> {
    match val_type {
        0 | 1 | 7 => { reader.seek(SeekFrom::Current(1))?; } // u8, i8, bool
        2 | 3 => { reader.seek(SeekFrom::Current(2))?; }     // u16, i16
        4 | 5 | 6 => { reader.seek(SeekFrom::Current(4))?; } // u32, i32, f32
        10 | 11 | 12 => { reader.seek(SeekFrom::Current(8))?; } // u64, i64, f64
        8 => {
            // String
            let mut len_buf = [0u8; 8];
            reader.read_exact(&mut len_buf)?;
            let len = u64::from_le_bytes(len_buf) as i64;
            reader.seek(SeekFrom::Current(len))?;
        }
        9 => {
            // Array: [elem_type: u32] [count: u64] [items...]
            let mut elem_type_buf = [0u8; 4];
            reader.read_exact(&mut elem_type_buf)?;
            let elem_type = u32::from_le_bytes(elem_type_buf);
            let mut count_buf = [0u8; 8];
            reader.read_exact(&mut count_buf)?;
            let count = u64::from_le_bytes(count_buf);
            for _ in 0..count {
                skip_metadata_value(reader, elem_type)?;
            }
        }
        other => {
            return Err(LlmZipError::MetadataParse(format!("Unknown GGUF metadata type {}", other)));
        }
    }
    Ok(())
}

fn quant_format_from_gguf_type(t: u32) -> Result<QuantFormat> {
    match t {
        0 => Ok(QuantFormat::GgufF32),
        1 => Ok(QuantFormat::GgufF16),
        8 => Ok(QuantFormat::GgufQ8_0),
        12 => Ok(QuantFormat::GgufQ4K),
        14 => Ok(QuantFormat::GgufQ6K),
        100 => Ok(QuantFormat::AwqInt4G128),
        101 => Ok(QuantFormat::Nvfp4Sb256),
        102 => Ok(QuantFormat::Nvfp4B16),
        103 => Ok(QuantFormat::Mxfp4B32),
        104 => Ok(QuantFormat::Fp8E4M3B32),
        other => Err(LlmZipError::DequantError {
            tensor_name: String::new(),
            reason: format!("Unsupported GGUF tensor type ID {}", other),
        }),
    }
}

/// Reads a GGUF file and dequantizes its tensors directly into memory chunks.
pub fn load_and_dequantize_gguf(
    path: &Path,
    target_dtype: DequantDtype,
) -> Result<Vec<ExportTensor>> {
    let mut file = File::open(path)?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if magic != GGUF_MAGIC_VERIFIED && magic != GGUF_MAGIC_UNVERIFIED {
        return Err(LlmZipError::MetadataParse(format!(
            "File '{}' does not have a valid GGUF header magic",
            path.display()
        )));
    }

    let mut version_buf = [0u8; 4];
    file.read_exact(&mut version_buf)?;
    let version = u32::from_le_bytes(version_buf);
    if version < 2 || version > 3 {
        return Err(LlmZipError::MetadataParse(format!("Unsupported GGUF version {}", version)));
    }

    let mut count_buf = [0u8; 8];
    file.read_exact(&mut count_buf)?;
    let tensor_count = u64::from_le_bytes(count_buf);

    let mut kv_buf = [0u8; 8];
    file.read_exact(&mut kv_buf)?;
    let kv_count = u64::from_le_bytes(kv_buf);

    let mut alignment: u64 = 32;

    for _ in 0..kv_count {
        let key = read_gguf_string(&mut file)?;
        let mut type_buf = [0u8; 4];
        file.read_exact(&mut type_buf)?;
        let val_type = u32::from_le_bytes(type_buf);

        if key == "general.alignment" && (val_type == 4 || val_type == 10) {
            let mut align_buf = [0u8; 4];
            file.read_exact(&mut align_buf)?;
            alignment = u32::from_le_bytes(align_buf) as u64;
            if val_type == 10 {
                file.seek(SeekFrom::Current(4))?; // skip upper 4 bytes
            }
        } else {
            skip_metadata_value(&mut file, val_type)?;
        }
    }

    // Read tensor descriptors
    let mut parsed_tensors = Vec::with_capacity(tensor_count as usize);

    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut file)?;

        let mut n_dims_buf = [0u8; 4];
        file.read_exact(&mut n_dims_buf)?;
        let n_dims = u32::from_le_bytes(n_dims_buf) as usize;

        let mut dims = Vec::with_capacity(n_dims);
        let mut total_elements: u64 = 1;
        for _ in 0..n_dims {
            let mut dim_buf = [0u8; 8];
            file.read_exact(&mut dim_buf)?;
            let d = u64::from_le_bytes(dim_buf);
            dims.push(d as usize);
            total_elements *= d;
        }

        let mut type_buf = [0u8; 4];
        file.read_exact(&mut type_buf)?;
        let t_type = u32::from_le_bytes(type_buf);
        let format = quant_format_from_gguf_type(t_type)?;

        let mut offset_buf = [0u8; 8];
        file.read_exact(&mut offset_buf)?;
        let data_offset = u64::from_le_bytes(offset_buf);

        // GGUF stores dimensions in reverse order
        let hf_shape: Vec<usize> = dims.into_iter().rev().collect();
        let blk_elements = format.block_elements() as u64;
        let blk_bytes = format.block_bytes();
        let num_blocks = total_elements.div_ceil(blk_elements);
        let byte_len = num_blocks * blk_bytes;

        parsed_tensors.push(ParsedGgufTensor {
            name,
            shape: hf_shape,
            format,
            data_offset,
            byte_len,
        });
    }

    let current_pos = file.stream_position()?;
    let data_section_base = align_up(current_pos, alignment);

    let mut export_tensors = Vec::with_capacity(parsed_tensors.len());

    for pt in parsed_tensors {
        let abs_offset = data_section_base + pt.data_offset;
        file.seek(SeekFrom::Start(abs_offset))?;
        let mut q_bytes = vec![0u8; pt.byte_len as usize];
        file.read_exact(&mut q_bytes)?;

        let deq_data = dequantize_to_dtype(&q_bytes, pt.format, target_dtype);
        let hf_name = gguf_to_hf_name(&pt.name);

        export_tensors.push(ExportTensor {
            name: hf_name,
            dtype: target_dtype,
            shape: pt.shape,
            data: deq_data,
        });
    }

    Ok(export_tensors)
}
