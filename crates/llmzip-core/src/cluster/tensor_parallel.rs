use crate::error::LlmZipError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpSplitKind {
    ColumnParallel,
    RowParallel,
}

pub struct TensorParallelScheduler {
    pub world_size: usize,
}

impl TensorParallelScheduler {
    pub fn new(world_size: usize) -> Self {
        Self {
            world_size: if world_size == 0 { 1 } else { world_size },
        }
    }

    /// Computes the start and end slice index for rank `rank` along the target parallel dimension
    pub fn get_slice_range(
        &self,
        shape: &[usize],
        kind: TpSplitKind,
        rank: usize,
    ) -> Result<(usize, usize), LlmZipError> {
        if rank >= self.world_size {
            return Err(LlmZipError::ClusterError {
                target: "tensor_parallel".to_string(),
                reason: format!("Rank {} out of bounds for world size {}", rank, self.world_size),
            });
        }

        if shape.is_empty() {
            return Err(LlmZipError::InvalidShape {
                tensor_name: "tp_slice".to_string(),
                shape: shape.to_vec(),
            });
        }

        let target_dim = match kind {
            TpSplitKind::ColumnParallel => 0,
            TpSplitKind::RowParallel => {
                if shape.len() > 1 { 1 } else { 0 }
            }
        };

        let dim_len = shape[target_dim];
        let slice_len = dim_len / self.world_size;
        let start = rank * slice_len;
        let end = if rank == self.world_size - 1 {
            dim_len
        } else {
            start + slice_len
        };

        Ok((start, end))
    }
}
