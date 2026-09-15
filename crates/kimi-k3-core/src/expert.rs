//! Resolution and bounded loading of one routed MXFP4 expert.
//!
//! The six tensors belonging to one expert are normally contiguous in a Kimi K3
//! checkpoint. That is an observed storage property, never an unchecked assumption:
//! this module verifies it and retains a six-read fallback for a repacked checkpoint.

use std::fmt;

use crate::{
    io::{ReadRequest, fit},
    safetensors::{DType, SafeTensorError, SafeTensorIndex, TensorInfo},
};

/// Number of logical MXFP4 values covered by one E8M0 scale.
pub const MXFP4_GROUP_SIZE: usize = 32;

const MATRIX_NAMES: [&str; 3] = ["w1", "w2", "w3"];
const EXPERT_PREFIX: &str = "language_model.model.layers";

/// Byte layout and geometry of a packed MXFP4 matrix in an expert run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedMatrix {
    /// Offset of the packed nibble data within [`ExpertRef::nbytes`].
    pub packed_offset: usize,
    /// Size of the packed nibble data in bytes.
    pub packed_bytes: usize,
    /// Offset of the E8M0 group scales within [`ExpertRef::nbytes`].
    pub scale_offset: usize,
    /// Size of the E8M0 group scales in bytes.
    pub scale_bytes: usize,
    /// Matrix row count.
    pub rows: usize,
    /// Stored packed columns; the logical width is twice this value.
    pub packed_columns: usize,
    /// E8M0 scales per row.
    pub scale_columns: usize,
}

impl QuantizedMatrix {
    /// Returns the number of dequantized logical values in this matrix.
    #[must_use]
    pub const fn numel(&self) -> usize {
        self.rows * self.packed_columns * 2
    }
}

/// One fully validated routed expert, ready for a single contiguous read when possible.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertRef {
    /// Zero-based transformer layer index.
    pub layer: usize,
    /// Zero-based routed-expert index.
    pub expert: usize,
    /// Stable safetensors shard index.
    pub shard: usize,
    /// Absolute start of the expert run, or the lowest tensor offset for a fallback.
    pub offset: u64,
    /// Size of the canonical in-memory expert layout.
    pub nbytes: usize,
    /// True when one shard read covers all six tensors with no gaps or overlap.
    pub contiguous: bool,
    /// The `w1`, `w2`, and `w3` matrix layouts, in kernel consumption order.
    pub matrices: [QuantizedMatrix; 3],
}

impl ExpertRef {
    /// Resolves and validates all six tensors of one routed expert.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertError`] if any tensor is missing, has invalid MXFP4 geometry, or
    /// is split across shards. A non-contiguous but otherwise valid expert is accepted.
    pub fn resolve(
        index: &SafeTensorIndex,
        layer: usize,
        expert: usize,
    ) -> Result<Self, ExpertError> {
        let mut pairs = Vec::with_capacity(3);
        for matrix in MATRIX_NAMES {
            let packed_name = tensor_name(layer, expert, matrix, "weight_packed");
            let scale_name = tensor_name(layer, expert, matrix, "weight_scale");
            let packed = index
                .tensor(&packed_name)
                .ok_or_else(|| ExpertError::MissingTensor {
                    name: packed_name.clone(),
                })?;
            let scales = index
                .tensor(&scale_name)
                .ok_or_else(|| ExpertError::MissingTensor {
                    name: scale_name.clone(),
                })?;
            pairs.push(validate_pair(layer, expert, matrix, packed, scales)?);
        }

        let shard = pairs[0].packed.shard;
        if pairs
            .iter()
            .any(|pair| pair.packed.shard != shard || pair.scales.shard != shard)
        {
            return Err(ExpertError::SplitAcrossShards { layer, expert });
        }

        let mut spans = Vec::with_capacity(6);
        for pair in &pairs {
            spans.push((pair.packed.offset, pair.packed.nbytes));
            spans.push((pair.scales.offset, pair.scales.nbytes));
        }
        let offset = spans
            .iter()
            .map(|&(offset, _)| offset)
            .min()
            .ok_or(ExpertError::OffsetOverflow { layer, expert })?;
        let high = spans
            .iter()
            .map(|&(start, bytes)| {
                start
                    .checked_add(
                        u64::try_from(bytes)
                            .map_err(|_| ExpertError::OffsetOverflow { layer, expert })?,
                    )
                    .ok_or(ExpertError::OffsetOverflow { layer, expert })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or(ExpertError::OffsetOverflow { layer, expert })?;
        let own_bytes = spans.iter().try_fold(0_usize, |total, &(_, bytes)| {
            total
                .checked_add(bytes)
                .ok_or(ExpertError::OffsetOverflow { layer, expert })
        })?;
        let own_span =
            u64::try_from(own_bytes).map_err(|_| ExpertError::OffsetOverflow { layer, expert })?;
        let contiguous = high
            .checked_sub(offset)
            .is_some_and(|span| span == own_span);

        let mut next = 0_usize;
        let matrices = [
            matrix_layout(&pairs[0], contiguous, offset, &mut next, layer, expert)?,
            matrix_layout(&pairs[1], contiguous, offset, &mut next, layer, expert)?,
            matrix_layout(&pairs[2], contiguous, offset, &mut next, layer, expert)?,
        ];

        Ok(Self {
            layer,
            expert,
            shard,
            offset,
            nbytes: own_bytes,
            contiguous,
            matrices,
        })
    }

    /// Loads this expert into its canonical six-tensor byte layout.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertError`] if `output` does not exactly fit the expert or a shard
    /// cannot be read in full.
    pub fn load_into(&self, index: &SafeTensorIndex, output: &mut [u8]) -> Result<(), ExpertError> {
        if output.len() != self.nbytes {
            return Err(ExpertError::OutputLength {
                layer: self.layer,
                expert: self.expert,
                expected: self.nbytes,
                actual: output.len(),
            });
        }
        let mut loaded = Self::load_batch(index, vec![(self, Vec::new())])?;
        output.copy_from_slice(&loaded.pop().unwrap_or_default());
        Ok(())
    }

    /// Loads several experts in one proactor batch, each into the buffer paired with it.
    ///
    /// Every buffer is fitted to its expert's canonical size and handed to the read
    /// itself, so passing a cache slot's storage loads the expert with no copy. A
    /// contiguous expert is one positioned read; a repacked one is six, submitted in the
    /// same batch and assembled afterwards. Returns the filled buffers in input order.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertError`] if a tensor is missing or any read fails. The buffers are
    /// consumed either way.
    pub fn load_batch(
        index: &SafeTensorIndex,
        experts: Vec<(&Self, Vec<u8>)>,
    ) -> Result<Vec<Vec<u8>>, ExpertError> {
        let mut requests = Vec::with_capacity(experts.len());
        let mut plans = Vec::with_capacity(experts.len());
        for (expert, mut buffer) in experts {
            fit(&mut buffer, expert.nbytes);
            if expert.contiguous {
                plans.push(LoadPlan::Whole {
                    request: requests.len(),
                });
                requests.push(ReadRequest {
                    shard: expert.shard,
                    offset: expert.offset,
                    buffer,
                });
                continue;
            }
            let first = requests.len();
            for matrix in MATRIX_NAMES {
                for suffix in ["weight_packed", "weight_scale"] {
                    let name = tensor_name(expert.layer, expert.expert, matrix, suffix);
                    let tensor = index
                        .tensor(&name)
                        .ok_or(ExpertError::MissingTensor { name })?;
                    requests.push(ReadRequest {
                        shard: tensor.shard,
                        offset: tensor.offset,
                        buffer: vec![0; tensor.nbytes],
                    });
                }
            }
            plans.push(LoadPlan::Pieces {
                expert,
                first,
                buffer,
            });
        }

        let mut filled: Vec<Option<Vec<u8>>> =
            index.read_batch(requests)?.into_iter().map(Some).collect();
        let mut take = |request: usize| filled[request].take().unwrap_or_default();
        Ok(plans
            .into_iter()
            .map(|plan| match plan {
                LoadPlan::Whole { request } => take(request),
                LoadPlan::Pieces {
                    expert,
                    first,
                    mut buffer,
                } => {
                    for (matrix_index, layout) in expert.matrices.iter().enumerate() {
                        let packed = take(first + matrix_index * 2);
                        let scales = take(first + matrix_index * 2 + 1);
                        buffer[layout.packed_offset..layout.packed_offset + layout.packed_bytes]
                            .copy_from_slice(&packed);
                        buffer[layout.scale_offset..layout.scale_offset + layout.scale_bytes]
                            .copy_from_slice(&scales);
                    }
                    buffer
                }
            })
            .collect())
    }
}

/// How one expert in a [`ExpertRef::load_batch`] maps onto its batch requests.
enum LoadPlan<'a> {
    /// A contiguous expert read by a single request into its own buffer.
    Whole { request: usize },
    /// A repacked expert: six requests starting at `first`, in `w1`..`w3`
    /// packed-then-scales order, copied into `buffer` at their canonical offsets.
    Pieces {
        expert: &'a ExpertRef,
        first: usize,
        buffer: Vec<u8>,
    },
}

/// A malformed or missing routed-expert tensor layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpertError {
    /// A required routed-expert tensor was absent from the checkpoint.
    MissingTensor { name: String },
    /// A tensor had a non-U8 dtype, non-matrix shape, or invalid MXFP4 dimensions.
    InvalidTensor {
        layer: usize,
        expert: usize,
        matrix: String,
        detail: String,
    },
    /// An expert's six tensors were not all stored in one shard.
    SplitAcrossShards { layer: usize, expert: usize },
    /// An offset or canonical byte layout overflowed the platform address space.
    OffsetOverflow { layer: usize, expert: usize },
    /// The output buffer did not fit the canonical expert layout.
    OutputLength {
        layer: usize,
        expert: usize,
        expected: usize,
        actual: usize,
    },
    /// Safetensors index or I/O failure.
    SafeTensor(SafeTensorError),
}

impl fmt::Display for ExpertError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor { name } => {
                write!(formatter, "missing routed-expert tensor {name}")
            }
            Self::InvalidTensor {
                layer,
                expert,
                matrix,
                detail,
            } => write!(
                formatter,
                "layer {layer} expert {expert} matrix {matrix} has invalid MXFP4 layout: {detail}"
            ),
            Self::SplitAcrossShards { layer, expert } => {
                write!(
                    formatter,
                    "layer {layer} expert {expert} is split across shards"
                )
            }
            Self::OffsetOverflow { layer, expert } => {
                write!(formatter, "layer {layer} expert {expert} layout overflowed")
            }
            Self::OutputLength {
                layer,
                expert,
                expected,
                actual,
            } => write!(
                formatter,
                "layer {layer} expert {expert} output has {actual} bytes; expected {expected}"
            ),
            Self::SafeTensor(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ExpertError {}

impl From<SafeTensorError> for ExpertError {
    fn from(error: SafeTensorError) -> Self {
        Self::SafeTensor(error)
    }
}

struct TensorPair<'a> {
    packed: &'a TensorInfo,
    scales: &'a TensorInfo,
    rows: usize,
    packed_columns: usize,
    scale_columns: usize,
}

fn validate_pair<'a>(
    layer: usize,
    expert: usize,
    matrix: &str,
    packed: &'a TensorInfo,
    scales: &'a TensorInfo,
) -> Result<TensorPair<'a>, ExpertError> {
    let invalid = |detail: String| ExpertError::InvalidTensor {
        layer,
        expert,
        matrix: matrix.to_owned(),
        detail,
    };
    if packed.dtype != DType::U8 || scales.dtype != DType::U8 {
        return Err(invalid(
            "both packed weights and scales must use U8".to_owned(),
        ));
    }
    let [rows, packed_columns]: [usize; 2] = packed
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| invalid("packed weights must be rank two".to_owned()))?;
    let [scale_rows, scale_columns]: [usize; 2] = scales
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| invalid("scales must be rank two".to_owned()))?;
    if rows != scale_rows {
        return Err(invalid(format!(
            "packed has {rows} rows but scales has {scale_rows}"
        )));
    }
    let logical_columns = packed_columns
        .checked_mul(2)
        .ok_or_else(|| invalid("logical column count overflowed".to_owned()))?;
    let scale_coverage = scale_columns
        .checked_mul(MXFP4_GROUP_SIZE)
        .ok_or_else(|| invalid("scale coverage overflowed".to_owned()))?;
    if logical_columns != scale_coverage {
        return Err(invalid(format!(
            "{scale_columns} scale columns cover {scale_coverage} values, not {logical_columns}"
        )));
    }
    Ok(TensorPair {
        packed,
        scales,
        rows,
        packed_columns,
        scale_columns,
    })
}

fn matrix_layout(
    pair: &TensorPair<'_>,
    contiguous: bool,
    offset: u64,
    next: &mut usize,
    layer: usize,
    expert: usize,
) -> Result<QuantizedMatrix, ExpertError> {
    let (packed_offset, scale_offset) = if contiguous {
        (
            usize::try_from(pair.packed.offset - offset)
                .map_err(|_| ExpertError::OffsetOverflow { layer, expert })?,
            usize::try_from(pair.scales.offset - offset)
                .map_err(|_| ExpertError::OffsetOverflow { layer, expert })?,
        )
    } else {
        let packed_offset = *next;
        *next = next
            .checked_add(pair.packed.nbytes)
            .ok_or(ExpertError::OffsetOverflow { layer, expert })?;
        let scale_offset = *next;
        *next = next
            .checked_add(pair.scales.nbytes)
            .ok_or(ExpertError::OffsetOverflow { layer, expert })?;
        (packed_offset, scale_offset)
    };
    Ok(QuantizedMatrix {
        packed_offset,
        packed_bytes: pair.packed.nbytes,
        scale_offset,
        scale_bytes: pair.scales.nbytes,
        rows: pair.rows,
        packed_columns: pair.packed_columns,
        scale_columns: pair.scale_columns,
    })
}

fn tensor_name(layer: usize, expert: usize, matrix: &str, suffix: &str) -> String {
    format!("{EXPERT_PREFIX}.{layer}.block_sparse_moe.experts.{expert}.{matrix}.{suffix}")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{ExpertRef, MXFP4_GROUP_SIZE};
    use crate::safetensors::SafeTensorIndex;

    fn fixture_index() -> SafeTensorIndex {
        let directory =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cache");
        SafeTensorIndex::open(directory).expect("cache fixture index opens")
    }

    #[test]
    fn resolves_contiguous_mxfp4_experts_from_the_cache_fixture() {
        let index = fixture_index();
        let expert = ExpertRef::resolve(&index, 0, 7).expect("expert resolves");

        assert!(expert.contiguous);
        assert_eq!(expert.shard, 0);
        assert_eq!(expert.nbytes, 1632);
        assert_eq!(expert.matrices.len(), 3);
        for matrix in &expert.matrices {
            assert_eq!(matrix.rows, 8);
            assert_eq!(matrix.packed_columns, 64);
            assert_eq!(matrix.scale_columns, 4);
            assert_eq!(matrix.numel(), 1024);
            assert_eq!(
                matrix.scale_columns * MXFP4_GROUP_SIZE,
                matrix.packed_columns * 2
            );
        }
    }

    #[test]
    fn loads_each_expert_into_the_same_canonical_layout_as_the_c_reader() {
        let index = fixture_index();
        let expert = ExpertRef::resolve(&index, 0, 7).expect("expert resolves");
        let mut bytes = vec![0; expert.nbytes];
        expert.load_into(&index, &mut bytes).expect("expert loads");

        let first_byte = |matrix: usize, scales: bool| {
            let layout = &expert.matrices[matrix];
            bytes[if scales {
                layout.scale_offset
            } else {
                layout.packed_offset
            }]
        };
        assert_eq!(first_byte(0, false), 183);
        assert_eq!(first_byte(0, true), 190);
        assert_eq!(first_byte(1, false), 214);
        assert_eq!(first_byte(1, true), 221);
        assert_eq!(first_byte(2, false), 245);
        assert_eq!(first_byte(2, true), 1);
    }

    #[test]
    fn a_batch_load_matches_loading_each_expert_alone() {
        let index = fixture_index();
        let experts = (0..24)
            .map(|expert| ExpertRef::resolve(&index, 0, expert).expect("expert resolves"))
            .collect::<Vec<_>>();

        let alone = experts
            .iter()
            .map(|expert| {
                let mut bytes = vec![0; expert.nbytes];
                expert.load_into(&index, &mut bytes).expect("expert loads");
                bytes
            })
            .collect::<Vec<_>>();
        // Reused, wrongly sized buffers must still come back exactly expert-sized.
        let batch = ExpertRef::load_batch(
            &index,
            experts
                .iter()
                .enumerate()
                .map(|(position, expert)| (expert, vec![0xAA; position * 100]))
                .collect(),
        )
        .expect("batch loads");

        assert_eq!(batch, alone);
    }

    #[test]
    fn rejects_a_missing_expert_instead_of_returning_a_zero_weight() {
        let index = fixture_index();
        assert!(ExpertRef::resolve(&index, 0, 24).is_err());
    }
}
