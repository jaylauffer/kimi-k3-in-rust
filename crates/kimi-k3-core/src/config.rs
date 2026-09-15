//! Strict Kimi K3 checkpoint configuration parsing.
//!
//! The released checkpoint uses a nested JSON shape, while the small reference fixture
//! is flat. Both describe the same model. A missing required key is an error: silently
//! substituting defaults can run all 93 layers with the wrong attention mechanism.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use serde_json::{Map, Value};

/// Maximum routed experts selected for one token.
///
/// This is the Rust port of `K3_MAX_TOPK`. It remains an explicit gate even though Rust
/// does not have the C implementation's fixed-size stack arrays.
pub const MAX_TOPK: usize = 64;

/// The largest layer-map accepted by the C reader's public call sites.
pub const MAX_FULL_ATTN_LAYERS: usize = 128;

/// A fully validated model configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct K3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,

    pub kda_num_heads: usize,
    pub kda_head_dim: usize,
    pub short_conv_kernel_size: usize,
    pub gate_lower_bound: f32,

    pub num_attention_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub mla_use_output_gate: bool,

    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub num_shared_experts: usize,
    pub routed_expert_hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub moe_renormalize: bool,
    pub latent_moe_use_norm: bool,

    pub first_k_dense_replace: usize,
    pub intermediate_size: usize,
    pub attn_res_block_size: usize,
    pub activation_situ_beta: f32,
    pub activation_situ_linear_beta: f32,

    /// One-based checkpoint layer positions that use Gated MLA.
    pub full_attn_layers: Vec<usize>,
}

impl K3Config {
    /// Parses one config object in either the released nested or reference flat shape.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when the JSON shape is incomplete or describes an
    /// architecture the engine cannot safely execute.
    pub fn from_value(root: &Value, source: impl Into<String>) -> Result<Self, ConfigError> {
        let source = source.into();
        let mut fields = ConfigFields::new(root);

        let config = Self {
            hidden_size: fields.required_usize("hidden_size", None),
            num_hidden_layers: fields.required_usize("num_hidden_layers", None),
            vocab_size: fields.required_usize("vocab_size", None),
            rms_norm_eps: fields.required_f32("rms_norm_eps", None),

            kda_num_heads: fields.required_usize("num_heads", Some("kda_num_heads")),
            kda_head_dim: fields.required_usize("head_dim", Some("kda_head_dim")),
            short_conv_kernel_size: fields.required_usize("short_conv_kernel_size", None),
            gate_lower_bound: fields.required_f32("gate_lower_bound", None),

            num_attention_heads: fields.required_usize("num_attention_heads", None),
            q_lora_rank: fields.required_usize("q_lora_rank", None),
            kv_lora_rank: fields.required_usize("kv_lora_rank", None),
            qk_nope_head_dim: fields.required_usize("qk_nope_head_dim", None),
            qk_rope_head_dim: fields.required_usize("qk_rope_head_dim", None),
            v_head_dim: fields.required_usize("v_head_dim", None),
            mla_use_output_gate: fields.optional_bool("mla_use_output_gate", None, true),

            num_experts: fields.required_usize("num_experts", None),
            num_experts_per_token: fields.required_usize("num_experts_per_token", None),
            num_shared_experts: fields.required_usize("num_shared_experts", None),
            routed_expert_hidden_size: fields.required_usize("routed_expert_hidden_size", None),
            moe_intermediate_size: fields.required_usize("moe_intermediate_size", None),
            routed_scaling_factor: fields.required_f32("routed_scaling_factor", None),
            moe_renormalize: fields.optional_bool("moe_renormalize", None, true),
            latent_moe_use_norm: fields.optional_bool("latent_moe_use_norm", None, true),

            first_k_dense_replace: fields.required_usize("first_k_dense_replace", None),
            intermediate_size: fields.required_usize("intermediate_size", None),
            attn_res_block_size: fields.required_usize("attn_res_block_size", None),
            activation_situ_beta: fields.required_f32("activation_situ_beta", Some("situ_beta")),
            activation_situ_linear_beta: fields
                .required_f32("activation_situ_linear_beta", Some("situ_linear_beta")),
            full_attn_layers: fields.full_attn_layers(),
        };

        if !fields.missing.is_empty() {
            return Err(ConfigError::MissingFields {
                source,
                fields: fields.missing,
            });
        }
        if let Some(detail) = fields.invalid_layer_map {
            return Err(ConfigError::InvalidLayerMap { source, detail });
        }

        config.validate(&source)?;
        Ok(config)
    }

    /// Reads and parses a checkpoint configuration file.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when the file cannot be read, does not contain JSON, or
    /// describes an invalid model architecture.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let document = fs::read_to_string(path).map_err(|error| ConfigError::Read {
            path: path.to_path_buf(),
            error: error.to_string(),
        })?;
        let root = serde_json::from_str(&document).map_err(|error| ConfigError::InvalidJson {
            source: path.display().to_string(),
            error: error.to_string(),
        })?;
        Self::from_value(&root, path.display().to_string())
    }

    /// True when a zero-based layer index is a Gated MLA layer.
    #[must_use]
    pub fn is_mla(&self, zero_based_layer: usize) -> bool {
        self.full_attn_layers
            .iter()
            .any(|&one_based_layer| one_based_layer == zero_based_layer + 1)
    }

    /// True when a zero-based layer index is a Kimi Delta Attention layer.
    #[must_use]
    pub fn is_kda(&self, zero_based_layer: usize) -> bool {
        !self.is_mla(zero_based_layer)
    }

    /// True when a zero-based layer uses the dense, non-routed MLP.
    #[must_use]
    pub fn is_dense(&self, zero_based_layer: usize) -> bool {
        zero_based_layer < self.first_k_dense_replace
    }

    fn validate(&self, source: &str) -> Result<(), ConfigError> {
        if self.num_hidden_layers == 0 || self.hidden_size == 0 || self.vocab_size == 0 {
            return Err(ConfigError::InvalidStructure {
                source: source.to_owned(),
                detail: "layers, hidden_size, and vocab_size must all be positive".to_owned(),
            });
        }
        if self.full_attn_layers.len() >= self.num_hidden_layers {
            return Err(ConfigError::InvalidStructure {
                source: source.to_owned(),
                detail: format!(
                    "{} of {} layers are full attention, leaving no KDA layers",
                    self.full_attn_layers.len(),
                    self.num_hidden_layers
                ),
            });
        }
        if let Some((index, layer)) = self
            .full_attn_layers
            .iter()
            .copied()
            .enumerate()
            .find(|(_, layer)| *layer == 0 || *layer > self.num_hidden_layers)
        {
            return Err(ConfigError::InvalidStructure {
                source: source.to_owned(),
                detail: format!(
                    "full_attn_layers[{index}] = {layer} is outside one-based 1..{}",
                    self.num_hidden_layers
                ),
            });
        }
        if self.num_experts_per_token > MAX_TOPK {
            return Err(ConfigError::InvalidStructure {
                source: source.to_owned(),
                detail: format!(
                    "top-{} exceeds MAX_TOPK ({MAX_TOPK})",
                    self.num_experts_per_token
                ),
            });
        }
        if self.num_experts_per_token > self.num_experts {
            return Err(ConfigError::InvalidStructure {
                source: source.to_owned(),
                detail: format!(
                    "top-{} selects more than {} routed experts",
                    self.num_experts_per_token, self.num_experts
                ),
            });
        }
        if self.attn_res_block_size == 0 {
            return Err(ConfigError::InvalidStructure {
                source: source.to_owned(),
                detail: "attn_res_block_size must be positive".to_owned(),
            });
        }
        if self.short_conv_kernel_size == 0 {
            return Err(ConfigError::InvalidStructure {
                source: source.to_owned(),
                detail: "short_conv_kernel_size must be positive".to_owned(),
            });
        }
        Ok(())
    }
}

/// A configuration failure that must stop model construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    Read {
        path: PathBuf,
        error: String,
    },
    InvalidJson {
        source: String,
        error: String,
    },
    MissingFields {
        source: String,
        fields: Vec<&'static str>,
    },
    InvalidLayerMap {
        source: String,
        detail: String,
    },
    InvalidStructure {
        source: String,
        detail: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, error } => {
                write!(formatter, "cannot read {}: {error}", path.display())
            }
            Self::InvalidJson { source, error } => {
                write!(formatter, "{source} is not valid JSON: {error}")
            }
            Self::MissingFields { source, fields } => write!(
                formatter,
                "{source} is missing {} required field(s): {}",
                fields.len(),
                fields.join(", ")
            ),
            Self::InvalidLayerMap { source, detail }
            | Self::InvalidStructure { source, detail } => {
                write!(
                    formatter,
                    "{source} has an invalid model configuration: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

struct ConfigFields<'a> {
    text: Option<&'a Map<String, Value>>,
    linear_attention: Option<&'a Map<String, Value>>,
    root: Option<&'a Map<String, Value>>,
    missing: Vec<&'static str>,
    invalid_layer_map: Option<String>,
}

impl<'a> ConfigFields<'a> {
    fn new(source: &'a Value) -> Self {
        let root = source.as_object();
        let text = root
            .and_then(|object| object.get("text_config"))
            .and_then(Value::as_object);
        let base = text.or(root);
        let linear_attention = base
            .and_then(|object| object.get("linear_attn_config"))
            .and_then(Value::as_object);
        Self {
            text,
            linear_attention,
            root,
            missing: Vec::new(),
            invalid_layer_map: None,
        }
    }

    fn find(&self, primary: &'static str, alias: Option<&'static str>) -> Option<&'a Value> {
        [self.text, self.linear_attention, self.root]
            .into_iter()
            .flatten()
            .find_map(|object| {
                object
                    .get(primary)
                    .or_else(|| alias.and_then(|alternative| object.get(alternative)))
            })
    }

    fn required_usize(&mut self, primary: &'static str, alias: Option<&'static str>) -> usize {
        self.find(primary, alias)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or_else(|| {
                self.missing.push(primary);
                0
            })
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "the C reference stores every configuration float as float32"
    )]
    fn required_f32(&mut self, primary: &'static str, alias: Option<&'static str>) -> f32 {
        self.find(primary, alias)
            .and_then(Value::as_f64)
            .map_or_else(
                || {
                    self.missing.push(primary);
                    0.0
                },
                |value| value as f32,
            )
    }

    fn optional_bool(
        &self,
        primary: &'static str,
        alias: Option<&'static str>,
        default: bool,
    ) -> bool {
        self.find(primary, alias)
            .and_then(|value| {
                value
                    .as_bool()
                    .or_else(|| value.as_f64().map(|number| number != 0.0))
            })
            .unwrap_or(default)
    }

    fn full_attn_layers(&mut self) -> Vec<usize> {
        let Some(array) = self
            .find("full_attn_layers", None)
            .and_then(Value::as_array)
        else {
            self.missing.push("full_attn_layers");
            return Vec::new();
        };
        if array.is_empty() {
            self.missing.push("full_attn_layers");
            return Vec::new();
        }
        if array.len() > MAX_FULL_ATTN_LAYERS {
            self.invalid_layer_map = Some(format!(
                "full_attn_layers contains {} entries, exceeding the {MAX_FULL_ATTN_LAYERS}-layer limit",
                array.len()
            ));
            return Vec::new();
        }
        let mut layers = Vec::with_capacity(array.len());
        for (index, value) in array.iter().enumerate() {
            if let Some(layer) = value.as_u64().and_then(|layer| usize::try_from(layer).ok()) {
                layers.push(layer);
            } else {
                self.invalid_layer_map = Some(format!(
                    "full_attn_layers[{index}] is not an unsigned layer index"
                ));
                return Vec::new();
            }
        }
        layers
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ConfigError, K3Config};

    fn fixture_config() -> Value {
        let document: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/ref_k3.json"
        )))
        .expect("reference fixture is JSON");
        document["config"].clone()
    }

    #[test]
    fn parses_the_flat_reference_fixture() {
        let config = K3Config::from_value(&fixture_config(), "reference fixture")
            .expect("fixture config parses");

        assert_eq!(config.hidden_size, 128);
        assert_eq!(config.num_hidden_layers, 13);
        assert_eq!(config.vocab_size, 256);
        assert_eq!(config.kda_num_heads, 4);
        assert_eq!(config.num_experts_per_token, 2);
        assert_eq!(config.full_attn_layers, [4, 8, 12, 13]);
        assert!(config.is_mla(3));
        assert!(config.is_mla(12));
        assert!(config.is_kda(0));
        assert!(config.is_dense(0));
        assert!(!config.is_dense(1));
    }

    #[test]
    fn parses_the_released_nested_shape() {
        let mut text = fixture_config();
        let object = text.as_object_mut().expect("fixture config is an object");
        let linear = json!({
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        });
        object.insert("linear_attn_config".to_owned(), linear);

        let nested = json!({ "text_config": text });
        let config = K3Config::from_value(&nested, "nested fixture").expect("nested config parses");

        assert_eq!(config.kda_num_heads, 4);
        assert_eq!(config.kda_head_dim, 16);
        assert_eq!(config.full_attn_layers.len(), 4);
    }

    #[test]
    fn rejects_every_checked_negative_fixture() {
        for (fixture, document) in [
            (
                "no_layermap",
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../tests/fixtures/cfg/no_layermap.json"
                )),
            ),
            (
                "bad_layer_index",
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../tests/fixtures/cfg/bad_layer_index.json"
                )),
            ),
            (
                "bad_topk",
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../tests/fixtures/cfg/bad_topk.json"
                )),
            ),
        ] {
            let document: Value = serde_json::from_str(document).expect("negative fixture is JSON");
            assert!(
                K3Config::from_value(&document, fixture).is_err(),
                "{fixture} must fail"
            );
        }
    }

    #[test]
    fn reports_missing_fields_together() {
        let invalid = json!({ "hidden_size": 128 });
        let error =
            K3Config::from_value(&invalid, "partial config").expect_err("partial config must fail");
        let ConfigError::MissingFields { fields, .. } = error else {
            panic!("missing fields should be reported as one error");
        };
        assert!(fields.contains(&"num_hidden_layers"));
        assert!(fields.contains(&"full_attn_layers"));
    }

    #[test]
    fn optional_flags_keep_the_c_reader_defaults() {
        let mut config = fixture_config();
        let object = config.as_object_mut().expect("fixture config is an object");
        object.remove("mla_use_output_gate");
        object.remove("moe_renormalize");
        object.remove("latent_moe_use_norm");

        let config = K3Config::from_value(&config, "optional flags").expect("defaults are valid");
        assert!(config.mla_use_output_gate);
        assert!(config.moe_renormalize);
        assert!(config.latent_moe_use_norm);
    }
}
