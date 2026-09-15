//! Fixture access shared by the op and oracle gates. The fixtures are the C engine's,
//! read from `tests/fixtures` at the repository root, never copied.

#![allow(dead_code, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::{fs, path::PathBuf};

use kimi_k3_core::config::K3Config;
use serde_json::Value;

pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

pub fn read_json(relative: &str) -> Value {
    let path = fixtures_dir().join(relative);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not JSON: {error}", path.display()))
}

/// One op fixture: `key -> {shape, data}` arrays plus scalar parameters.
pub struct Fixture {
    name: String,
    root: Value,
}

impl Fixture {
    pub fn load(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            root: read_json(&format!("ops/{name}.json")),
        }
    }

    fn get(&self, key: &str) -> &Value {
        self.root
            .get(key)
            .unwrap_or_else(|| panic!("{}: missing key {key}", self.name))
    }

    /// The flat `data` array of `key`, as the C harness reads it.
    pub fn arr(&self, key: &str) -> Vec<f32> {
        let value = self.get(key);
        let data = value.get("data").unwrap_or(value);
        data.as_array()
            .unwrap_or_else(|| panic!("{}: {key} has no data array", self.name))
            .iter()
            .map(|v| v.as_f64().expect("numeric fixture data") as f32)
            .collect()
    }

    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.get(key)["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: {key} has no shape", self.name))
            .iter()
            .map(|v| v.as_u64().expect("integer shape") as usize)
            .collect()
    }

    pub fn num(&self, key: &str) -> f64 {
        self.get(key)
            .as_f64()
            .unwrap_or_else(|| panic!("{}: {key} is not a number", self.name))
    }

    pub fn usize(&self, key: &str) -> usize {
        self.num(key) as usize
    }

    /// A scalar parameter only some fixtures carry.
    pub fn try_usize(&self, key: &str) -> Option<usize> {
        self.root
            .get(key)
            .and_then(Value::as_f64)
            .map(|v| v as usize)
    }

    pub fn boolean(&self, key: &str) -> bool {
        self.get(key)
            .as_bool()
            .unwrap_or_else(|| panic!("{}: {key} is not a boolean", self.name))
    }
}

/// The tiny model's configuration, which every op fixture was generated from.
pub fn tiny_config() -> K3Config {
    let reference = read_json("ref_k3.json");
    K3Config::from_value(&reference["config"], "ref_k3.json").expect("tiny config parses")
}

/// The published pass criterion from `ops/MANIFEST.json`.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub abs: f64,
    pub rel: f64,
}

pub fn manifest_tolerance() -> Tolerance {
    let manifest = read_json("ops/MANIFEST.json");
    let tolerance = &manifest["tolerance"];
    Tolerance {
        abs: tolerance["fp32_abs"].as_f64().expect("fp32_abs"),
        rel: tolerance["fp32_rel"].as_f64().expect("fp32_rel"),
    }
}

/// Worst `|got - want| / (abs + rel * |want|)` over the whole array; a pass is <= 1.
pub fn worst_ratio(got: &[f32], want: &[f32], tol: Tolerance) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter()
        .zip(want)
        .map(|(&g, &w)| {
            (f64::from(g) - f64::from(w)).abs() / (tol.abs + tol.rel * f64::from(w).abs())
        })
        .fold(0.0, f64::max)
}

pub fn assert_close(name: &str, got: &[f32], want: &[f32], tol: Tolerance) {
    let worst = worst_ratio(got, want, tol);
    println!("{name}: n={} worst={worst:.2}x tolerance", got.len());
    assert!(
        worst <= 1.0,
        "{name}: worst element is {worst:.2}x the tolerance"
    );
}
