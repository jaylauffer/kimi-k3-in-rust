//! A clean Rust port of the Kimi K3 inference engine's platform-neutral core.
//!
//! The C engine remains the behavioral oracle during the port. Every Rust module must
//! be gated against the same fixtures before it is used by the eventual Rust CLI.

#![forbid(unsafe_code)]

pub mod cache;
pub mod config;
pub mod expert;
pub mod io;
pub mod safetensors;
