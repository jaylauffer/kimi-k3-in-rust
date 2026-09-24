//! Bounded local Core ML/ANE projection probe. No full-model acceleration claim.

#[cfg(target_os = "macos")]
mod probe;

fn main() {
    if std::env::args().len() == 1 || std::env::args().any(|a| a == "--help" || a == "-h") {
        println!(
            "k3-ane: validate and benchmark a dense projection using public Core ML APIs.
Usage: k3-ane --work-dir DIR [options]
  --work-dir DIR     required; new artifacts are created, never overwritten
  --width N          optional synthetic square matrix width (default 1024; max 4096)
  --positions N      optional token positions (default 16; max 64)
  --iterations N     optional warmed repetitions per policy (default 7; 3..100)
  --checkpoint DIR   optional local Kimi weights; requires --tensor
  --tensor NAME      optional exact rank-2 BF16 tensor name; requires --checkpoint
  --help, -h         print this description and exit
Example: k3-ane --work-dir /tmp/k3-ane-first --width 1024 --positions 16
Reports compile/load/first/warm latency, numerical error and planned placement.
CPU+NPU permits CPU fallback. No network, private APIs, or system tuning.
Runtime hardware evidence and end-to-end Kimi speedup are separate gates."
        );
        return;
    }
    #[cfg(target_os = "macos")]
    if let Err(error) = probe::run() {
        eprintln!("k3-ane: {error}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("k3-ane: Core ML backend is available only on macOS 15+; run --help");
        std::process::exit(1);
    }
}
