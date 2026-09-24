//! `--accel`: where the trunk's bf16 dense products run.
//!
//! `cpu` is the reference path, bit-identical to the C engine. `ane` sends every bf16
//! trunk projection and the LM head, batched over the positions of a forward pass, and
//! every routed expert's three MXFP4 matrices, batched over the tokens that selected it,
//! to the Apple Neural Engine through loadngo's Core ML dense engine (macOS only). It
//! computes in fp16, so its logits differ slightly from `cpu`; any product it refuses
//! (a value outside fp16's range, a Core ML error) is computed on the CPU instead and
//! counted. Routing, attention recurrences, norms and activations stay on the CPU.
//! Products that a layer hands over together (Kimi Linear does, a step at a time) are
//! pipelined: the next weight converts to fp16 on a helper thread while the Neural
//! Engine computes the current one.

use kimi_k3_core::layer::Accel;
#[cfg(target_os = "macos")]
use kimi_k3_core::layer::DenseAccel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelKind {
    Cpu,
    Ane,
}

impl AccelKind {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "ane" => Ok(Self::Ane),
            other => Err(format!(
                "--accel must be cpu or ane, not {other}; run --help"
            )),
        }
    }
}

/// The selected device, owned for the whole process.
pub enum Device {
    Cpu,
    #[cfg(target_os = "macos")]
    Ane(Box<ane::Ane>),
}

impl Device {
    pub fn open(kind: AccelKind) -> Result<Self, String> {
        match kind {
            AccelKind::Cpu => Ok(Self::Cpu),
            #[cfg(target_os = "macos")]
            AccelKind::Ane => ane::Ane::new().map(|device| Self::Ane(Box::new(device))),
            #[cfg(not(target_os = "macos"))]
            AccelKind::Ane => Err("--accel ane needs macOS 15+ (Core ML); use --accel cpu".into()),
        }
    }

    pub fn accel(&self) -> Accel<'_> {
        match self {
            Self::Cpu => None,
            #[cfg(target_os = "macos")]
            Self::Ane(device) => Some(&**device as &dyn DenseAccel),
        }
    }

    /// One line for `/stats` and run summaries; empty for the CPU.
    pub fn summary(&self) -> String {
        match self {
            Self::Cpu => String::new(),
            #[cfg(target_os = "macos")]
            Self::Ane(device) => device.summary(),
        }
    }
}

#[cfg(target_os = "macos")]
mod ane {
    use super::DenseAccel;
    use kimi_k3_core::expert::MXFP4_GROUP_SIZE;
    use kimi_k3_core::layer::Bf16Job;
    use loadngo_coreml::dense::{DenseEngine, Job, MX_BLOCK};
    use loadngo_inference::compute::ComputePolicy;
    use std::cell::{Cell, RefCell};

    // Kimi's MXFP4 groups are the OCP MX block the engine implements.
    const _: () = assert!(MXFP4_GROUP_SIZE == MX_BLOCK);

    pub struct Ane {
        engine: RefCell<DenseEngine>,
        declined: Cell<u64>,
    }

    impl Ane {
        pub fn new() -> Result<Self, String> {
            Ok(Self {
                engine: RefCell::new(DenseEngine::new(ComputePolicy::CpuAndNpu)?),
                declined: Cell::new(0),
            })
        }

        #[allow(clippy::cast_precision_loss)] // a byte count shown to one decimal of a GB
        pub fn summary(&self) -> String {
            let engine = self.engine.borrow();
            let s = engine.stats();
            format!(
                "ane: {} products ({:.1} GB of weights), {} predictions; \
                 convert {:.1}s, predict {:.1}s, compile/load {:.1}s; \
                 {} compiled shapes, {} not planned on the NPU; {} declined to CPU",
                s.calls,
                s.weight_bytes as f64 / 1e9,
                s.predictions,
                s.convert_s,
                s.predict_s,
                s.compile_and_load_s,
                s.models,
                s.models_not_on_npu,
                self.declined.get()
            )
        }
    }

    impl DenseAccel for Ane {
        fn matmul_bf16(
            &self,
            w: &[u16],
            x: &[f32],
            y: &mut [f32],
            rows: usize,
            inp: usize,
            out: usize,
        ) -> bool {
            let result = self
                .engine
                .borrow_mut()
                .matmul_bf16(w, x, y, rows, inp, out);
            self.accepted(result, || format!("{rows}x{inp}->{out}"))
        }

        fn run_bf16(&self, jobs: &mut [Bf16Job<'_>]) -> bool {
            let products: usize = jobs.iter().map(|j| j.parts.len()).sum();
            let mut engine_jobs: Vec<Job<'_>> = jobs
                .iter_mut()
                .map(|job| Job {
                    x: job.x,
                    rows: job.rows,
                    inputs: job.inp,
                    parts: job
                        .parts
                        .iter_mut()
                        .map(|(w, out, y)| (*w, *out, &mut **y))
                        .collect(),
                })
                .collect();
            let result = self.engine.borrow_mut().run_bf16(&mut engine_jobs);
            self.accepted(result, || format!("{products} products of one step"))
        }

        fn matmul_mxfp4(
            &self,
            packed: &[u8],
            scales: &[u8],
            x: &[f32],
            y: &mut [f32],
            rows: usize,
            inp: usize,
            out: usize,
        ) -> bool {
            let result = self
                .engine
                .borrow_mut()
                .matmul_mxfp4(packed, scales, x, y, rows, inp, out);
            self.accepted(result, || format!("{rows}x{inp}->{out}"))
        }
    }

    impl Ane {
        fn accepted(&self, result: Result<(), String>, what: impl FnOnce() -> String) -> bool {
            match result {
                Ok(()) => true,
                Err(error) => {
                    let declined = self.declined.get() + 1;
                    self.declined.set(declined);
                    if declined <= 3 {
                        eprintln!("ane: {} computed on the CPU instead: {error}", what());
                    }
                    false
                }
            }
        }
    }
}
