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
//!
//! `gpu` (macOS) moves the trunk, LM head and resident MXFP4 experts into memory the GPU
//! and CPU share, once, and computes every decode-step product (one position) there
//! with loadngo's Metal kernels, in fp32 from the stored bf16/MXFP4 values; each step's
//! products go to the GPU as one command buffer. A prompt's positions share each weight
//! read, eight positions at a time. Products whose weights are not in GPU memory (K3,
//! streamed bf16 experts) go to the Neural Engine, as with `ane`.

use kimi_k3_core::layer::Accel;
#[cfg(target_os = "macos")]
use kimi_k3_core::layer::DenseAccel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelKind {
    Cpu,
    Ane,
    Gpu,
}

impl AccelKind {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "ane" => Ok(Self::Ane),
            "gpu" => Ok(Self::Gpu),
            other => Err(format!(
                "--accel must be cpu, ane or gpu, not {other}; run --help"
            )),
        }
    }
}

/// The selected device, owned for the whole process.
pub enum Device {
    Cpu,
    #[cfg(target_os = "macos")]
    Ane(Box<ane::Ane>),
    #[cfg(target_os = "macos")]
    Gpu(Box<gpu::Gpu>),
}

impl Device {
    pub fn open(kind: AccelKind) -> Result<Self, String> {
        match kind {
            AccelKind::Cpu => Ok(Self::Cpu),
            #[cfg(target_os = "macos")]
            AccelKind::Ane => ane::Ane::new().map(|device| Self::Ane(Box::new(device))),
            #[cfg(target_os = "macos")]
            AccelKind::Gpu => gpu::Gpu::new().map(|device| Self::Gpu(Box::new(device))),
            #[cfg(not(target_os = "macos"))]
            AccelKind::Ane => Err("--accel ane needs macOS 15+ (Core ML); use --accel cpu".into()),
            #[cfg(not(target_os = "macos"))]
            AccelKind::Gpu => Err("--accel gpu needs macOS (Metal); use --accel cpu".into()),
        }
    }

    pub fn accel(&self) -> Accel<'_> {
        match self {
            Self::Cpu => None,
            #[cfg(target_os = "macos")]
            Self::Ane(device) => Some(&**device as &dyn DenseAccel),
            #[cfg(target_os = "macos")]
            Self::Gpu(device) => Some(&**device as &dyn DenseAccel),
        }
    }

    /// One line for `/stats` and run summaries; empty for the CPU.
    pub fn summary(&self) -> String {
        match self {
            Self::Cpu => String::new(),
            #[cfg(target_os = "macos")]
            Self::Ane(device) => device.summary(),
            #[cfg(target_os = "macos")]
            Self::Gpu(device) => device.summary(),
        }
    }
}

#[cfg(target_os = "macos")]
mod ane {
    use super::DenseAccel;
    use kimi_k3_core::expert::MXFP4_GROUP_SIZE;
    use kimi_k3_core::layer::{DenseJob, WeightRef};
    use loadngo_coreml::dense::{DenseEngine, Job, MX_BLOCK, Weight};
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

        fn run_dense(&self, jobs: &mut [DenseJob<'_>]) -> bool {
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
                        .map(|(w, out, y)| {
                            let w = match *w {
                                WeightRef::Bf16(w) => Weight::Bf16(w),
                                WeightRef::Mxfp4 { packed, scales } => {
                                    Weight::Mxfp4 { packed, scales }
                                }
                            };
                            (w, *out, &mut **y)
                        })
                        .collect(),
                })
                .collect();
            let result = self.engine.borrow_mut().run(&mut engine_jobs);
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

#[cfg(target_os = "macos")]
mod gpu {
    use super::DenseAccel;
    use super::ane::Ane;
    use kimi_k3_core::layer::{
        AttentionJob, DenseJob, DeviceCache, ExpertsJob, KdaBlockJob, MlaBlockJob, RecurrenceJob,
        Shared, SharedWeight, WeightRef, WeightShape,
    };
    use loadngo_metal_compute::{
        AttentionShape, Buffer, Completed, Dispatch, Gpu as Metal, RecurrenceShape, Resident, Slice,
    };
    use loadngo_proactor::{PlatformPort, Proactor, new_platform_proactor};
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, Weak};
    use std::time::Instant;

    /// A weight moved into GPU memory; the model reads it through [`SharedWeight`].
    struct Memory(Arc<Resident>);

    impl SharedWeight for Memory {
        fn bytes(&self) -> &[u8] {
            self.0.as_bytes()
        }

        fn words(&self) -> &[u16] {
            self.0.as_words()
        }
    }

    #[derive(Clone, Copy, Default)]
    struct Stats {
        steps: u64,
        products: u64,
        weight_bytes: u64,
        gpu_s: f64,
        wall_s: f64,
        encode_s: f64,
        wait_s: f64,
        dispatches: u64,
        shared_bytes: u64,
        to_ane: u64,
        failed: u64,
        attention: u64,
        attention_gpu_s: f64,
        attention_wall_s: f64,
        attention_failed: u64,
        recurrence: u64,
        recurrence_gpu_s: f64,
        recurrence_wall_s: f64,
        recurrence_failed: u64,
        kda_blocks: u64,
        kda_block_gpu_s: f64,
        kda_block_wall_s: f64,
        kda_block_failed: u64,
        expert_layers: u64,
        expert_gpu_s: f64,
        expert_wall_s: f64,
        expert_failed: u64,
        mla_blocks: u64,
        mla_block_gpu_s: f64,
        mla_block_wall_s: f64,
        mla_block_failed: u64,
    }

    /// A KDA layer's recurrent state in GPU memory, equal to the session's own copy.
    struct StateCopy(Buffer);

    /// One attention layer's cache copied into GPU memory: `len` positions valid, room
    /// for `capacity`. Kept in the session's [`kimi_k3_core::layer::DeviceCache`].
    struct CacheCopy {
        kv: Buffer,
        rope: Buffer,
        len: usize,
        capacity: usize,
    }

    /// Staging for one step's inputs and outputs, reused and grown as needed.
    const X: usize = 0;
    const Y: usize = 1;

    pub struct Gpu {
        metal: Metal,
        proactor: Proactor<PlatformPort>,
        /// Weights moved into GPU memory, by the address the model reads them at. Weak:
        /// the model owns them, and a weight it drops is freed.
        residents: RefCell<HashMap<usize, Weak<Resident>>>,
        staging: RefCell<Option<Vec<Buffer>>>,
        /// Intermediates of a fused block, reused and grown as needed.
        work: RefCell<Option<Buffer>>,
        /// Small `f32` model constants (convolution taps, norms, decay terms) copied into
        /// GPU memory once, by the address and length the model holds them at.
        constants: RefCell<HashMap<(usize, usize), Arc<Resident>>>,
        ane: Ane,
        stats: Cell<Stats>,
    }

    fn align16(n: usize) -> usize {
        n.div_ceil(16) * 16
    }

    /// A weight in GPU memory: bf16, or MXFP4 elements with their scales.
    type ResidentWeight = (Arc<Resident>, Option<Arc<Resident>>);

    /// Encodes `y = W x` for `rows` rows of `inp` floats at `x` (buffer, byte offset, row
    /// stride) into `out` floats per row at `y`; strides are multiples of 16. Returns the
    /// weight bytes it reads.
    fn encode_product(
        batch: &mut loadngo_metal_compute::Batch<'_>,
        (w, scales): &ResidentWeight,
        (xb, x_at, x_stride): (usize, usize, usize),
        (yb, y_at, y_stride): (usize, usize, usize),
        (rows, inp, out): (usize, usize, usize),
    ) -> Result<usize, String> {
        let wi = batch.attach(w);
        let si = scales.as_ref().map(|s| batch.attach(s));
        let reads = if rows == 1 || si.is_some() {
            rows
        } else {
            rows.div_ceil(8)
        };
        let bytes = reads * (w.len() + scales.as_ref().map_or(0, |s| s.len()));
        let result = if rows == 1 || si.is_some() {
            // One position, or an MXFP4 weight: one product per position (the
            // multi-position MXFP4 kernel measured slower; see METAL_COMPUTE_PLAN).
            (0..rows).try_for_each(|r| {
                let x = Slice::new(xb, x_at + r * x_stride, inp * 4);
                let y = Slice::new(yb, y_at + r * y_stride, out * 4);
                match si {
                    Some(si) => batch.gemv_mxfp4(wi, si, x, y, out, inp),
                    None => batch.gemv_bf16(wi, x, y, out, inp),
                }
            })
        } else {
            // bf16 over several positions: one dispatch, one weight read per eight.
            let (xs, ys) = (x_stride / 4, y_stride / 4);
            let x = Slice::new(xb, x_at, ((rows - 1) * xs + inp) * 4);
            let y = Slice::new(yb, y_at, ((rows - 1) * ys + out) * 4);
            batch.gemm_bf16(wi, x, y, out, inp, rows, (xs, ys))
        };
        result.map_err(|e| e.to_string())?;
        Ok(bytes)
    }

    impl Gpu {
        pub fn new() -> Result<Self, String> {
            Ok(Self {
                metal: Metal::new().map_err(|e| e.to_string())?,
                proactor: new_platform_proactor().map_err(|e| e.to_string())?,
                residents: RefCell::new(HashMap::new()),
                staging: RefCell::new(None),
                work: RefCell::new(None),
                constants: RefCell::new(HashMap::new()),
                ane: Ane::new()?,
                stats: Cell::new(Stats::default()),
            })
        }

        fn update(&self, f: impl FnOnce(&mut Stats)) {
            let mut s = self.stats.get();
            f(&mut s);
            self.stats.set(s);
        }

        #[allow(clippy::cast_precision_loss)] // byte counts and rates shown to one decimal
        pub fn summary(&self) -> String {
            let s = self.stats.get();
            let attention_each = |v: f64| {
                if s.attention == 0 {
                    0.0
                } else {
                    v / s.attention as f64 * 1e3
                }
            };
            let mla_each = |v: f64| {
                if s.mla_blocks == 0 {
                    0.0
                } else {
                    v / s.mla_blocks as f64 * 1e3
                }
            };
            let expert_each = |v: f64| {
                if s.expert_layers == 0 {
                    0.0
                } else {
                    v / s.expert_layers as f64 * 1e3
                }
            };
            let kda_each = |v: f64| {
                if s.kda_blocks == 0 {
                    0.0
                } else {
                    v / s.kda_blocks as f64 * 1e3
                }
            };
            let recurrence_each = |v: f64| {
                if s.recurrence == 0 {
                    0.0
                } else {
                    v / s.recurrence as f64 * 1e3
                }
            };
            let per = |v: f64| {
                if s.steps == 0 {
                    0.0
                } else {
                    v / s.steps as f64 * 1e3
                }
            };
            format!(
                "gpu ({}): {:.1} GB of weights in GPU memory; {} steps, {} products, \
                 {:.1} GB read at {:.0} GB/s; per step {:.2} ms GPU, {:.2} ms wall \
                 ({:.2} ms encoding {:.0} dispatches, {:.2} ms to completion); \
                 {} steps to the ANE (weights not in GPU memory), {} GPU failures; \
                 attention {} layers on the GPU, {:.2} ms GPU, {:.2} ms wall each, \
                 {} on the CPU after a GPU failure; KDA recurrence {} layers on the GPU, \
                 {:.2} ms GPU, {:.2} ms wall each, {} on the CPU after a GPU failure; \
                 KDA blocks {} fused, {:.2} ms GPU, {:.2} ms wall each, {} run step by step \
                 after a GPU failure; expert layers {} fused, {:.2} ms GPU, {:.2} ms wall \
                 each, {} as separate products after a GPU failure; MLA blocks {} fused, \
                 {:.2} ms GPU, {:.2} ms wall each, {} run step by step after a GPU failure | {}",
                self.metal.name(),
                s.shared_bytes as f64 / 1e9,
                s.steps,
                s.products,
                s.weight_bytes as f64 / 1e9,
                if s.gpu_s > 0.0 {
                    s.weight_bytes as f64 / 1e9 / s.gpu_s
                } else {
                    0.0
                },
                per(s.gpu_s),
                per(s.wall_s),
                per(s.encode_s),
                if s.steps == 0 {
                    0.0
                } else {
                    s.dispatches as f64 / s.steps as f64
                },
                per(s.wait_s),
                s.to_ane,
                s.failed,
                s.attention,
                attention_each(s.attention_gpu_s),
                attention_each(s.attention_wall_s),
                s.attention_failed,
                s.recurrence,
                recurrence_each(s.recurrence_gpu_s),
                recurrence_each(s.recurrence_wall_s),
                s.recurrence_failed,
                s.kda_blocks,
                kda_each(s.kda_block_gpu_s),
                kda_each(s.kda_block_wall_s),
                s.kda_block_failed,
                s.expert_layers,
                expert_each(s.expert_gpu_s),
                expert_each(s.expert_wall_s),
                s.expert_failed,
                s.mla_blocks,
                mla_each(s.mla_block_gpu_s),
                mla_each(s.mla_block_wall_s),
                s.mla_block_failed,
                self.ane.summary()
            )
        }

        fn share(
            &self,
            resident: Result<Arc<Resident>, loadngo_metal_compute::Error>,
        ) -> Option<Shared> {
            let resident = resident.ok()?;
            self.update(|s| s.shared_bytes += resident.len() as u64);
            self.residents.borrow_mut().insert(
                resident.as_bytes().as_ptr() as usize,
                Arc::downgrade(&resident),
            );
            Some(Arc::new(Memory(resident)))
        }

        /// The GPU memory a weight at `address` of `len` bytes was moved into.
        fn resident(&self, address: usize, len: usize) -> Option<Arc<Resident>> {
            self.residents
                .borrow()
                .get(&address)
                .and_then(Weak::upgrade)
                .filter(|r| r.len() == len)
        }

        /// The GPU memory holding `w`, if it was moved there.
        fn resident_weight(&self, w: WeightRef<'_>) -> Option<ResidentWeight> {
            Some(match w {
                WeightRef::Bf16(w) => (self.resident(w.as_ptr() as usize, w.len() * 2)?, None),
                WeightRef::Mxfp4 { packed, scales } => (
                    self.resident(packed.as_ptr() as usize, packed.len())?,
                    Some(self.resident(scales.as_ptr() as usize, scales.len())?),
                ),
            })
        }

        /// `values` in GPU memory, copied on first use. The model's constants live as
        /// long as it does, and one model is loaded per process.
        fn constant(&self, values: &[f32]) -> Result<Arc<Resident>, String> {
            let key = (values.as_ptr() as usize, values.len());
            if let Some(resident) = self.constants.borrow().get(&key) {
                return Ok(Arc::clone(resident));
            }
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let resident = self.metal.resident(&bytes).map_err(|e| e.to_string())?;
            self.constants
                .borrow_mut()
                .insert(key, Arc::clone(&resident));
            Ok(resident)
        }

        /// A whole KDA block as one command buffer: only its input and output, the
        /// convolution history and the recurrent state cross between CPU and GPU.
        #[allow(clippy::too_many_lines)] // one block, stage by stage
        fn fused_kda(&self, job: &mut KdaBlockJob<'_>) -> Result<(), String> {
            const WORK: usize = 0;
            const STATE: usize = 1;
            let start = Instant::now();
            let (t, heads, d, kernel) = (job.t, job.heads, job.d, job.kernel);
            let (p, hist) = (heads * d, kernel - 1);
            let [wq, wk, wv, wb, wfa, wfb, wga, wgb, wo]: [WeightShape<'_>; 9] = job.weights;
            let hidden = wq.inp;
            let (fdim, gdim) = (wfa.out, wga.out);
            if [hidden, p, heads, fdim, gdim].iter().any(|n| n % 4 != 0) {
                return Err("a KDA width is not a multiple of 4 floats".into());
            }
            let resident = |w: WeightShape<'_>| {
                self.resident_weight(w.w)
                    .ok_or_else(|| "a KDA weight is not in GPU memory".to_string())
            };
            let taps = [
                self.constant(job.conv[0])?,
                self.constant(job.conv[1])?,
                self.constant(job.conv[2])?,
            ];
            let (a_log, dt_bias, o_norm) = (
                self.constant(job.a_log)?,
                self.constant(job.dt_bias)?,
                self.constant(job.o_norm)?,
            );

            // Float offsets in the work buffer, each on a 16-byte boundary.
            let mut total = 0;
            let mut take = |floats: usize| {
                let at = total;
                total += floats.div_ceil(4) * 4;
                at
            };
            let lx = take(t * hidden);
            let raw = [take(t * p), take(t * p), take(t * p)];
            let lb = take(t * heads);
            let lfa = take(t * fdim);
            let lga = take(t * gdim);
            let lz = take(t * p);
            let lgb = take(t * p);
            let conv_out = [take(t * p), take(t * p), take(t * p)];
            let la = take(t * p);
            let lo = take(t * p);
            let lout = take(t * hidden);
            let lh = take(3 * p * hist);

            let mut work = self.take_work(total)?;
            work.as_f32_mut()[lx..lx + t * hidden].copy_from_slice(job.x);
            work.as_f32_mut()[lh..lh + 3 * p * hist].copy_from_slice(job.conv_state);
            let state_len = job.recurrent.len();
            let state = match job
                .device
                .0
                .take()
                .and_then(|copy| copy.downcast::<StateCopy>().ok())
            {
                Some(copy) if copy.0.len() >= state_len * 4 => copy.0,
                _ => {
                    let mut buffer = self
                        .metal
                        .buffer(state_len * 4)
                        .map_err(|e| e.to_string())?;
                    buffer.as_f32_mut()[..state_len].copy_from_slice(job.recurrent);
                    buffer
                }
            };

            let mut batch = self
                .metal
                .batch(vec![work, state], Dispatch::Concurrent)
                .map_err(|e| e.to_string())?;
            let region = |at: usize, floats: usize| Slice::new(WORK, at * 4, floats * 4);
            let checked =
                |r: Result<(), loadngo_metal_compute::Error>| r.map_err(|error| error.to_string());
            let mut bytes = 0;
            // Everything read from x.
            for (w, at) in [
                (wq, raw[0]),
                (wk, raw[1]),
                (wv, raw[2]),
                (wb, lb),
                (wfa, lfa),
                (wga, lga),
            ] {
                bytes += encode_product(
                    &mut batch,
                    &resident(w)?,
                    (WORK, lx * 4, hidden * 4),
                    (WORK, at * 4, w.out * 4),
                    (t, hidden, w.out),
                )?;
            }
            batch.barrier();
            // Both low-rank second halves, the convolutions (reading the old history), beta.
            bytes += encode_product(
                &mut batch,
                &resident(wfb)?,
                (WORK, lfa * 4, fdim * 4),
                (WORK, lz * 4, p * 4),
                (t, fdim, p),
            )?;
            bytes += encode_product(
                &mut batch,
                &resident(wgb)?,
                (WORK, lga * 4, gdim * 4),
                (WORK, lgb * 4, p * 4),
                (t, gdim, p),
            )?;
            for i in 0..3 {
                let taps = batch.attach(&taps[i]);
                checked(batch.causal_conv_silu(
                    (
                        region(raw[i], t * p),
                        taps,
                        region(lh + i * p * hist, p * hist),
                    ),
                    region(conv_out[i], t * p),
                    t,
                    p,
                    kernel,
                ))?;
            }
            checked(batch.sigmoid_in_place(region(lb, t * heads), t * heads))?;
            batch.barrier();
            // New history, normalized q (scaled) and k, the decay.
            for (i, &raw_at) in raw.iter().enumerate() {
                checked(batch.causal_conv_history(
                    region(raw_at, t * p),
                    region(lh + i * p * hist, p * hist),
                    t,
                    p,
                    kernel,
                ))?;
            }
            #[allow(clippy::cast_precision_loss)] // head width, far below 2^24
            let qscale = 1.0_f32 / (d as f32).sqrt();
            checked(batch.l2norm_rows(region(conv_out[0], t * p), t * heads, d, 1e-6, qscale))?;
            checked(batch.l2norm_rows(region(conv_out[1], t * p), t * heads, d, 1e-6, 1.0))?;
            let (a_log, dt_bias, o_norm) = (
                batch.attach(&a_log),
                batch.attach(&dt_bias),
                batch.attach(&o_norm),
            );
            checked(batch.softplus_decay(
                (region(lz, t * p), a_log, dt_bias),
                region(la, t * p),
                t * p,
                p,
                d,
            ))?;
            batch.barrier();
            checked(batch.delta_rule_recurrence(
                (
                    region(conv_out[0], t * p),
                    region(conv_out[1], t * p),
                    region(conv_out[2], t * p),
                    region(la, t * p),
                    region(lb, t * heads),
                ),
                Slice::new(STATE, 0, state_len * 4),
                region(lo, t * p),
                RecurrenceShape {
                    t,
                    heads,
                    dk: d,
                    dv: d,
                },
            ))?;
            batch.barrier();
            checked(batch.rmsnorm_gated_rows(
                region(lo, t * p),
                (region(lgb, t * p), o_norm),
                t * heads,
                d,
                job.eps,
            ))?;
            batch.barrier();
            bytes += encode_product(
                &mut batch,
                &resident(wo)?,
                (WORK, lo * 4, p * 4),
                (WORK, lout * 4, hidden * 4),
                (t, p, hidden),
            )?;

            let done = self.submit(batch)?;
            let mut buffers = done.buffers;
            let state = buffers.pop().ok_or("state buffer missing")?;
            let work = buffers.pop().ok_or("work buffer missing")?;
            let gpu_time = done.gpu_time.map_err(|e| e.to_string())?;
            let floats = work.as_f32();
            job.out.copy_from_slice(&floats[lout..lout + t * hidden]);
            job.conv_state
                .copy_from_slice(&floats[lh..lh + 3 * p * hist]);
            job.recurrent.copy_from_slice(&state.as_f32()[..state_len]);
            *self.work.borrow_mut() = Some(work);
            job.device.0 = Some(Box::new(StateCopy(state)));
            self.update(|s| {
                s.kda_blocks += 1;
                s.kda_block_gpu_s += gpu_time.as_secs_f64();
                s.kda_block_wall_s += start.elapsed().as_secs_f64();
                s.weight_bytes += bytes as u64;
            });
            Ok(())
        }

        /// The fused blocks' work buffer, at least `floats` long, taken out for one batch.
        fn take_work(&self, floats: usize) -> Result<Buffer, String> {
            match self.work.borrow_mut().take() {
                Some(buffer) if buffer.len() >= floats * 4 => Ok(buffer),
                _ => self.metal.buffer(floats * 4).map_err(|e| e.to_string()),
            }
        }

        /// A whole MLA block as one command buffer, appending its new cache rows to the
        /// session's GPU copy; the CPU cache receives the same rows afterwards.
        #[allow(clippy::too_many_lines)] // one block, stage by stage
        fn fused_mla(&self, job: &mut MlaBlockJob<'_>) -> Result<(), String> {
            const WORK: usize = 0;
            const KV: usize = 1;
            const ROPE: usize = 2;
            let start = Instant::now();
            let (t, cached, heads) = (job.t, job.cached, job.heads);
            let (qn, qr, vh, kvr) = (job.qn, job.qr, job.vh, job.kvr);
            let [wq, wkv_a, wkv_b, wo]: [WeightShape<'_>; 4] = job.weights;
            let hidden = wq.inp;
            let (rows, row, dq, kvw) = (cached + t, heads * (qn + vh), qn + qr, kvr + qr);
            if [hidden, heads * dq, kvw, row, qr, heads * vh]
                .iter()
                .any(|n| n % 4 != 0)
            {
                return Err("an MLA width is not a multiple of 4 floats".into());
            }
            let resident = |w: WeightShape<'_>| {
                self.resident_weight(w.w)
                    .ok_or_else(|| "an MLA weight is not in GPU memory".to_string())
            };
            let norm = self.constant(job.kv_a_norm)?;
            let mut copy = self.cache_copy((&mut *job.device, cached), rows, (row, qr))?;
            // Rows before `cached` the copy lacks (computed on the CPU earlier).
            let from = copy.len.min(cached);
            copy.kv.as_f32_mut()[from * row..cached * row]
                .copy_from_slice(&job.kv[from * row..cached * row]);
            copy.rope.as_f32_mut()[from * qr..cached * qr]
                .copy_from_slice(&job.rope[from * qr..cached * qr]);

            let mut total = 0;
            let mut take = |floats: usize| {
                let at = total;
                total += floats.div_ceil(4) * 4;
                at
            };
            let (lx, lq, lct) = (take(t * hidden), take(t * heads * dq), take(t * kvw));
            let (lacc, lout) = (take(t * heads * vh), take(t * hidden));
            let mut work = self.take_work(total)?;
            work.as_f32_mut()[lx..lx + t * hidden].copy_from_slice(job.x);
            let capacity = copy.capacity;
            let mut batch = self
                .metal
                .batch(vec![work, copy.kv, copy.rope], Dispatch::Concurrent)
                .map_err(|e| e.to_string())?;
            let region = |at: usize, floats: usize| Slice::new(WORK, at * 4, floats * 4);
            let checked =
                |r: Result<(), loadngo_metal_compute::Error>| r.map_err(|error| error.to_string());
            let mut bytes = 0;
            for (w, at) in [(wq, lq), (wkv_a, lct)] {
                bytes += encode_product(
                    &mut batch,
                    &resident(w)?,
                    (WORK, lx * 4, hidden * 4),
                    (WORK, at * 4, w.out * 4),
                    (t, hidden, w.out),
                )?;
            }
            batch.barrier();
            let norm = batch.attach(&norm);
            checked(batch.rmsnorm_rows(
                (Slice::new(WORK, lct * 4, ((t - 1) * kvw + kvr) * 4), norm),
                t,
                (kvr, kvw),
                job.eps,
            ))?;
            checked(batch.copy_rows(
                (
                    Slice::new(WORK, (lct + kvr) * 4, ((t - 1) * kvw + qr) * 4),
                    kvw,
                ),
                (Slice::new(ROPE, cached * qr * 4, t * qr * 4), qr),
                t,
                qr,
            ))?;
            batch.barrier();
            bytes += encode_product(
                &mut batch,
                &resident(wkv_b)?,
                (WORK, lct * 4, kvw * 4),
                (KV, cached * row * 4, row * 4),
                (t, kvr, row),
            )?;
            batch.barrier();
            checked(batch.attention_split_key(
                region(lq, t * heads * dq),
                Slice::new(KV, 0, rows * row * 4),
                Slice::new(ROPE, 0, rows * qr * 4),
                region(lacc, t * heads * vh),
                AttentionShape {
                    t,
                    cached,
                    heads,
                    qa: qn,
                    qb: qr,
                    dv: vh,
                    scale: job.scale,
                },
            ))?;
            batch.barrier();
            bytes += encode_product(
                &mut batch,
                &resident(wo)?,
                (WORK, lacc * 4, heads * vh * 4),
                (WORK, lout * 4, hidden * 4),
                (t, heads * vh, hidden),
            )?;

            let done = self.submit(batch)?;
            let mut buffers = done.buffers;
            let (rope, kv) = (
                buffers.pop().ok_or("rope buffer missing")?,
                buffers.pop().ok_or("kv buffer missing")?,
            );
            let work = buffers.pop().ok_or("work buffer missing")?;
            let gpu_time = done.gpu_time.map_err(|e| e.to_string())?;
            job.out
                .copy_from_slice(&work.as_f32()[lout..lout + t * hidden]);
            job.kv[cached * row..rows * row]
                .copy_from_slice(&kv.as_f32()[cached * row..rows * row]);
            job.rope[cached * qr..rows * qr]
                .copy_from_slice(&rope.as_f32()[cached * qr..rows * qr]);
            *self.work.borrow_mut() = Some(work);
            job.device.0 = Some(Box::new(CacheCopy {
                kv,
                rope,
                len: rows,
                capacity,
            }));
            self.update(|s| {
                s.mla_blocks += 1;
                s.mla_block_gpu_s += gpu_time.as_secs_f64();
                s.mla_block_wall_s += start.elapsed().as_secs_f64();
                s.weight_bytes += bytes as u64;
            });
            Ok(())
        }

        /// One layer's routed experts and shared down projection as one command buffer:
        /// gate and up, `silu(gate) * up`, then down, without the CPU in between.
        #[allow(clippy::too_many_lines)] // one batch, stage by stage
        fn fused_experts(&self, job: &mut ExpertsJob<'_>) -> Result<(), String> {
            const WORK: usize = 0;
            let start = Instant::now();
            let widths = job
                .experts
                .iter()
                .flat_map(|x| [x.w1.inp, x.w1.out, x.w2.out])
                .chain([job.shared.inp, job.shared.out]);
            if widths.clone().any(|n| n % 4 != 0) {
                return Err("an expert width is not a multiple of 4 floats".into());
            }
            let mut total = 0;
            let mut take = |floats: usize| {
                let at = total;
                total += floats.div_ceil(4) * 4;
                at
            };
            let rows = job.rows;
            let (sx, sy) = (take(rows * job.shared.inp), take(rows * job.shared.out));
            // Per expert: input, gate, up, output.
            let layout: Vec<[usize; 4]> = job
                .experts
                .iter()
                .map(|x| {
                    [
                        take(x.rows * x.w1.inp),
                        take(x.rows * x.w1.out),
                        take(x.rows * x.w3.out),
                        take(x.rows * x.w2.out),
                    ]
                })
                .collect();
            let resident = |w: WeightShape<'_>| {
                self.resident_weight(w.w)
                    .ok_or_else(|| "an expert weight is not in GPU memory".to_string())
            };
            let mut work = self.take_work(total)?;
            let floats = work.as_f32_mut();
            floats[sx..sx + job.shared_x.len()].copy_from_slice(job.shared_x);
            for (x, at) in job.experts.iter().zip(&layout) {
                floats[at[0]..at[0] + x.x.len()].copy_from_slice(x.x);
            }
            let mut batch = self
                .metal
                .batch(vec![work], Dispatch::Concurrent)
                .map_err(|e| e.to_string())?;
            let region = |at: usize, floats: usize| Slice::new(WORK, at * 4, floats * 4);
            let mut bytes = encode_product(
                &mut batch,
                &resident(job.shared)?,
                (WORK, sx * 4, job.shared.inp * 4),
                (WORK, sy * 4, job.shared.out * 4),
                (rows, job.shared.inp, job.shared.out),
            )?;
            for (x, at) in job.experts.iter().zip(&layout) {
                for (w, out) in [(x.w1, at[1]), (x.w3, at[2])] {
                    bytes += encode_product(
                        &mut batch,
                        &resident(w)?,
                        (WORK, at[0] * 4, w.inp * 4),
                        (WORK, out * 4, w.out * 4),
                        (x.rows, w.inp, w.out),
                    )?;
                }
            }
            batch.barrier();
            for (x, at) in job.experts.iter().zip(&layout) {
                let n = x.rows * x.w1.out;
                batch
                    .silu_mul(region(at[1], n), region(at[2], n), n)
                    .map_err(|e| e.to_string())?;
            }
            batch.barrier();
            for (x, at) in job.experts.iter().zip(&layout) {
                bytes += encode_product(
                    &mut batch,
                    &resident(x.w2)?,
                    (WORK, at[1] * 4, x.w2.inp * 4),
                    (WORK, at[3] * 4, x.w2.out * 4),
                    (x.rows, x.w2.inp, x.w2.out),
                )?;
            }
            let done = self.submit(batch)?;
            let work = done
                .buffers
                .into_iter()
                .next()
                .ok_or("work buffer missing")?;
            let gpu_time = done.gpu_time.map_err(|e| e.to_string())?;
            let floats = work.as_f32();
            job.shared_y
                .copy_from_slice(&floats[sy..sy + job.shared_y.len()]);
            for (x, at) in job.experts.iter_mut().zip(&layout) {
                let n = x.y.len();
                x.y.copy_from_slice(&floats[at[3]..at[3] + n]);
            }
            *self.work.borrow_mut() = Some(work);
            self.update(|s| {
                s.expert_layers += 1;
                s.expert_gpu_s += gpu_time.as_secs_f64();
                s.expert_wall_s += start.elapsed().as_secs_f64();
                s.weight_bytes += bytes as u64;
            });
            Ok(())
        }

        /// Runs single-position `jobs` on the GPU as one command buffer. `None` when a
        /// weight is not in GPU memory (the caller sends the step elsewhere).
        fn step(&self, jobs: &mut [DenseJob<'_>]) -> Option<Result<(), String>> {
            // Every weight's resident memory, in job and part order.
            let mut weights = Vec::new();
            for job in jobs.iter() {
                for (w, _, _) in &job.parts {
                    weights.push(self.resident_weight(*w)?);
                }
            }
            Some(self.run(jobs, &weights))
        }

        /// The staging buffers, grown if needed, holding every job's input rows (each on
        /// a 16-byte boundary); output rows follow the same layout in the second buffer.
        fn stage(&self, jobs: &[DenseJob<'_>]) -> Result<Vec<Buffer>, String> {
            let x_len: usize = jobs.iter().map(|j| j.rows * align16(j.inp * 4)).sum();
            let y_len: usize = jobs
                .iter()
                .flat_map(|j| j.parts.iter().map(|(_, out, _)| j.rows * align16(out * 4)))
                .sum();
            let mut staging = self.take_staging(x_len, y_len)?;
            let mut at = 0;
            for job in jobs {
                let stride = align16(job.inp * 4);
                for row in job.x.chunks_exact(job.inp).take(job.rows) {
                    staging[X].as_f32_mut()[at / 4..at / 4 + job.inp].copy_from_slice(row);
                    at += stride;
                }
            }
            Ok(staging)
        }

        /// The staging pair, at least `x_len` and `y_len` bytes, taken out for one step.
        fn take_staging(&self, x_len: usize, y_len: usize) -> Result<Vec<Buffer>, String> {
            let mut staging = self.staging.borrow_mut().take().unwrap_or_default();
            if staging.len() != 2 || staging[X].len() < x_len || staging[Y].len() < y_len {
                let grow = |have: Option<&Buffer>, need: usize| {
                    let size = need.max(have.map_or(0, Buffer::len)).max(1 << 20);
                    self.metal.buffer(size).map_err(|e| e.to_string())
                };
                staging = vec![grow(staging.get(X), x_len)?, grow(staging.get(Y), y_len)?];
            }
            Ok(staging)
        }

        /// The layer's cache copy with room for `rows` positions: the session's own if it
        /// fits, else a larger one holding what the old one had.
        fn cache_copy(
            &self,
            (device, cached): (&mut DeviceCache, usize),
            rows: usize,
            (row, rope_row): (usize, usize),
        ) -> Result<CacheCopy, String> {
            let old = device
                .0
                .take()
                .and_then(|state| state.downcast::<CacheCopy>().ok())
                .map(|copy| *copy);
            match old {
                Some(copy) if copy.capacity >= rows => Ok(copy),
                old => {
                    // Doubling keeps the copying of a growing conversation linear. A new
                    // copy leaves half again as much room: a long prompt is followed by
                    // decoding, and sizing it exactly made the first decoded token grow it.
                    let capacity = rows
                        .max(old.as_ref().map_or(rows + rows / 2, |c| c.capacity * 2))
                        .max(1024);
                    let buffer = |floats: usize| {
                        self.metal
                            .buffer(floats.max(1) * 4)
                            .map_err(|e| e.to_string())
                    };
                    let mut grown = CacheCopy {
                        kv: buffer(capacity * row)?,
                        rope: buffer(capacity * rope_row)?,
                        len: 0,
                        capacity,
                    };
                    if let Some(old) = old {
                        let keep = old.len.min(cached);
                        grown.kv.as_f32_mut()[..keep * row]
                            .copy_from_slice(&old.kv.as_f32()[..keep * row]);
                        grown.rope.as_f32_mut()[..keep * rope_row]
                            .copy_from_slice(&old.rope.as_f32()[..keep * rope_row]);
                        grown.len = keep;
                    }
                    Ok(grown)
                }
            }
        }

        /// Attention on the GPU against the session's cache copy, brought up to date with
        /// the rows it lacks. On success the copy goes back into the session.
        fn attend(&self, job: &mut AttentionJob<'_>) -> Result<(), String> {
            let start = Instant::now();
            let row = job.heads * (job.qn + job.vh);
            let rows = job.cached + job.t;
            let mut copy = self.cache_copy((&mut *job.device, job.cached), rows, (row, job.qr))?;
            // Rows before `cached` that the copy lacks, then this call's new rows.
            let from = copy.len.min(job.cached);
            copy.kv.as_f32_mut()[from * row..rows * row]
                .copy_from_slice(&job.kv[from * row..rows * row]);
            copy.rope.as_f32_mut()[from * job.qr..rows * job.qr]
                .copy_from_slice(&job.rope[from * job.qr..rows * job.qr]);
            copy.len = rows;

            let (q_len, out_len) = (job.q.len() * 4, job.out.len() * 4);
            let mut staging = self.take_staging(q_len, out_len)?;
            staging[X].as_f32_mut()[..job.q.len()].copy_from_slice(job.q);
            let (kv_len, rope_len) = (rows * row * 4, rows * job.qr * 4);
            staging.push(copy.kv);
            staging.push(copy.rope);
            let mut batch = self
                .metal
                .batch(staging, Dispatch::Serial)
                .map_err(|e| e.to_string())?;
            batch
                .attention_split_key(
                    Slice::new(X, 0, q_len),
                    Slice::new(2, 0, kv_len),
                    Slice::new(3, 0, rope_len),
                    Slice::new(Y, 0, out_len),
                    AttentionShape {
                        t: job.t,
                        cached: job.cached,
                        heads: job.heads,
                        qa: job.qn,
                        qb: job.qr,
                        dv: job.vh,
                        scale: job.scale,
                    },
                )
                .map_err(|e| e.to_string())?;
            let done = self.submit(batch)?;
            let mut buffers = done.buffers;
            let (rope, kv) = (
                buffers.pop().ok_or("rope buffer missing")?,
                buffers.pop().ok_or("kv buffer missing")?,
            );
            let gpu_time = done.gpu_time.map_err(|e| e.to_string())?;
            job.out
                .copy_from_slice(&buffers[Y].as_f32()[..job.out.len()]);
            *self.staging.borrow_mut() = Some(buffers);
            job.device.0 = Some(Box::new(CacheCopy {
                kv,
                rope,
                len: copy.len,
                capacity: copy.capacity,
            }));
            self.update(|s| {
                s.attention += 1;
                s.attention_gpu_s += gpu_time.as_secs_f64();
                s.attention_wall_s += start.elapsed().as_secs_f64();
            });
            Ok(())
        }

        /// The KDA recurrence on the GPU. The session's state stays authoritative: it is
        /// copied in when the GPU copy is missing and copied back after every run.
        fn recur(&self, job: &mut RecurrenceJob<'_>) -> Result<(), String> {
            let start = Instant::now();
            let state_len = job.state.len();
            let state = match job
                .device
                .0
                .take()
                .and_then(|copy| copy.downcast::<StateCopy>().ok())
            {
                Some(copy) if copy.0.len() >= state_len * 4 => copy.0,
                _ => {
                    let mut buffer = self
                        .metal
                        .buffer(state_len * 4)
                        .map_err(|e| e.to_string())?;
                    buffer.as_f32_mut()[..state_len].copy_from_slice(job.state);
                    buffer
                }
            };
            let inputs = [job.q, job.k, job.v, job.alpha, job.beta];
            let x_len: usize = inputs.iter().map(|x| align16(x.len() * 4)).sum();
            let out_len = job.out.len() * 4;
            let mut staging = self.take_staging(x_len, out_len)?;
            let mut slices = Vec::with_capacity(inputs.len());
            let mut at = 0;
            for input in inputs {
                staging[X].as_f32_mut()[at / 4..at / 4 + input.len()].copy_from_slice(input);
                slices.push(Slice::new(X, at, input.len() * 4));
                at += align16(input.len() * 4);
            }
            staging.push(state);
            let mut batch = self
                .metal
                .batch(staging, Dispatch::Serial)
                .map_err(|e| e.to_string())?;
            batch
                .delta_rule_recurrence(
                    (slices[0], slices[1], slices[2], slices[3], slices[4]),
                    Slice::new(2, 0, state_len * 4),
                    Slice::new(Y, 0, out_len),
                    RecurrenceShape {
                        t: job.t,
                        heads: job.heads,
                        dk: job.dk,
                        dv: job.dv,
                    },
                )
                .map_err(|e| e.to_string())?;
            let done = self.submit(batch)?;
            let mut buffers = done.buffers;
            let state = buffers.pop().ok_or("state buffer missing")?;
            let gpu_time = done.gpu_time.map_err(|e| e.to_string())?;
            job.out
                .copy_from_slice(&buffers[Y].as_f32()[..job.out.len()]);
            job.state.copy_from_slice(&state.as_f32()[..state_len]);
            *self.staging.borrow_mut() = Some(buffers);
            job.device.0 = Some(Box::new(StateCopy(state)));
            self.update(|s| {
                s.recurrence += 1;
                s.recurrence_gpu_s += gpu_time.as_secs_f64();
                s.recurrence_wall_s += start.elapsed().as_secs_f64();
            });
            Ok(())
        }

        /// Commits `batch` and runs this device's proactor until its completion arrives.
        fn submit(&self, batch: loadngo_metal_compute::Batch<'_>) -> Result<Completed, String> {
            let slot: Arc<Mutex<Option<Completed>>> = Arc::default();
            let filled = Arc::clone(&slot);
            batch.commit(&self.proactor.handle(), move |done| {
                *filled.lock().expect("completion slot") = Some(done);
            });
            loop {
                self.proactor.run_once().map_err(|e| e.to_string())?;
                if let Some(done) = slot.lock().expect("completion slot").take() {
                    return Ok(done);
                }
            }
        }

        fn run(&self, jobs: &mut [DenseJob<'_>], weights: &[ResidentWeight]) -> Result<(), String> {
            let start = Instant::now();
            let staging = self.stage(jobs)?;
            let mut batch = self
                .metal
                .batch(staging, Dispatch::Concurrent)
                .map_err(|e| e.to_string())?;
            let (mut x_at, mut y_at, mut next) = (0, 0, 0);
            let mut bytes = 0;
            for job in jobs.iter() {
                let x_stride = align16(job.inp * 4);
                for (_, out, _) in &job.parts {
                    let y_stride = align16(out * 4);
                    bytes += encode_product(
                        &mut batch,
                        &weights[next],
                        (X, x_at, x_stride),
                        (Y, y_at, y_stride),
                        (job.rows, job.inp, *out),
                    )?;
                    next += 1;
                    y_at += job.rows * y_stride;
                }
                x_at += job.rows * x_stride;
            }

            let encoded = start.elapsed().as_secs_f64();
            let dispatches = batch.dispatches() as u64;
            let done = self.submit(batch)?;
            let waited = start.elapsed().as_secs_f64() - encoded;
            let gpu_time = done.gpu_time.map_err(|e| e.to_string());
            let staging = done.buffers;
            if gpu_time.is_ok() {
                let mut at = 0;
                for job in jobs.iter_mut() {
                    let rows = job.rows;
                    for (_, out, y) in &mut job.parts {
                        let stride = align16(*out * 4);
                        for row in y.chunks_exact_mut(*out).take(rows) {
                            row.copy_from_slice(&staging[Y].as_f32()[at / 4..at / 4 + *out]);
                            at += stride;
                        }
                    }
                }
            }
            *self.staging.borrow_mut() = Some(staging);
            let gpu_time = gpu_time?;
            let products = next as u64;
            self.update(|s| {
                s.steps += 1;
                s.products += products;
                s.weight_bytes += bytes as u64;
                s.gpu_s += gpu_time.as_secs_f64();
                s.wall_s += start.elapsed().as_secs_f64();
                s.encode_s += encoded;
                s.wait_s += waited;
                s.dispatches += dispatches;
            });
            Ok(())
        }
    }

    impl DenseAccel for Gpu {
        fn matmul_bf16(
            &self,
            w: &[u16],
            x: &[f32],
            y: &mut [f32],
            rows: usize,
            inp: usize,
            out: usize,
        ) -> bool {
            self.run_dense(&mut [DenseJob {
                x,
                rows,
                inp,
                parts: vec![(WeightRef::Bf16(w), out, y)],
            }])
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
            self.run_dense(&mut [DenseJob {
                x,
                rows,
                inp,
                parts: vec![(WeightRef::Mxfp4 { packed, scales }, out, y)],
            }])
        }

        fn run_dense(&self, jobs: &mut [DenseJob<'_>]) -> bool {
            {
                match self.step(jobs) {
                    Some(Ok(())) => return true,
                    Some(Err(error)) => {
                        self.update(|s| s.failed += 1);
                        if self.stats.get().failed <= 3 {
                            eprintln!("gpu: step sent to the Neural Engine instead: {error}");
                        }
                    }
                    None => {}
                }
            }
            self.update(|s| s.to_ane += 1);
            self.ane.run_dense(jobs)
        }

        fn share_words(&self, words: &[u16]) -> Option<Shared> {
            self.share(self.metal.resident_words(words))
        }

        fn share_bytes(&self, bytes: &[u8]) -> Option<Shared> {
            self.share(self.metal.resident(bytes))
        }

        /// Router logits through the GPU's `f32` products.
        fn routes(&self) -> bool {
            true
        }

        fn mla_block(&self, job: &mut MlaBlockJob<'_>) -> bool {
            match self.fused_mla(job) {
                Ok(()) => true,
                Err(error) => {
                    self.update(|s| s.mla_block_failed += 1);
                    if self.stats.get().mla_block_failed <= 3 {
                        eprintln!("gpu: MLA block run step by step instead: {error}");
                    }
                    false
                }
            }
        }

        fn experts(&self, job: &mut ExpertsJob<'_>) -> bool {
            match self.fused_experts(job) {
                Ok(()) => true,
                Err(error) => {
                    self.update(|s| s.expert_failed += 1);
                    if self.stats.get().expert_failed <= 3 {
                        eprintln!("gpu: experts run as separate products instead: {error}");
                    }
                    false
                }
            }
        }

        fn kda_block(&self, job: &mut KdaBlockJob<'_>) -> bool {
            match self.fused_kda(job) {
                Ok(()) => true,
                Err(error) => {
                    self.update(|s| s.kda_block_failed += 1);
                    if self.stats.get().kda_block_failed <= 3 {
                        eprintln!("gpu: KDA block run step by step instead: {error}");
                    }
                    false
                }
            }
        }

        fn recurrence(&self, job: &mut RecurrenceJob<'_>) -> bool {
            match self.recur(job) {
                Ok(()) => true,
                Err(error) => {
                    self.update(|s| s.recurrence_failed += 1);
                    if self.stats.get().recurrence_failed <= 3 {
                        eprintln!("gpu: KDA recurrence computed on the CPU instead: {error}");
                    }
                    false
                }
            }
        }

        fn attention(&self, job: &mut AttentionJob<'_>) -> bool {
            match self.attend(job) {
                Ok(()) => true,
                Err(error) => {
                    // The caller computes it on the CPU and drops any cache copy.
                    self.update(|s| s.attention_failed += 1);
                    if self.stats.get().attention_failed <= 3 {
                        eprintln!("gpu: attention computed on the CPU instead: {error}");
                    }
                    false
                }
            }
        }
    }
}
