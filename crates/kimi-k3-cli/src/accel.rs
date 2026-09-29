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
        AttentionJob, DenseJob, RecurrenceJob, Shared, SharedWeight, WeightRef,
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
        ane: Ane,
        stats: Cell<Stats>,
    }

    fn align16(n: usize) -> usize {
        n.div_ceil(16) * 16
    }

    impl Gpu {
        pub fn new() -> Result<Self, String> {
            Ok(Self {
                metal: Metal::new().map_err(|e| e.to_string())?,
                proactor: new_platform_proactor().map_err(|e| e.to_string())?,
                residents: RefCell::new(HashMap::new()),
                staging: RefCell::new(None),
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
                 {:.2} ms GPU, {:.2} ms wall each, {} on the CPU after a GPU failure | {}",
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

        /// Runs single-position `jobs` on the GPU as one command buffer. `None` when a
        /// weight is not in GPU memory (the caller sends the step elsewhere).
        fn step(&self, jobs: &mut [DenseJob<'_>]) -> Option<Result<(), String>> {
            // Every weight's resident memory, in job and part order.
            let mut weights = Vec::new();
            for job in jobs.iter() {
                for (w, _, _) in &job.parts {
                    weights.push(match *w {
                        WeightRef::Bf16(w) => {
                            (self.resident(w.as_ptr() as usize, w.len() * 2)?, None)
                        }
                        WeightRef::Mxfp4 { packed, scales } => (
                            self.resident(packed.as_ptr() as usize, packed.len())?,
                            Some(self.resident(scales.as_ptr() as usize, scales.len())?),
                        ),
                    });
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
            job: &mut AttentionJob<'_>,
            rows: usize,
            (row, rope_row): (usize, usize),
        ) -> Result<CacheCopy, String> {
            let old = job
                .device
                .0
                .take()
                .and_then(|state| state.downcast::<CacheCopy>().ok())
                .map(|copy| *copy);
            match old {
                Some(copy) if copy.capacity >= rows => Ok(copy),
                old => {
                    // Doubling keeps the copying of a growing conversation linear.
                    let capacity = rows
                        .max(old.as_ref().map_or(0, |c| c.capacity * 2))
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
                        let keep = old.len.min(job.cached);
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
            let mut copy = self.cache_copy(job, rows, (row, job.qr))?;
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

        #[allow(clippy::type_complexity)]
        fn run(
            &self,
            jobs: &mut [DenseJob<'_>],
            weights: &[(Arc<Resident>, Option<Arc<Resident>>)],
        ) -> Result<(), String> {
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
                    let (w, scales) = &weights[next];
                    next += 1;
                    let wi = batch.attach(w);
                    let si = scales.as_ref().map(|s| batch.attach(s));
                    let w_slice = wi;
                    let reads = if job.rows == 1 || si.is_some() {
                        job.rows
                    } else {
                        job.rows.div_ceil(8)
                    };
                    bytes += reads * (w.len() + scales.as_ref().map_or(0, |s| s.len()));
                    let result = if job.rows == 1 || si.is_some() {
                        // One position, or an MXFP4 weight: one product per position (the
                        // multi-position MXFP4 kernel measured slower; see METAL_COMPUTE_PLAN).
                        (0..job.rows).try_for_each(|r| {
                            let x = Slice::new(X, x_at + r * x_stride, job.inp * 4);
                            let y = Slice::new(Y, y_at + r * y_stride, out * 4);
                            match (scales, si) {
                                (Some(_), Some(si)) => {
                                    batch.gemv_mxfp4(w_slice, si, x, y, *out, job.inp)
                                }
                                _ => batch.gemv_bf16(w_slice, x, y, *out, job.inp),
                            }
                        })
                    } else {
                        // bf16 over several positions: one dispatch, one weight read per eight.
                        let (xs, ys) = (x_stride / 4, y_stride / 4);
                        let x = Slice::new(X, x_at, ((job.rows - 1) * xs + job.inp) * 4);
                        let y = Slice::new(Y, y_at, ((job.rows - 1) * ys + out) * 4);
                        batch.gemm_bf16(w_slice, x, y, *out, job.inp, job.rows, (xs, ys))
                    };
                    result.map_err(|e| e.to_string())?;
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
