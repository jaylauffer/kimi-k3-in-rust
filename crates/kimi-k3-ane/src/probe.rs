use kimi_k3_core::{
    ops,
    safetensors::{DType, SafeTensorIndex},
};
use loadngo_coreml::{
    CompiledModel, CoreMlModel, available_devices,
    model::{DenseShape, encode_dense},
};
use loadngo_inference::compute::{ComputePolicy, PreparedCompute, Tensor};
use loadngo_proactor::{CompletionKind, PlatformPort, Proactor, new_platform_proactor};
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::mpsc,
    time::{Duration, Instant},
};

struct Args {
    dir: PathBuf,
    width: usize,
    positions: usize,
    iterations: usize,
    checkpoint: Option<PathBuf>,
    tensor: Option<String>,
}
impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
        let mut result = Self {
            dir: PathBuf::new(),
            width: 1024,
            positions: 16,
            iterations: 7,
            checkpoint: None,
            tensor: None,
        };
        while let Some(key) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("{key} needs a value; run --help"))?;
            match key.as_str() {
                "--work-dir" => result.dir = value.into(),
                "--checkpoint" => result.checkpoint = Some(value.into()),
                "--tensor" => result.tensor = Some(value),
                "--width" | "--positions" | "--iterations" => {
                    let n = value
                        .parse()
                        .map_err(|_| format!("invalid {key}; run --help"))?;
                    match key.as_str() {
                        "--width" => result.width = n,
                        "--positions" => result.positions = n,
                        _ => result.iterations = n,
                    }
                }
                _ => return Err(format!("unknown {key}; run --help")),
            }
        }
        if result.dir.as_os_str().is_empty()
            || !(1..=4096).contains(&result.width)
            || !(1..=64).contains(&result.positions)
            || !(3..=100).contains(&result.iterations)
            || result.checkpoint.is_some() != result.tensor.is_some()
        {
            return Err(
                "missing work dir, invalid bounds or unmatched checkpoint/tensor; run --help"
                    .into(),
            );
        }
        Ok(result)
    }
}

struct Weights {
    values: Vec<f32>,
    bf16: Option<Vec<u16>>,
    shape: DenseShape,
    source: String,
}
fn weights(args: &Args) -> Result<Weights, String> {
    if let (Some(dir), Some(name)) = (&args.checkpoint, &args.tensor) {
        let index = SafeTensorIndex::open(dir).map_err(|e| e.to_string())?;
        let tensor = index.tensor(name).ok_or("tensor not found")?;
        if tensor.shape.len() != 2
            || tensor.dtype != DType::Bf16
            || tensor.numel() > 64 * 1024 * 1024
        {
            return Err("expected a BF16 matrix no larger than 64M elements".into());
        }
        let raw = index.read_raw(tensor).map_err(|e| e.to_string())?;
        let bf16: Vec<u16> = raw
            .chunks_exact(2)
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .collect();
        let values = bf16
            .iter()
            .map(|&b| f32::from_bits(u32::from(b) << 16))
            .collect();
        Ok(Weights {
            values,
            bf16: Some(bf16),
            shape: DenseShape {
                inputs: tensor.shape[1],
                outputs: tensor.shape[0],
                positions: args.positions,
            },
            source: name.clone(),
        })
    } else {
        let values = (0..args.width * args.width)
            .map(|i| sample(i, 7) / 32.0)
            .collect();
        Ok(Weights {
            values,
            bf16: None,
            shape: DenseShape {
                inputs: args.width,
                outputs: args.width,
                positions: args.positions,
            },
            source: "synthetic deterministic dense projection".into(),
        })
    }
}

#[allow(clippy::cast_precision_loss)]
fn sample(index: usize, seed: u32) -> f32 {
    let x = u32::try_from(index)
        .expect("bounded fixture")
        .wrapping_mul(1_664_525)
        .wrapping_add(seed.wrapping_mul(1_013_904_223));
    ((x >> 16) as f32 - 32768.0) / 32768.0
}

fn reference(weights: &Weights, input: &Tensor) -> Vec<f32> {
    let s = weights.shape;
    let mut result = vec![0.0; s.outputs * s.positions];
    let mut x = vec![0.0; s.inputs];
    let mut y = vec![0.0; s.outputs];
    for position in 0..s.positions {
        for (i, x) in x.iter_mut().enumerate() {
            *x = input.values()[i * s.positions + position];
        }
        if let Some(bf16) = &weights.bf16 {
            ops::matmul_bf16(&mut y, &x, bf16, s.inputs, s.outputs);
        } else {
            ops::matmul(&mut y, &x, &weights.values, s.inputs, s.outputs);
        }
        for (o, &y) in y.iter().enumerate() {
            result[o * s.positions + position] = y;
        }
    }
    result
}

fn predict(
    model: &mut CoreMlModel,
    input: Tensor,
    proactor: &Proactor<PlatformPort>,
) -> Result<Tensor, String> {
    let (tx, rx) = mpsc::sync_channel(1);
    let handle = proactor.handle();
    model.submit(
        input,
        Box::new(move |result| {
            let fallback = tx.clone();
            if let Err(e) = handle.enqueue_work(move |_| {
                let _ = tx.send(result);
            }) {
                let _ = fallback.send(Err(format!("proactor post failed: {e}")));
                let _ = handle.stop();
            }
        }),
    )?;
    loop {
        if let Ok(result) = rx.try_recv() {
            return result;
        }
        // run_once blocks in the OS completion port, not a polling sleep loop.
        if proactor.run_once().map_err(|e| e.to_string())?.stopped {
            return Err("probe deadline expired".into());
        }
    }
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> Result<(f64, f64), String> {
    if actual.len() != expected.len() || actual.iter().any(|v| !v.is_finite()) {
        return Err("invalid output".into());
    }
    let mut max = 0.0_f64;
    let mut squared = 0.0;
    let mut norm = 0.0;
    for (&a, &e) in actual.iter().zip(expected) {
        let d = f64::from(a) - f64::from(e);
        max = max.max(d.abs());
        squared += d * d;
        norm += f64::from(e).powi(2);
    }
    let relative_rms = (squared / norm.max(1e-20)).sqrt();
    if relative_rms > 0.005 {
        return Err(format!(
            "numerical gate failed: relative RMS {relative_rms} > 0.005"
        ));
    }
    Ok((max, relative_rms))
}

#[allow(clippy::too_many_lines)]
pub fn run() -> Result<(), String> {
    let args = Args::parse()?;
    std::fs::create_dir_all(&args.dir).map_err(|e| e.to_string())?;
    let weights = weights(&args)?;
    let shape = weights.shape;
    let devices = available_devices();
    eprintln!(
        "devices: {devices:?}; {} x {}, {} positions",
        shape.outputs, shape.inputs, shape.positions
    );
    let source = args.dir.join("projection.mlmodel");
    let bytes = encode_dense(shape, weights.values.clone())?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&source)
        .and_then(|mut f| f.write_all(&bytes))
        .map_err(|e| e.to_string())?;
    drop(bytes);
    let start = Instant::now();
    let compiled = CompiledModel::compile(&source)?;
    let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("compiled in {compile_ms:.1} ms");
    let proactor = new_platform_proactor().map_err(|e| e.to_string())?;
    let stop = proactor.handle();
    proactor
        .handle()
        .defer_for(
            Duration::from_secs(120),
            CompletionKind::Timer,
            0,
            move |_| {
                let _ = stop.stop();
            },
        )
        .map_err(|e| e.to_string())?;
    let mut reports = Vec::<Value>::new();
    for policy in [ComputePolicy::CpuOnly, ComputePolicy::CpuAndNpu] {
        let start = Instant::now();
        let mut model = compiled.load(policy, shape)?;
        let load_and_plan_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{policy:?}: planned {:?}", model.placements);
        let mut times = Vec::new();
        let mut scalar_times = Vec::new();
        let mut max_error = 0.0_f64;
        let mut relative_rms = 0.0_f64;
        for iteration in 0..=args.iterations {
            let seed = u32::try_from(iteration + 1).unwrap();
            let input = Tensor::new(
                shape.input_shape(),
                (0..shape.inputs * shape.positions)
                    .map(|i| sample(i, seed))
                    .collect(),
            )?;
            let start = Instant::now();
            let expected = reference(&weights, &input);
            scalar_times.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = Instant::now();
            let got = predict(&mut model, input, &proactor)?;
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            let (max, rms) = error_metrics(got.values(), &expected)?;
            max_error = max_error.max(max);
            relative_rms = relative_rms.max(rms);
        }
        let first_ms = times.remove(0);
        times.sort_by(f64::total_cmp);
        scalar_times.remove(0);
        scalar_times.sort_by(f64::total_cmp);
        let median_ms = times[times.len() / 2];
        eprintln!(
            "{policy:?}: first {first_ms:.3}ms, warm median {median_ms:.3}ms, max error {max_error:.6}, relative RMS {relative_rms:.6}"
        );
        reports.push(json!({"policy":format!("{policy:?}"),"load_and_plan_ms":load_and_plan_ms,"first_ms":first_ms,"warm_median_ms":median_ms,"warm_samples_ms":times,"rust_reference_median_ms":scalar_times[scalar_times.len()/2],"max_abs_error":max_error,"max_relative_rms":relative_rms,"placements":model.placements.iter().map(|p|json!({"operation":p.operation,"preferred":format!("{:?}",p.preferred),"supported":format!("{:?}",p.supported)})).collect::<Vec<_>>()}));
    }
    let report = json!({"source":weights.source,"input_channels":shape.inputs,"output_channels":shape.outputs,"positions":shape.positions,"compile_ms":compile_ms,"devices":format!("{devices:?}"),"policies":reports,"hardware_execution_trace":"not collected by this probe","end_to_end_kimi_speedup":"not measured","timing_includes":"input allocation/copy, native async prediction, output conversion, proactor delivery","numerical_gate":"relative RMS <= 0.005 on every input; not full-model parity"});
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.dir.join("report.json"))
        .and_then(|mut f| writeln!(f, "{json}"))
        .map_err(|e| e.to_string())?;
    println!("{json}");
    proactor.handle().stop().map_err(|e| e.to_string())?;
    proactor.run_until_stopped().map_err(|e| e.to_string())?;
    Ok(())
}
