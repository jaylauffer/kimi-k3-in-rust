//! Keeps generation cool. Before every forward pass the gate feeds one observation of the
//! operating system's thermal pressure into a `loadngo-thermal` governor (hysteresis:
//! escalation immediate, recovery after three lower samples and 15 s):
//!
//! - nominal / fair / unavailable: proceed (transitions are printed once);
//! - serious: finish nothing new; wait, re-sampling every 2 s through a loadngo proactor
//!   deadline (the process blocks in the OS completion port: no sleep loop, no timer
//!   thread), until the governor publishes something lower. Ctrl-C ends the wait;
//! - critical: stop before the next token. The reply stays unfinished for `/continue`.
//!
//! Model weights and caches stay loaded while paused; only arithmetic stops.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use loadngo_proactor::{CompletionKind, PlatformPort, Proactor, new_platform_proactor};
use loadngo_thermal::{
    GovernorConfig, ThermalGovernor, ThermalPressure, ThermalProvider, platform_provider,
};

/// Re-sample cadence while paused (the contract's floor for fair/serious).
const PAUSED_SAMPLE: Duration = Duration::from_secs(2);

pub struct Gate {
    provider: Box<dyn ThermalProvider>,
    governor: ThermalGovernor,
    interval: Duration,
    proactor: Proactor<PlatformPort>,
    reported: Option<ThermalPressure>,
}

impl Gate {
    pub fn new() -> Result<Self, String> {
        Self::with(
            platform_provider(),
            GovernorConfig::default(),
            PAUSED_SAMPLE,
        )
    }

    fn with(
        provider: Box<dyn ThermalProvider>,
        config: GovernorConfig,
        interval: Duration,
    ) -> Result<Self, String> {
        Ok(Self {
            provider,
            governor: ThermalGovernor::new(config),
            interval,
            proactor: new_platform_proactor().map_err(|e| format!("thermal gate: {e}"))?,
            reported: None,
        })
    }

    fn sample(&mut self) -> ThermalPressure {
        self.governor
            .observe(&self.provider.observe(), Instant::now());
        let pressure = self.governor.snapshot().pressure;
        if self.reported != Some(pressure) {
            eprintln!("[thermal: {pressure}]");
            self.reported = Some(pressure);
        }
        pressure
    }

    /// Blocks in the proactor until one deadline passes.
    fn wait(&self, delay: Duration) -> Result<(), String> {
        let fired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&fired);
        self.proactor
            .handle()
            .defer_for(delay, CompletionKind::Timer, 0, move |_| {
                flag.store(true, Ordering::Release);
            })
            .map_err(|e| format!("thermal gate: {e}"))?;
        while !fired.load(Ordering::Acquire) {
            self.proactor
                .run_once()
                .map_err(|e| format!("thermal gate: {e}"))?;
        }
        Ok(())
    }

    /// Call before each forward pass. `Err` means do not start it.
    pub fn checkpoint(&mut self, cancel: &AtomicBool) -> Result<(), String> {
        let mut pressure = self.sample();
        if pressure == ThermalPressure::Serious {
            eprintln!("[thermal: pausing until the Mac cools (Ctrl-C to stop)]");
            let paused = Instant::now();
            while pressure == ThermalPressure::Serious {
                if cancel.load(Ordering::Relaxed) {
                    return Err("cancelled while waiting for the Mac to cool".into());
                }
                self.wait(self.interval)?;
                pressure = self.sample();
            }
            if pressure != ThermalPressure::Critical {
                eprintln!(
                    "[thermal: resuming after {:.0}s]",
                    paused.elapsed().as_secs_f64()
                );
            }
        }
        if pressure == ThermalPressure::Critical {
            return Err(
                "thermal pressure is critical: stopped before the next token; /continue once the Mac has cooled"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ThermalPressure::{Critical, Nominal, Serious};
    use loadngo_thermal::FakeProvider;

    fn gate(script: Vec<ThermalPressure>) -> Gate {
        let config = GovernorConfig {
            recovery_samples: 3,
            recovery_dwell: Duration::ZERO,
        };
        Gate::with(
            Box::new(FakeProvider::new(script)),
            config,
            Duration::from_millis(10),
        )
        .unwrap()
    }

    #[test]
    fn serious_pauses_through_proactor_deadlines_until_recovery() {
        let cancel = AtomicBool::new(false);
        let mut g = gate(vec![Nominal, Serious, Serious, Nominal, Nominal, Nominal]);
        g.checkpoint(&cancel).unwrap();
        let start = Instant::now();
        g.checkpoint(&cancel).unwrap();
        // Four re-samples at 10 ms: one still Serious, then three lower to recover.
        assert!(
            start.elapsed() >= Duration::from_millis(40),
            "{:?}",
            start.elapsed()
        );
        assert_eq!(g.governor.snapshot().pressure, Nominal);
    }

    #[test]
    fn critical_stops_and_cancel_ends_a_pause() {
        let cancel = AtomicBool::new(false);
        assert!(gate(vec![Critical]).checkpoint(&cancel).is_err());
        assert!(gate(vec![Serious, Critical]).checkpoint(&cancel).is_err());
        cancel.store(true, Ordering::Relaxed);
        let error = gate(vec![Serious]).checkpoint(&cancel).unwrap_err();
        assert!(error.contains("cancelled"), "{error}");
    }
}
