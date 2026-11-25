#[derive(Default)]
pub struct Metrics {
    pub msm: u64,
    pub ntt: u64,
    pub quotient: u64,
    pub device_errors: u64,
    pub fallbacks: u64,
    pub declines: Vec<(&'static str, &'static str, u64)>,
    pub hook_declines: Vec<(&'static str, &'static str, u64)>,
    pub h2d_copies: u64,
    pub h2d_bytes: u64,
    pub d2h_copies: u64,
    pub d2h_bytes: u64,
    pub msm_d2h_copies: u64,
    pub msm_d2h_bytes: u64,
    pub stage: [u64; 8],
    pub peak_bytes: u64,
}

#[cfg(feature = "rc1-bench-cuda")]
pub mod device {
    use midnight_curves::Bls12;
    use midnight_gpu as gpu;
    use midnight_proofs::{
        gpu::{self as hooks, adapter},
        poly::{Coeff, LagrangeCoeff, kzg::params::ParamsKZG},
    };

    use super::Metrics;
    use crate::{Key, Result};

    /// The device model and its total memory in bytes.
    pub fn info() -> Result<(String, u64)> {
        let info = gpu::present().ok_or("CUDA device unavailable")?;
        Ok((info.name.clone(), info.total_bytes))
    }

    pub fn install(
        params: &ParamsKZG<Bls12>,
        pk: &Key,
    ) -> Result<([gpu::SrsHandle; 2], gpu::KeyHandle)> {
        if gpu::present().is_none() {
            return Err("CUDA device unavailable".into());
        }
        let srs = [params.bases::<Coeff>(), params.bases::<LagrangeCoeff>()]
            .map(|p| gpu::register_srs(adapter::affine(p)));
        let polys = pk.gpu_resident_polys().ok_or("unsupported quotient shape")?;
        let key = gpu::register_key(&polys.iter().map(|p| adapter::scalars(p)).collect::<Vec<_>>());
        if !hooks::install(hooks::GpuHooks {
            msm_g1: Some(gpu::msm_g1),
            fft: Some(gpu::fft),
            ifft: Some(gpu::ifft),
            quotient: Some(gpu::quotient),
        }) {
            return Err("hooks already installed".into());
        }
        Ok((srs, key))
    }

    pub struct Baseline {
        quotient: u64,
        stage: [u64; 8],
    }

    pub fn begin() -> Baseline {
        gpu::begin_request();
        gpu::reset_stats();
        hooks::reset_hook_declines();
        Baseline {
            quotient: gpu::stats().quotient,
            stage: gpu::xfer::stats().stage,
        }
    }

    pub fn finish(before: Baseline) -> Result<Metrics> {
        let stats = gpu::stats();
        let xfer = gpu::xfer::stats();
        let hook_declines = hooks::hook_declines();
        let peak_bytes = gpu::device_bytes().1;
        if !gpu::end_request() {
            return Err("device cleanup failed".into());
        }
        Ok(Metrics {
            msm: stats.msm,
            ntt: stats.ntt,
            quotient: stats.quotient - before.quotient,
            device_errors: stats.device_errors,
            fallbacks: hook_declines.iter().map(|d| d.2).sum(),
            declines: gpu::declines(),
            hook_declines,
            h2d_copies: xfer.h2d_copies,
            h2d_bytes: xfer.h2d_bytes,
            d2h_copies: xfer.d2h_copies,
            d2h_bytes: xfer.d2h_bytes,
            msm_d2h_copies: xfer.msm_d2h_copies,
            msm_d2h_bytes: xfer.msm_d2h_bytes,
            stage: std::array::from_fn(|i| {
                if i == 7 {
                    xfer.stage[i]
                } else {
                    xfer.stage[i] - before.stage[i]
                }
            }),
            peak_bytes,
        })
    }
}

/// The CUDA `device` module above on a `midnight-accel` backend: Metal, else
/// Vulkan.
#[cfg(all(
    any(feature = "rc1-bench-metal", feature = "rc1-bench-vulkan"),
    not(feature = "rc1-bench-cuda")
))]
pub mod device {
    use midnight_accel::{self as accel, KeyHandle, SrsHandle};
    use midnight_curves::Bls12;
    use midnight_proofs::{
        gpu::{self as hooks, adapter},
        poly::{
            Coeff, LagrangeCoeff, LagrangeDeltaCoeff, LagrangeDoubleDeltaCoeff,
            kzg::params::ParamsKZG,
        },
    };

    use super::Metrics;
    use crate::{Key, Result};

    #[cfg(feature = "rc1-bench-metal")]
    type B = midnight_metal::Metal;
    #[cfg(feature = "rc1-bench-metal")]
    const NAME: &str = "Metal";
    #[cfg(feature = "rc1-bench-metal")]
    const HOOKS: hooks::GpuHooks = hooks::GpuHooks {
        msm_g1: Some(accel::msm_g1::<B>),
        fft: Some(accel::fft::<B>),
        ifft: Some(accel::ifft::<B>),
        quotient: Some(accel::quotient::<B>),
    };

    #[cfg(not(feature = "rc1-bench-metal"))]
    type B = midnight_vulkan::Vulkan;
    #[cfg(not(feature = "rc1-bench-metal"))]
    const NAME: &str = "Vulkan";
    #[cfg(not(feature = "rc1-bench-metal"))]
    use phases::HOOKS;

    /// The device model and its memory budget in bytes (Metal: the
    /// recommended working set).
    pub fn info() -> Result<(String, u64)> {
        let info = accel::present::<B>().ok_or_else(|| format!("{NAME} device unavailable"))?;
        Ok((info.name.clone(), info.memory_budget))
    }

    /// Registers the SRS vectors and the key's resident polynomials, and
    /// installs the hooks.
    pub fn install(
        params: &ParamsKZG<Bls12>,
        pk: &Key,
    ) -> Result<([SrsHandle<B>; 4], KeyHandle<B>)> {
        if !accel::device::<B>() {
            return Err(format!("{NAME} device unavailable").into());
        }
        let srs = [
            params.bases::<Coeff>(),
            params.bases::<LagrangeCoeff>(),
            params.bases::<LagrangeDeltaCoeff>(),
            params.bases::<LagrangeDoubleDeltaCoeff>(),
        ]
        .map(|p| accel::register_srs::<B>(adapter::affine(p)));
        let polys = pk.gpu_resident_polys().ok_or("unsupported quotient shape")?;
        let key = accel::register_key::<B>(
            &polys.iter().map(|p| adapter::scalars(p)).collect::<Vec<_>>(),
        );
        if !hooks::install(HOOKS) {
            return Err("hooks already installed".into());
        }
        Ok((srs, key))
    }

    /// The crate's counters are reset at `begin`.
    pub struct Baseline;

    pub fn begin() -> Baseline {
        accel::begin_request::<B>();
        accel::reset_stats::<B>();
        hooks::reset_hook_declines();
        #[cfg(not(feature = "rc1-bench-metal"))]
        phases::reset();
        Baseline
    }

    /// The crates count no transfers: the transfer and staging counters are
    /// 0, also on a discrete Vulkan GPU.
    pub fn finish(_: Baseline) -> Result<Metrics> {
        #[cfg(not(feature = "rc1-bench-metal"))]
        phases::report();
        let stats = accel::stats::<B>();
        let hook_declines = hooks::hook_declines();
        let declines = accel::declines::<B>();
        let peak_bytes = accel::device_bytes::<B>().1;
        if !accel::end_request::<B>() {
            return Err("device cleanup failed".into());
        }
        Ok(Metrics {
            msm: stats.msm.served,
            ntt: stats.fft.served + stats.ifft.served,
            quotient: stats.quotient.served,
            device_errors: declines.iter().filter(|d| d.1 == "device-error").map(|d| d.2).sum(),
            fallbacks: hook_declines.iter().map(|d| d.2).sum(),
            declines,
            hook_declines,
            peak_bytes,
            ..Metrics::default()
        })
    }

    /// The hooks of a Vulkan run, timed, and a stderr line per proof of where
    /// its device time went.
    #[cfg(not(feature = "rc1-bench-metal"))]
    mod phases {
        use std::{
            ffi::c_void,
            sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
            time::Instant,
        };

        use midnight_accel::{self as accel, G1AffineLimbs, G1JacobianLimbs, Scalar};
        use midnight_proofs::gpu::GpuHooks;

        use super::B;

        /// Wall time inside each hook (MSM, FFT, inverse FFT, quotient),
        /// served or declined, in nanoseconds since `begin`.
        static WALL: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

        fn timed<R>(class: usize, f: impl FnOnce() -> R) -> R {
            let t = Instant::now();
            let r = f();
            WALL[class].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            r
        }

        fn msm_g1(s: &[Scalar], b: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
            timed(0, || accel::msm_g1::<B>(s, b))
        }

        fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
            timed(1, || accel::fft::<B>(a, omega, log_n))
        }

        fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
            timed(2, || accel::ifft::<B>(a, omega_inv, log_n, divisor))
        }

        /// # Safety
        /// As [`accel::quotient`].
        unsafe fn quotient(job: *const c_void, out: &mut [Scalar]) -> bool {
            // SAFETY: the prover's job, passed through unchanged.
            timed(3, || unsafe { accel::quotient::<B>(job, out) })
        }

        pub const HOOKS: GpuHooks = GpuHooks {
            msm_g1: Some(msm_g1),
            fft: Some(fft),
            ifft: Some(ifft),
            quotient: Some(quotient),
        };

        pub fn reset() {
            for w in &WALL {
                w.store(0, Relaxed);
            }
        }

        /// Per hook class the wall time inside the hook and the GPU time of
        /// the calls served, then the host's copies, zeroing and command
        /// buffers; after the first proof also the pipeline creation times.
        pub fn report() {
            let ms = |ns: u64| ns as f64 * 1e-6;
            let st = accel::stats::<B>();
            let h = accel::host_times::<B>();
            let w = |i: usize| ms(WALL[i].load(Relaxed));
            eprintln!(
                "phases_ms msm {:.1}/{:.1} fft {:.1}/{:.1} ifft {:.1}/{:.1} quotient {:.1}/{:.1} \
                 copy {:.1} zero {:.1} zero_bg {:.1} runs {} run {:.1}/{:.1} (wall/gpu)",
                w(0),
                ms(st.msm.device_nanos),
                w(1),
                ms(st.fft.device_nanos),
                w(2),
                ms(st.ifft.device_nanos),
                w(3),
                ms(st.quotient.device_nanos),
                ms(h.copy_nanos),
                ms(h.zero_nanos),
                ms(h.zero_background_nanos),
                h.runs,
                ms(h.run_nanos),
                ms(h.run_gpu_nanos),
            );
            static LISTED: AtomicBool = AtomicBool::new(false);
            let mut p = midnight_vulkan::pipeline_times();
            if !p.is_empty() && !LISTED.swap(true, Relaxed) {
                p.sort_by_key(|x| std::cmp::Reverse(x.2));
                let total: f64 = p.iter().map(|x| x.2.as_secs_f64() * 1e3).sum();
                let list: Vec<String> = p
                    .iter()
                    .map(|(k, wg, t)| format!("{k}/{wg} {:.1}", t.as_secs_f64() * 1e3))
                    .collect();
                eprintln!("pipelines_ms total {total:.1}: {}", list.join(", "));
            }
        }
    }
}
