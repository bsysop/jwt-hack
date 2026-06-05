/// GPU-accelerated HMAC-SHA256 cracking via Metal.
///
/// # Compatibility
///
/// | Platform                  | GPU            | Supported |
/// |---------------------------|----------------|-----------|
/// | macOS Apple Silicon (M1+) | Integrated     | ✅        |
/// | macOS Intel + AMD GPU     | Dedicated      | ✅        |
/// | macOS Intel + Intel GPU   | Integrated     | ✅ (weak) |
/// | macOS Intel + NVIDIA GPU  | Dedicated      | ❌        |
/// | macOS VM / Hackintosh     | None / partial | ❌        |
/// | Linux / Windows (any GPU) | —              | ❌        |
///
/// On non-macOS targets this module compiles to a stub.  Callers should
/// check [`is_available`] before attempting GPU cracking and fall back
/// to the CPU path when it returns `false`.

// ── Stub for non-macOS ──────────────────────────────────────────────────────
#[cfg(not(target_os = "macos"))]
mod imp {
    use anyhow::Result;

    pub struct GpuCracker;

    /// Always `false` on non-macOS — Metal does not exist.
    pub fn is_available() -> bool {
        false
    }

    /// Returns a human-readable description of why GPU is unavailable.
    pub fn availability_reason() -> String {
        "GPU mode requires macOS with Apple Silicon or an AMD GPU".to_string()
    }

    impl GpuCracker {
        pub fn new(_signing_input: &[u8], _expected_sig: &[u8]) -> Result<Self> {
            anyhow::bail!("{}", availability_reason());
        }

        /// Returns the indices (thread IDs) that matched, if any.
        pub fn crack_batch(
            &self,
            _candidates: &[u8],
            _offsets: &[u32],
        ) -> Result<Vec<u32>> {
            anyhow::bail!("{}", availability_reason());
        }
    }
}

// ── Metal implementation for macOS ──────────────────────────────────────────
#[cfg(target_os = "macos")]
mod imp {
    use anyhow::{Context, Result};
    use metal::*;
    use std::mem;

    /// Source of the Metal compute kernel (`src/gpu/sha256.metal`).
    const KERNEL_SRC: &str = include_str!("sha256.metal");

    /// Number of candidates to dispatch per GPU invocation.
    pub const GPU_BATCH_SIZE: u64 = 1_000_000;

    /// Check whether a Metal-capable GPU is available on this system.
    /// Safe to call at any time; does not compile the kernel.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Human-readable reason for GPU unavailability, or GPU name if available.
    pub fn availability_reason() -> String {
        match Device::system_default() {
            Some(d) => d.name().into(),
            None => {
                if std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists() {
                    "Metal framework present but no compatible GPU found".to_string()
                } else {
                    "Metal framework not available on this platform".to_string()
                }
            }
        }
    }

    pub struct GpuCracker {
        device: Device,
        cmd_queue: CommandQueue,
        _library: Library,
        pipeline: ComputePipelineState,
        signing_input_buf: Buffer,
        si_len_buf: Buffer,
        expected_sig_buf: Buffer,
    }

    // SAFETY: Metal objects are safe to send across threads.
    unsafe impl Send for GpuCracker {}
    unsafe impl Sync for GpuCracker {}

    impl GpuCracker {
        /// Initialise the Metal device, compile the kernel, and upload the
        /// constant buffers (`signing_input`, `expected_sig`).
        pub fn new(signing_input: &[u8], expected_sig: &[u8]) -> Result<Self> {
            let device =
                Device::system_default().context("No Metal-capable GPU found on this system")?;

            // Compile the kernel from embedded MSL source at runtime.
            let compile_opts = CompileOptions::new();
            let library = device
                .new_library_with_source(KERNEL_SRC, &compile_opts)
                .map_err(|e| {
                    anyhow::anyhow!("Metal kernel compilation failed: {}", e)
                })?;
            let kernel = library
                .get_function("hmac_sha256_verify", None)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Kernel function 'hmac_sha256_verify' not found: {}",
                        e
                    )
                })?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| {
                    anyhow::anyhow!("Failed to create compute pipeline: {}", e)
                })?;

            // Upload constant buffers (shared CPU/GPU on Apple Silicon).
            let cmd_queue = device.new_command_queue();
            let signing_input_buf = Self::make_buffer(&device, signing_input);
            let si_len = signing_input.len() as u32;
            let si_len_buf = Self::make_buffer(&device, &si_len.to_ne_bytes());
            let expected_sig_buf = Self::make_buffer(&device, expected_sig);

            Ok(Self {
                device,
                cmd_queue,
                _library: library,
                pipeline,
                signing_input_buf,
                si_len_buf,
                expected_sig_buf,
            })
        }

        /// Run HMAC-SHA256 verification on a batch of candidate secrets.
        ///
        /// `candidates` contains packed secret bytes. `offsets` maps thread
        /// indices to byte ranges: candidate `tid` occupies
        /// `candidates[offsets[tid-1]..offsets[tid]]`.
        ///
        /// Returns the thread IDs (1-based) of matching candidates.
        pub fn crack_batch(
            &self,
            candidates: &[u8],
            offsets: &[u32],
        ) -> Result<Vec<u32>> {
            let num_candidates = offsets.len().saturating_sub(1) as u64;
            if num_candidates == 0 {
                return Ok(Vec::new());
            }

            // Wrap in an autorelease pool so Metal temporary objects
            // (command buffer, encoder, per-batch buffers) are freed
            // immediately rather than accumulating across batches.
            objc::rc::autoreleasepool(|| {
                // Upload per-batch buffers.
                let candidate_buf = Self::make_buffer(&self.device, candidates);
                let offset_buf = Self::make_buffer(&self.device, offsets);
                let results_len = offsets.len() as u64 * mem::size_of::<u32>() as u64;
                let results_buf = self
                    .device
                    .new_buffer(results_len, MTLResourceOptions::StorageModeShared);
                // Metal does not guarantee zero-initialised buffers — stale
                // values from a prior dispatch could cause a false positive.
                unsafe {
                    std::ptr::write_bytes(
                        results_buf.contents() as *mut u8,
                        0,
                        results_len as usize,
                    );
                }

                // Build command (reuse the persistent command queue).
                let cmd_buf = self.cmd_queue.new_command_buffer();
                let encoder = cmd_buf.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.pipeline);

                encoder.set_buffer(0, Some(&self.signing_input_buf), 0);
                encoder.set_buffer(1, Some(&self.si_len_buf), 0);
                encoder.set_buffer(2, Some(&self.expected_sig_buf), 0);
                encoder.set_buffer(3, Some(&candidate_buf), 0);
                encoder.set_buffer(4, Some(&offset_buf), 0);
                encoder.set_buffer(5, Some(&results_buf), 0);

                let grid = MTLSize {
                    width: offsets.len() as u64,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: self
                        .pipeline
                        .max_total_threads_per_threadgroup()
                        .min(grid.width),
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_threads(grid, tg);
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();

                // Collect matching indices.
                let results_ptr = results_buf.contents() as *const u32;
                let mut matches = Vec::new();
                for i in 1..offsets.len() {
                    let val = unsafe { *results_ptr.add(i) };
                    if val != 0 {
                        matches.push(i as u32);
                    }
                }
                Ok(matches)
            })
        }

        fn make_buffer<T: Sized>(device: &Device, data: &[T]) -> Buffer {
            let len = (data.len() * mem::size_of::<T>()) as u64;
            let buf = device.new_buffer(len, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    len as usize,
                );
            }
            buf
        }
    }
}

pub use imp::*;
