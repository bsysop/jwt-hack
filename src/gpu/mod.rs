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
/// The table above describes the **default** backend (native Metal on macOS,
/// a stub elsewhere). Building with the optional `gpu-wgpu` feature replaces it
/// with a cross-vendor wgpu/WGSL backend that also covers NVIDIA, AMD and Intel
/// GPUs on Linux and Windows (and macOS via Metal); see [`imp`].
///
/// On an unsupported target this module compiles to a stub.  Callers should
/// check [`is_available`] before attempting GPU cracking and fall back
/// to the CPU path when it returns `false`.

// ── Stub (no GPU backend compiled in) ───────────────────────────────────────
#[cfg(all(not(target_os = "macos"), not(feature = "gpu-wgpu")))]
mod imp {
    use anyhow::Result;

    pub struct GpuCracker;

    /// Number of candidates dispatched per GPU invocation. Mirrors the macOS
    /// value so callers can reference `gpu::GPU_BATCH_SIZE` on every platform.
    pub const GPU_BATCH_SIZE: u64 = 1_000_000;

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
        pub fn crack_batch(&self, _candidates: &[u8], _offsets: &[u32]) -> Result<Vec<u32>> {
            anyhow::bail!("{}", availability_reason());
        }
    }
}

// ── Metal implementation for macOS (default backend) ────────────────────────
#[cfg(all(target_os = "macos", not(feature = "gpu-wgpu")))]
mod imp {
    use anyhow::{Context, Result};
    use metal::*;
    use std::mem;
    use std::sync::OnceLock;

    /// Source of the Metal compute kernel (`src/gpu/sha256.metal`).
    const KERNEL_SRC: &str = include_str!("sha256.metal");

    /// Number of candidates to dispatch per GPU invocation.
    pub const GPU_BATCH_SIZE: u64 = 1_000_000;

    // Cache device info so `is_available` + `availability_reason` don't each
    // call into Objective-C separately.
    static DEVICE_INFO: OnceLock<(bool, String)> = OnceLock::new();

    fn device_info() -> &'static (bool, String) {
        DEVICE_INFO.get_or_init(|| {
            match Device::system_default() {
                Some(d) => {
                    // `crack_batch` dispatches a 1M-wide grid via
                    // `dispatch_threads`, which requires *non-uniform
                    // threadgroup* support. Metal guarantees that only on Apple
                    // GPU family 4+ (A11 / Apple Silicon) and Mac family 2.
                    // Gating on these (rather than the weaker `Common2`) means a
                    // GPU we accept can actually run the dispatch, instead of
                    // reporting "available" and then erroring at dispatch time.
                    let family_ok = d.supports_family(MTLGPUFamily::Apple4)
                        || d.supports_family(MTLGPUFamily::Mac2);
                    let name: String = d.name().into();
                    if family_ok {
                        (true, name)
                    } else {
                        (
                            false,
                            format!(
                                "GPU {} lacks non-uniform threadgroup support \
                                 (needs Apple GPU family 4+ or Mac family 2)",
                                name
                            ),
                        )
                    }
                }
                None => {
                    let reason =
                        if std::path::Path::new("/System/Library/Frameworks/Metal.framework")
                            .exists()
                        {
                            "Metal framework present but no compatible GPU found".to_string()
                        } else {
                            "Metal framework not available on this platform".to_string()
                        };
                    (false, reason)
                }
            }
        })
    }

    /// Check whether a Metal-capable GPU is available on this system.
    pub fn is_available() -> bool {
        device_info().0
    }

    /// Human-readable GPU name, or reason for unavailability.
    pub fn availability_reason() -> String {
        device_info().1.clone()
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
                .map_err(|e| anyhow::anyhow!("Metal kernel compilation failed: {}", e))?;
            let kernel = library
                .get_function("hmac_sha256_verify", None)
                .map_err(|e| {
                    anyhow::anyhow!("Kernel function 'hmac_sha256_verify' not found: {}", e)
                })?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| anyhow::anyhow!("Failed to create compute pipeline: {}", e))?;

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
        pub fn crack_batch(&self, candidates: &[u8], offsets: &[u32]) -> Result<Vec<u32>> {
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

                // Wipe the plaintext candidate secrets from the shared buffer
                // before it is released. `make_buffer` copied them into GPU-
                // visible memory, so zeroizing only the caller's CPU Vec would
                // leave a second plaintext copy lingering in freed pages. The
                // GPU has finished reading (`wait_until_completed` above), so
                // this is safe. `signing_input`/`expected_sig`/`offsets` are
                // public token material and need no wiping.
                unsafe {
                    std::ptr::write_bytes(candidate_buf.contents() as *mut u8, 0, candidates.len());
                }

                Ok(matches)
            })
        }

        fn make_buffer<T: Sized>(device: &Device, data: &[T]) -> Buffer {
            let len = data.len().saturating_mul(mem::size_of::<T>()) as u64;
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

// ── Cross-vendor wgpu/WGSL implementation (opt-in: `--features gpu-wgpu`) ─────
//
// Selected on *any* OS when the `gpu-wgpu` feature is enabled, replacing the
// Metal backend above. wgpu dispatches the shared WGSL kernel through Vulkan,
// DX12, Metal or GL, so NVIDIA / AMD / Intel GPUs on Linux and Windows are
// covered in addition to Apple GPUs. The public surface is identical to the
// Metal backend so `cmd::crack` needs no backend-specific code.
#[cfg(feature = "gpu-wgpu")]
mod imp {
    use anyhow::{Context, Result};
    use std::sync::OnceLock;
    use wgpu::util::DeviceExt;

    /// WGSL source of the compute kernel (`src/gpu/hmac_sha256.wgsl`).
    const KERNEL_SRC: &str = include_str!("hmac_sha256.wgsl");

    /// Number of candidates dispatched per GPU invocation. Mirrors the Metal
    /// backend so callers can reference `gpu::GPU_BATCH_SIZE` on every platform.
    pub const GPU_BATCH_SIZE: u64 = 1_000_000;

    /// Threads per workgroup — must match `@workgroup_size` in the WGSL kernel.
    const WORKGROUP_SIZE: u32 = 64;

    // Cache adapter probe so `is_available` + `availability_reason` don't each
    // spin up an instance and enumerate adapters.
    static ADAPTER_INFO: OnceLock<(bool, String)> = OnceLock::new();

    fn new_instance() -> wgpu::Instance {
        // Compute-only: no windowing/display handle needed.
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
    }

    fn request_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter> {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("no compatible GPU adapter found: {e}"))
    }

    fn adapter_info() -> &'static (bool, String) {
        ADAPTER_INFO.get_or_init(|| match request_adapter(&new_instance()) {
            Ok(adapter) => {
                let info = adapter.get_info();
                (
                    true,
                    format!(
                        "{} ({:?}, {:?} backend)",
                        info.name, info.device_type, info.backend
                    ),
                )
            }
            Err(e) => (false, e.to_string()),
        })
    }

    /// Whether a usable GPU adapter (any wgpu backend) is available.
    pub fn is_available() -> bool {
        adapter_info().0
    }

    /// Human-readable adapter description, or the reason none is usable.
    pub fn availability_reason() -> String {
        adapter_info().1.clone()
    }

    pub struct GpuCracker {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        signing_input_buf: wgpu::Buffer,
        params_buf: wgpu::Buffer,
        expected_sig_buf: wgpu::Buffer,
        si_len: u32,
    }

    /// Pack raw bytes for upload into a WGSL `array<u32>` (little-endian words).
    /// Buffers must be a non-zero multiple of 4 bytes; the tail is zero-padded
    /// and never read because the kernel bounds every access by `offsets`.
    fn pad_to_words(bytes: &[u8]) -> Vec<u8> {
        let mut v = bytes.to_vec();
        while v.len() % 4 != 0 {
            v.push(0);
        }
        if v.is_empty() {
            v.extend_from_slice(&[0u8; 4]);
        }
        v
    }

    /// Encode the 32-byte signature as 8 big-endian words so the kernel can
    /// compare the final SHA-256 state word-for-word (matching the WGSL, which
    /// keeps the digest in big-endian word form).
    fn expected_sig_words(expected_sig: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(32);
        for chunk in expected_sig.chunks(4) {
            let mut w = [0u8; 4];
            w[..chunk.len()].copy_from_slice(chunk);
            out.extend_from_slice(&u32::from_be_bytes(w).to_le_bytes());
        }
        out
    }

    impl GpuCracker {
        pub fn new(signing_input: &[u8], expected_sig: &[u8]) -> Result<Self> {
            let instance = new_instance();
            let adapter = request_adapter(&instance)?;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("jwt-hack-gpu"),
                    required_features: wgpu::Features::empty(),
                    // Use the adapter's full limits so large candidate buffers
                    // (up to GPU_BATCH_SIZE * max_len bytes) are not rejected by
                    // the conservative WebGPU defaults.
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    ..Default::default()
                }))
                .context("failed to create wgpu device")?;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hmac_sha256"),
                source: wgpu::ShaderSource::Wgsl(KERNEL_SRC.into()),
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("hmac_sha256_bgl"),
                    entries: &[
                        storage_entry(0, true),  // signing_input
                        uniform_entry(1),        // params
                        storage_entry(2, true),  // expected_sig
                        storage_entry(3, true),  // candidates
                        storage_entry(4, true),  // offsets
                        storage_entry(5, false), // results (read_write)
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hmac_sha256_pl"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hmac_sha256_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("hmac_sha256_verify"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let signing_input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("signing_input"),
                contents: &pad_to_words(signing_input),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let expected_sig_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("expected_sig"),
                contents: &expected_sig_words(expected_sig),
                usage: wgpu::BufferUsages::STORAGE,
            });
            // Params is {si_len, num_threads}; num_threads is rewritten per
            // batch. Padded to 16 bytes to satisfy uniform-buffer alignment.
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            Ok(Self {
                device,
                queue,
                pipeline,
                bind_group_layout,
                signing_input_buf,
                params_buf,
                expected_sig_buf,
                si_len: signing_input.len() as u32,
            })
        }

        /// Run HMAC-SHA256 verification on a batch of candidate secrets.
        ///
        /// Layout matches the Metal backend: `candidates` is packed secret
        /// bytes and candidate `tid-1` occupies `candidates[offsets[tid-1]..
        /// offsets[tid]]`. Returns the 1-based thread ids that matched.
        pub fn crack_batch(&self, candidates: &[u8], offsets: &[u32]) -> Result<Vec<u32>> {
            let num_threads = offsets.len() as u32; // slot 0 is reserved
            if num_threads <= 1 {
                return Ok(Vec::new());
            }

            let candidate_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("candidates"),
                    contents: &pad_to_words(candidates),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let offset_bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
            let offset_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("offsets"),
                    contents: &offset_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let results_len = std::mem::size_of_val(offsets) as u64;
            // wgpu zero-initialises buffers, so results starts all-zero.
            let results_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("results"),
                size: results_len,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("results_staging"),
                size: results_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Upload params {si_len, num_threads} (padded to 16 bytes).
            let mut params = [0u8; 16];
            params[0..4].copy_from_slice(&self.si_len.to_le_bytes());
            params[4..8].copy_from_slice(&num_threads.to_le_bytes());
            self.queue.write_buffer(&self.params_buf, 0, &params);

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hmac_sha256_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    bind(0, &self.signing_input_buf),
                    bind(1, &self.params_buf),
                    bind(2, &self.expected_sig_buf),
                    bind(3, &candidate_buf),
                    bind(4, &offset_buf),
                    bind(5, &results_buf),
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("hmac_sha256_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hmac_sha256_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = num_threads.div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&results_buf, 0, &staging_buf, 0, results_len);
            self.queue.submit(Some(encoder.finish()));

            // Map the staging buffer and block until the GPU finishes.
            let slice = staging_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| anyhow::anyhow!("GPU poll failed: {e}"))?;
            rx.recv()
                .context("GPU result channel closed")?
                .map_err(|e| anyhow::anyhow!("failed to map GPU results: {e}"))?;

            let data = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("failed to read GPU results: {e}"))?;
            let mut matches = Vec::new();
            for (i, word) in data.chunks_exact(4).enumerate() {
                if i == 0 {
                    continue; // thread 0 reserved
                }
                if u32::from_le_bytes([word[0], word[1], word[2], word[3]]) != 0 {
                    matches.push(i as u32);
                }
            }
            drop(data);
            staging_buf.unmap();

            Ok(matches)
        }
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        }
    }
}

pub use imp::*;
