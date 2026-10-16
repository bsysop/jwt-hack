# jwt-hack Benchmarks

Benchmarks are run on a real brute-force crack of a 3-character lowercase HS256 secret (`abc`,
18 278 total candidates, `--chars=abcdefghijklmnopqrstuvwxyz --max=3`) plus Criterion micro-benchmarks
for encode / decode / verify. All binaries built with `cargo build --release`.

---

## GPU brute-force cracking

Target: `secret = "abc"`, 18 278 candidates (len 1–3, a–z).

### Metal GPU vs wgpu GPU vs CPU

| Backend | Build feature | Machine | Keys/sec | Wall time | vs CPU |
|---------|--------------|---------|----------|-----------|--------|
| **Metal** (default) | _(none)_ | M4 Pro · macOS 15 | **1 662 032** | 45 ms | **49×** |
| **Metal** (default) | _(none)_ | M1 Max · macOS 26 | 29 159 | 1 137 ms | 3× |
| **wgpu / WGSL** | `gpu-wgpu` | M4 Pro · macOS 15 | 237 609 | 313 ms | 7× |
| **wgpu / WGSL** | `gpu-wgpu` | M1 Max · macOS 26 | 71 123 | 753 ms | 4× |
| CPU (Rayon) | _(none)_ `--no-gpu` | M4 Pro · macOS 15 | 33 921 | 512 ms | 1× |
| CPU (Rayon) | _(none)_ `--no-gpu` | M1 Max · macOS 26 | 9 390 | 524 ms | 1× |

**Key takeaways:**

- On macOS 15, Metal is the clear winner at **49× CPU** — its hand-written MSL kernel has no
  abstraction overhead.
- On macOS 26, wgpu beats native Metal (4× vs 3× CPU). The Metal backend compiles its MSL kernel
  from source at every startup; this cold-compile cost is significantly higher on macOS 26.
  wgpu's WGSL path is pre-cached by the Metal driver and avoids it.
- wgpu is consistently faster than CPU on both OS versions and is the safer default when macOS
  version is unknown.

---

## Criterion micro-benchmarks (M4 Pro · macOS 15)

Measured with `cargo bench`, release profile, 20 samples each.

| Benchmark | Mean | Notes |
|-----------|------|-------|
| `encode_hs256` | 655 ns | HS256 JWT sign |
| `encode_hs256_compressed` | 9.99 µs | + gzip payload |
| `decode` | 680 ns | base64 decode + parse |
| `verify_hs256` | 1.68 µs | full signature verify |
| `verify_hs256_fastpath` | 778 ns | fastpath (pre-parsed) |
| `crack_dict_8_words` | 6.20 µs | 8-word dictionary |
| `crack_brute_len3_lower` | 13.7 ms | len-3 lowercase, CPU Rayon |

---

## How to reproduce

```bash
# 1. Encode a token with a known short secret
./target/release/jwt-hack encode '{"sub":"bench"}' --secret=abc

# 2. Benchmark Metal GPU (default on macOS)
./target/release/jwt-hack crack -m brute <TOKEN> --max=3 --chars=abcdefghijklmnopqrstuvwxyz

# 3. Benchmark CPU only
./target/release/jwt-hack crack -m brute <TOKEN> --max=3 --chars=abcdefghijklmnopqrstuvwxyz --no-gpu

# 4. Benchmark wgpu backend (build separately)
cargo build --release --features gpu-wgpu
./target/release/jwt-hack crack -m brute <TOKEN> --max=3 --chars=abcdefghijklmnopqrstuvwxyz

# 5. Run Criterion suite
cargo bench --bench jwt_benchmarks
```

---

## Notes

- **Metal family gate:** The native Metal backend requires `MTLGPUFamily::Mac2` (discrete AMD on
  Intel Macs) or `Apple4+` (A11 / Apple Silicon). Intel integrated GPUs (`Mac1`) fall back to CPU
  automatically.
- **wgpu fallback:** If no compatible GPU adapter is found, wgpu transparently falls back to CPU;
  `is_available()` returns `false` and the crack path logs the reason.
- **Unified memory:** Apple Silicon's shared CPU/GPU memory means `StorageModeShared` buffers are
  zero-copy, avoiding PCIe transfer overhead present on Intel Macs with discrete GPUs.
