// Cross-vendor HMAC-SHA256 verify kernel (WGSL).
//
// Port of `sha256.metal` to WGSL so the same brute-force verification runs on
// any wgpu backend (Vulkan/NVIDIA·AMD·Intel, DX12, Metal, GL). WGSL has no
// byte type, so everything is expressed in 32-bit words:
//
//   * `signing_input` / `candidates` are packed byte buffers, read one byte at
//     a time with little-endian extraction (`byte i == (word[i/4] >> (i%4*8))`),
//     matching the little-endian packing the host uploads.
//   * SHA-256 message blocks are assembled as 16 big-endian words (the natural
//     form the transform consumes), so the key and the outer block are built
//     directly as words and only the variable-length `signing_input` middle
//     section needs per-byte assembly.
//   * `expected_sig` is uploaded as 8 big-endian words so the final state can be
//     compared word-for-word without re-serialising the digest to bytes.
//
// Thread model mirrors Metal: `tid` is 1-based, candidate `tid-1` occupies
// `candidates[offsets[tid-1]..offsets[tid]]`, and `tid == 0` is reserved.

struct Params {
    si_len: u32,       // signing_input length in bytes
    num_threads: u32,  // offsets length == batch_size + 1
};

@group(0) @binding(0) var<storage, read>       signing_input: array<u32>;
@group(0) @binding(1) var<uniform>             params: Params;
@group(0) @binding(2) var<storage, read>       expected_sig: array<u32>; // 8 BE words
@group(0) @binding(3) var<storage, read>       candidates: array<u32>;
@group(0) @binding(4) var<storage, read>       offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> results: array<u32>;

const K = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn ch(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ (~x & z); }
fn maj(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ (x & z) ^ (y & z); }
fn big_sig0(x: u32) -> u32 { return rotr(x, 2u) ^ rotr(x, 13u) ^ rotr(x, 22u); }
fn big_sig1(x: u32) -> u32 { return rotr(x, 6u) ^ rotr(x, 11u) ^ rotr(x, 25u); }
fn small_sig0(x: u32) -> u32 { return rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u); }
fn small_sig1(x: u32) -> u32 { return rotr(x, 17u) ^ rotr(x, 19u) ^ (x >> 10u); }

// Compress one 512-bit block (16 big-endian words) into `state`.
fn sha256_transform(state: ptr<function, array<u32, 8>>, m: ptr<function, array<u32, 16>>) {
    var w: array<u32, 64>;
    for (var i = 0u; i < 16u; i = i + 1u) { w[i] = (*m)[i]; }
    for (var i = 16u; i < 64u; i = i + 1u) {
        w[i] = small_sig1(w[i - 2u]) + w[i - 7u] + small_sig0(w[i - 15u]) + w[i - 16u];
    }
    var a = (*state)[0]; var b = (*state)[1]; var c = (*state)[2]; var d = (*state)[3];
    var e = (*state)[4]; var f = (*state)[5]; var g = (*state)[6]; var h = (*state)[7];
    for (var i = 0u; i < 64u; i = i + 1u) {
        let t1 = h + big_sig1(e) + ch(e, f, g) + K[i] + w[i];
        let t2 = big_sig0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    (*state)[0] = (*state)[0] + a; (*state)[1] = (*state)[1] + b;
    (*state)[2] = (*state)[2] + c; (*state)[3] = (*state)[3] + d;
    (*state)[4] = (*state)[4] + e; (*state)[5] = (*state)[5] + f;
    (*state)[6] = (*state)[6] + g; (*state)[7] = (*state)[7] + h;
}

fn init_state(state: ptr<function, array<u32, 8>>) {
    (*state)[0] = 0x6a09e667u; (*state)[1] = 0xbb67ae85u;
    (*state)[2] = 0x3c6ef372u; (*state)[3] = 0xa54ff53au;
    (*state)[4] = 0x510e527fu; (*state)[5] = 0x9b05688cu;
    (*state)[6] = 0x1f83d9abu; (*state)[7] = 0x5be0cd19u;
}

// Candidate / signing-input bytes are little-endian packed into words.
fn cand_byte(i: u32) -> u32 { return (candidates[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu; }
fn si_byte(i: u32) -> u32 { return (signing_input[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu; }

// Set byte `pos` (0..63) of a big-endian word block, leaving the rest intact.
fn set_byte_be(block: ptr<function, array<u32, 16>>, pos: u32, val: u32) {
    let w = pos >> 2u;
    let shift = (3u - (pos & 3u)) * 8u;
    (*block)[w] = ((*block)[w] & ~(0xffu << shift)) | ((val & 0xffu) << shift);
}

@compute @workgroup_size(64)
fn hmac_sha256_verify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid == 0u || tid >= params.num_threads) { return; }
    if (results[tid] != 0u) { return; }

    let start = offsets[tid - 1u];
    let end = offsets[tid];
    if (end <= start) { return; }
    let secret_len = end - start;

    // Keys longer than one block would require SHA256(key) first; the host gate
    // excludes them and they are infeasible to enumerate — skip (as Metal does).
    if (secret_len > 64u) { return; }

    // Padded key block K' (64 bytes) as 16 big-endian words.
    var kw: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { kw[i] = 0u; }
    for (var i = 0u; i < secret_len; i = i + 1u) {
        set_byte_be(&kw, i, cand_byte(start + i));
    }

    // --- Inner hash: SHA256((K' ^ ipad) || signing_input) ---
    var inner_block: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { inner_block[i] = kw[i] ^ 0x36363636u; }

    var state: array<u32, 8>;
    init_state(&state);
    sha256_transform(&state, &inner_block);

    // Feed signing_input, then SHA-256 padding, in 64-byte blocks.
    let si_len = params.si_len;
    var block: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { block[i] = 0u; }
    var bpos = 0u;
    for (var i = 0u; i < si_len; i = i + 1u) {
        set_byte_be(&block, bpos, si_byte(i));
        bpos = bpos + 1u;
        if (bpos == 64u) {
            sha256_transform(&state, &block);
            for (var j = 0u; j < 16u; j = j + 1u) { block[j] = 0u; }
            bpos = 0u;
        }
    }

    // Append 0x80, then (possibly) flush, then the 64-bit big-endian bit length.
    // Total message length for the inner hash is (64 + si_len) bytes.
    let total_bits = (64u + si_len) * 8u;
    set_byte_be(&block, bpos, 0x80u);
    bpos = bpos + 1u;
    if (bpos > 56u) {
        sha256_transform(&state, &block);
        for (var j = 0u; j < 16u; j = j + 1u) { block[j] = 0u; }
        bpos = 0u;
    }
    // High 32 bits of the length are always 0 for our message sizes.
    block[14] = 0u;
    block[15] = total_bits;
    sha256_transform(&state, &block);

    // Inner digest words (big-endian word form == the 32 digest bytes).
    var inner_digest: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) { inner_digest[i] = state[i]; }

    // --- Outer hash: SHA256((K' ^ opad) || inner_digest) ---
    var outer_block: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { outer_block[i] = kw[i] ^ 0x5c5c5c5cu; }
    init_state(&state);
    sha256_transform(&state, &outer_block);

    // Second outer block: 32-byte inner digest + padding. Total = 96 bytes =
    // 768 bits, so 0x80 lands at byte 32 (word 8 MSB) and the length word is 768.
    var ob: array<u32, 16>;
    for (var i = 0u; i < 8u; i = i + 1u) { ob[i] = inner_digest[i]; }
    ob[8] = 0x80000000u;
    for (var i = 9u; i < 15u; i = i + 1u) { ob[i] = 0u; }
    ob[14] = 0u;
    ob[15] = 768u;
    sha256_transform(&state, &ob);

    // Compare the final state against the expected signature (8 BE words).
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (state[i] != expected_sig[i]) { return; }
    }
    results[tid] = 1u;
}
