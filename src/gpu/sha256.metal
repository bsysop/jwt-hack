#include <metal_stdlib>
using namespace metal;

// --- SHA-256 constants and helpers -------------------------------------------

constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
inline uint ch(uint x, uint y, uint z)   { return (x & y) ^ (~x & z); }
inline uint maj(uint x, uint y, uint z)  { return (x & y) ^ (x & z) ^ (y & z); }
inline uint Sig0(uint x)  { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
inline uint Sig1(uint x)  { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
inline uint sig0(uint x)  { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
inline uint sig1(uint x)  { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

// --- SHA-256 transform (64 rounds) -------------------------------------------

void sha256_transform(thread uint state[8], const thread uchar* block) {
    uint W[64];
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint)block[i*4]   << 24) |
               ((uint)block[i*4+1] << 16) |
               ((uint)block[i*4+2] <<  8) |
               ((uint)block[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        W[i] = sig1(W[i-2]) + W[i-7] + sig0(W[i-15]) + W[i-16];
    }
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint T1 = h + Sig1(e) + ch(e,f,g) + K[i] + W[i];
        uint T2 = Sig0(a) + maj(a,b,c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// --- SHA-256 over a single 512-bit block (no padding needed) -----------------

void sha256_single_block(thread uint digest[8], const thread uchar block[64]) {
    // Initial SHA-256 state
    digest[0] = 0x6a09e667; digest[1] = 0xbb67ae85;
    digest[2] = 0x3c6ef372; digest[3] = 0xa54ff53a;
    digest[4] = 0x510e527f; digest[5] = 0x9b05688c;
    digest[6] = 0x1f83d9ab; digest[7] = 0x5be0cd19;
    sha256_transform(digest, block);
}

// --- HMAC-SHA256 verify kernel -----------------------------------------------

/// Each thread tests one candidate secret against the fixed signing_input.
///
/// Buffers:
///   0  signing_input  (constant)  — the JWT header.payload bytes
///   1  si_len          (constant)  — [n] = signing_input length in bytes
///   2  expected_sig    (constant)  — [32] = target HMAC output
///   3  candidates      (constant)  — packed candidate secret bytes
///   4  offsets         (constant)  — [tid] = byte offset, [tid+1] = end offset
///   5  results         (device)    — output: 1 = match, 0 = no match
kernel void hmac_sha256_verify(
    const device uchar*  signing_input [[buffer(0)]],
    const device uint*   si_len        [[buffer(1)]],
    const device uchar*  expected_sig  [[buffer(2)]],
    const device uchar*  candidates    [[buffer(3)]],
    const device uint*   offsets       [[buffer(4)]],
    device   uint*       results       [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    // Thread 0 never processes a candidate (reserved - no candidate 0)
    if (tid == 0) return;

    // Skip if already found in a prior dispatch
    if (results[tid] != 0) return;

    // Candidate (tid-1) occupies offsets[tid-1]..offsets[tid]
    uint start = offsets[tid - 1];
    uint end   = offsets[tid];
    if (end <= start) return;
    uint secret_len = end - start;

    // --- Load secret bytes ---
    uchar secret[64] = {0};
    for (uint i = 0; i < secret_len && i < 64; i++) {
        secret[i] = candidates[start + i];
    }

    // --- HMAC-SHA256: inner hash H((K ^ ipad) || message) ---
    // Build padded key block: K' = secret padded to 64 bytes with zeros
    uchar k_padded[64] = {0};
    for (uint i = 0; i < secret_len && i < 64; i++) {
        k_padded[i] = secret[i];
    }

    // inner key block: K' ^ 0x36 (ipad)
    uchar inner_block[64];
    for (uint i = 0; i < 64; i++) {
        inner_block[i] = k_padded[i] ^ 0x36;
    }

    // Compute SHA256(inner_block || signing_input)
    // We need to hash (64 + si_len[0]) bytes with SHA256 padding.
    // Approach: do a two-step transform.
    // Step 1: hash block 0 (inner_block) — this is exactly 512 bits
    uint state[8];
    sha256_single_block(state, inner_block);

    // Step 2: continue hashing signing_input + SHA256 padding
    uint si_len_val = si_len[0];
    uint total_remaining = si_len_val + 65; // 1 byte for 0x80 marker + 8 bytes for length

    // Allocate buffer for remaining data + padding
    // Max needed: signing_input (up to ~16KB) + 9 bytes padding
    // We process in 64-byte blocks
    uchar buf[64];
    uint buf_pos = 0;

    // Feed signing_input
    for (uint i = 0; i < si_len_val; i++) {
        buf[buf_pos++] = signing_input[i];
        if (buf_pos == 64) {
            sha256_transform(state, buf);
            buf_pos = 0;
        }
    }

    // SHA256 padding: append 0x80, then zeros, then 64-bit big-endian
    // length in bits.  For our message sizes the high 32 bits are always 0.
    ulong total_bits = (64UL + (ulong)si_len_val) * 8UL;
    buf[buf_pos++] = 0x80;
    if (buf_pos > 56) {
        for (uint i = buf_pos; i < 64; i++) buf[i] = 0;
        sha256_transform(state, buf);
        buf_pos = 0;
    }
    for (uint i = buf_pos; i < 64; i++) buf[i] = 0;
    // low 32 bits of bit-length (big-endian); high 32 bits are 0
    uint low32 = (uint)(total_bits & 0xFFFFFFFFUL);
    buf[60] = (low32 >> 24) & 0xFF;
    buf[61] = (low32 >> 16) & 0xFF;
    buf[62] = (low32 >> 8) & 0xFF;
    buf[63] = low32 & 0xFF;
    sha256_transform(state, buf);

    // Extract inner digest
    uchar inner_digest[32];
    for (int i = 0; i < 8; i++) {
        inner_digest[i*4]   = (state[i] >> 24) & 0xFF;
        inner_digest[i*4+1] = (state[i] >> 16) & 0xFF;
        inner_digest[i*4+2] = (state[i] >> 8) & 0xFF;
        inner_digest[i*4+3] = state[i] & 0xFF;
    }

    // --- HMAC-SHA256: outer hash H((K ^ opad) || inner_digest) ---
    // outer key block: K' ^ 0x5c (opad)
    uchar outer_block[64];
    for (uint i = 0; i < 64; i++) {
        outer_block[i] = k_padded[i] ^ 0x5c;
    }

    // Hash outer_block (64 bytes = exactly 512 bits)
    sha256_single_block(state, outer_block);

    // Feed inner_digest (32 bytes) + SHA256 padding.
    // Total = 64 + 32 = 96 bytes → 96*8 = 768 bits = 0x300.
    uchar outer_buf[64] = {0};
    for (int i = 0; i < 32; i++) outer_buf[i] = inner_digest[i];
    outer_buf[32] = 0x80;
    // 64-bit big-endian length: high 32 bits = 0, low 32 bits = 768
    outer_buf[62] = 0x03;
    outer_buf[63] = 0x00;
    sha256_transform(state, outer_buf);

    // Extract outer digest
    uchar outer_digest[32];
    for (int i = 0; i < 8; i++) {
        outer_digest[i*4]   = (state[i] >> 24) & 0xFF;
        outer_digest[i*4+1] = (state[i] >> 16) & 0xFF;
        outer_digest[i*4+2] = (state[i] >> 8) & 0xFF;
        outer_digest[i*4+3] = state[i] & 0xFF;
    }

    // --- Compare against expected signature ---
    for (int i = 0; i < 32; i++) {
        if (outer_digest[i] != expected_sig[i]) {
            return; // mismatch
        }
    }
    results[tid] = 1; // match!
}
