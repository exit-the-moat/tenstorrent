// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10 bb12: one codebook head's draw on one core (DRAMA3_SAMPLER_FUSED=1, tt/device_sampler.py), the
// math of the lean sampler (DRAMA3_SAMPLER_LEAN) in one program instead of ~20:
//   top-32 of the 1024 fp32 logits (sorted, ties: lower code first), m = the max;
//   E = exp(X - m) over all 1024 on the compute core (kernels/sampler_fused_compute.cpp: the SFPU sub + exp of
//   ttnn.subtract / ttnn.exp), z = sum(E) (exact 64-bit fixed-point sum, rounded once to fp32);
//   p_r = e_r / z and s_r = e_r / noise_r on the SFPU (div_binary_tile, the op of ttnn.divide; second compute pass),
//   cum_r = the Kahan-compensated running sum of p (the recurrence of ttnn.cumsum on fp32, in IEEE fp32 here);
//   keep_r = r == 0 || (r < top_k && cum_r <= top_p); the code of the first kept rank with the largest s_r.
// Outputs: code [1, 1, 1, 1] fp32 TILE (the code at (0, 0), zeros elsewhere) and the one-hot [1, 1, 1, N] bf16 TILE
// (1.0 at the code in row 0, zeros elsewhere). Inputs: logits [1, 1, 1, N] fp32 TILE, noise [1, 1, 1, 32] fp32 TILE
// (rank-indexed); only row 0 of each tile is read (64-byte chunks: row 0 of faces 0 and 1).
// Two RISCs (same source; SAMPLER_B = the BRISC half): A (NCRISC) reads the inputs, takes the top-32 of logits
// 0..511, sums E 0..511, merges and draws, writes the code and the one-hot tile; B (BRISC) writes the 32 zero
// one-hot tiles at once, then takes the top-32 of logits 512..1023 and sums E 512..1023. Semaphores: 0 = inputs in
// L1 (A -> B), 1 = B's top-32 ready, 2 = B's sum ready, 3 = the zero tiles written (B -> A).
// Runtime args: logits, noise, code, one-hot addresses. Compile-time: N, top_k, top_p bits, accessors.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_m = 1, cb_e = 2, cb_code = 3, cb_oh = 4, cb_scr = 5, cb_v = 6, cb_z = 7, cb_n = 8,
                   cb_p = 9, cb_s = 10;
constexpr uint32_t K = 32, HALF = 512;
// scratch (cb_scr) offsets: B's top-32 keys + codes, B's partial sum, B's stamps
constexpr uint32_t S_KEY = 256, S_IDX = 384, S_SUM = 512, S_TB = 640;

// element of rank r (0..31) in row 0 of a tile: faces 0 and 1
static inline uint32_t row0(uint32_t r) { return r < 16 ? r : 256 + (r - 16); }
#ifdef SAMPLER_PROFILE  // (checks) wall-clock stamps in row 0 of the code tile: A's at 1.., B's at 16..
#define TS(arr, i) arr[i] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L)
#else
#define TS(arr, i)
#endif

// fp32 math in IEEE order (the kernels build with -ffast-math): P, S = row 0 of the p / s tiles
__attribute__((optimize("no-fast-math"))) static uint32_t draw(
    const float* P, const float* S, const uint32_t* idx, uint32_t top_k, float top_p) {
    float acc = 0.0f, c = 0.0f, best = -1.0f;
    uint32_t code = idx[0];
    for (uint32_t r = 0; r < K; r++) {
        const float y = P[row0(r)] - c;  // Kahan, as ttnn.cumsum: y = p - c, t = acc + y, c = (t - acc) - y
        const float t = acc + y;
        c = (t - acc) - y;
        acc = t;
        if (r > 0 && !(r < top_k && t <= top_p)) {
            break;  // cum only grows: every later rank is dropped too (score 0)
        }
        const float sc = S[row0(r)];
        if (sc > best) {
            best = sc;
            code = idx[r];
        }
    }
    return code;
}

// fixed-point sum (2^-40 units, exact for values >= 2^-40; the rest add < 1024 * 2^-40 ~ 1e-9 to z >= 1) of fp32
// values in [0, 1]
static uint64_t fixed_sum(const uint32_t* v, uint32_t n) {
    uint64_t s = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t b = v[i];
        const uint32_t ex = b >> 23;  // (sign 0)
        const uint32_t mant = (b & 0x7FFFFFu) | 0x800000u;  // value = mant * 2^(ex - 150)
        if (ex >= 110) {
            s += static_cast<uint64_t>(mant) << (ex - 110);
        } else if (ex > 78) {
            s += mant >> (110 - ex);
        }
    }
    return s;
}

// the fixed-point sum rounded once to fp32 (round to nearest even)
static uint32_t fixed_to_f32(uint64_t s) {
    if (s == 0) {
        return 0;
    }
    int msb = 63;
    while (!((s >> msb) & 1)) {
        msb--;
    }
    uint32_t mant;
    int e2 = msb - 40;
    if (msb <= 23) {
        mant = static_cast<uint32_t>(s << (23 - msb));
    } else {
        const int sh = msb - 23;
        mant = static_cast<uint32_t>(s >> sh);
        const uint64_t rem = s & ((1ull << sh) - 1), half = 1ull << (sh - 1);
        if (rem > half || (rem == half && (mant & 1))) {
            mant++;
            if (mant == (1u << 24)) {
                mant >>= 1;
                e2++;
            }
        }
    }
    return (static_cast<uint32_t>(e2 + 127) << 23) | (mant & 0x7FFFFFu);
}

// "a is worse than b": lower key, or the same key and a higher code (ties keep the lower code)
static inline bool worse(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) { return ka < kb || (ka == kb && ia > ib); }

static void sift_down(uint32_t* hk, uint32_t* hi, uint32_t p) {
    while (true) {
        const uint32_t l = 2 * p + 1;
        if (l >= K) {
            return;
        }
        uint32_t c = l;
        if (l + 1 < K && worse(hk[l + 1], hi[l + 1], hk[l], hi[l])) {
            c = l + 1;
        }
        if (!worse(hk[c], hi[c], hk[p], hi[p])) {
            return;
        }
        const uint32_t tk = hk[p], ti = hi[p];
        hk[p] = hk[c];
        hi[p] = hi[c];
        hk[c] = tk;
        hi[c] = ti;
        p = c;
    }
}

// top-32 of x[i0 .. i0 + n) (n >= 32) by an order-preserving key (positive: set the sign bit, negative: invert),
// sorted descending, ties: the lower code first. A min-heap of the 32 best (root = the worst kept).
static void top32(const uint32_t* x, uint32_t i0, uint32_t n, uint32_t* key, uint32_t* idx) {
    uint32_t hk[K], hi[K];
    for (uint32_t j = 0; j < K; j++) {
        const uint32_t b = x[i0 + j];
        hk[j] = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        hi[j] = i0 + j;
    }
    for (int p = K / 2 - 1; p >= 0; p--) {
        sift_down(hk, hi, p);
    }
    uint32_t kth = hk[0];
    for (uint32_t i = i0 + K; i < i0 + n; i++) {
        const uint32_t b = x[i];
        const uint32_t k = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        if (k <= kth) {
            continue;  // (a later code with an equal key is worse)
        }
        hk[0] = k;
        hi[0] = i;
        sift_down(hk, hi, 0);
        kth = hk[0];
    }
    for (uint32_t j = 0; j < K; j++) {  // insertion sort, best first
        const uint32_t k = hk[j], c = hi[j];
        uint32_t q = j;
        while (q > 0 && worse(key[q - 1], idx[q - 1], k, c)) {
            key[q] = key[q - 1];
            idx[q] = idx[q - 1];
            q--;
        }
        key[q] = k;
        idx[q] = c;
    }
}

static inline volatile tt_l1_ptr uint32_t* sem(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t TOP_K = get_compile_time_arg_val(1);
    constexpr uint32_t TOP_P_BITS = get_compile_time_arg_val(2);
    constexpr auto lg_a = TensorAccessorArgs<3>();
    constexpr auto nz_a = TensorAccessorArgs<lg_a.next_compile_time_args_offset()>();
    constexpr auto code_a = TensorAccessorArgs<nz_a.next_compile_time_args_offset()>();
    constexpr auto oh_a = TensorAccessorArgs<code_a.next_compile_time_args_offset()>();
    constexpr uint32_t NT = N / 32;
    static_assert(N == 2 * HALF, "one fp32 tile holds the N logits");
    const uint32_t tb32 = 4096, tb16 = 2048;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);

    CircularBuffer cx(cb_x), cm(cb_m), ce(cb_e), cc(cb_code), coh(cb_oh), cs(cb_scr);
    // both RISCs see the same (initial) write pointers; only A pushes / pops
    const uint32_t x_l1 = cx.get_write_ptr(), m_l1 = cm.get_write_ptr(), code_l1 = cc.get_write_ptr();
    const uint32_t z_l1 = coh.get_write_ptr(), h_l1 = z_l1 + tb16, s_l1 = cs.get_write_ptr();
    const uint32_t* x = reinterpret_cast<const uint32_t*>(x_l1);

#ifdef SAMPLER_B
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* tb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1 + S_TB);
    TS(tb, 0);
    const auto oh = TensorAccessor(oh_a, get_arg_val<uint32_t>(3), tb16);
    for (uint32_t k = 0; k < tb16; k += 512) {
        noc_async_read(zeros, z_l1 + k, 512);
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < NT; t++) {
        noc_async_write(z_l1, oh.get_noc_addr(t), tb16);
    }
    noc_semaphore_wait(sem(0), 1);
    TS(tb, 1);
    top32(x, HALF, HALF, reinterpret_cast<uint32_t*>(s_l1 + S_KEY), reinterpret_cast<uint32_t*>(s_l1 + S_IDX));
    TS(tb, 2);
    asm volatile("" ::: "memory");  // the top-32 stores before the flag
    noc_semaphore_set(sem(1), 1);
    noc_async_write_barrier();
    TS(tb, 3);
    noc_semaphore_set(sem(3), 1);
    ce.wait_front(1);
    TS(tb, 4);
    const uint32_t* eb = reinterpret_cast<const uint32_t*>(ce.get_read_ptr());
    *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(s_l1 + S_SUM) = fixed_sum(eb + HALF, HALF);
    TS(tb, 5);
    noc_semaphore_set(sem(2), 1);
#else
    [[maybe_unused]] uint32_t ta[16];
    TS(ta, 0);
    const auto lg = TensorAccessor(lg_a, get_arg_val<uint32_t>(0), tb32);
    const auto nz = TensorAccessor(nz_a, get_arg_val<uint32_t>(1), tb32);
    const auto co = TensorAccessor(code_a, get_arg_val<uint32_t>(2), tb32);
    const auto oh = TensorAccessor(oh_a, get_arg_val<uint32_t>(3), tb16);
    cx.reserve_back(1);
    cm.reserve_back(1);
    // X in natural order: chunk (t, f) = logits 32 t + 16 f .. + 15 (row 0 of face f of tile t, 64 bytes)
    for (uint32_t t = 0; t < NT; t++) {
        noc_async_read(lg.get_noc_addr(t, 0), x_l1 + (2 * t) * 64, 64);
        noc_async_read(lg.get_noc_addr(t, 1024), x_l1 + (2 * t + 1) * 64, 64);
    }
    CircularBuffer cv(cb_v), czt(cb_z), cn(cb_n), cp(cb_p), csc(cb_s);
    cv.reserve_back(1);
    czt.reserve_back(1);
    cn.reserve_back(1);
    const uint32_t n_l1 = cn.get_write_ptr();
    noc_async_read(nz.get_noc_addr(0, 0), n_l1, 64);  // the noise row lands in row 0 of N
    noc_async_read(nz.get_noc_addr(0, 1024), n_l1 + 1024, 64);
    noc_async_read_barrier();
    TS(ta, 1);
    noc_semaphore_set(sem(0), 1);
    for (uint32_t k = 0; k < tb16; k += 512) {  // (lands while the top-32 runs)
        noc_async_read(zeros, h_l1 + k, 512);
        noc_async_read(zeros, code_l1 + 2 * k, 512);
        noc_async_read(zeros, code_l1 + 2 * k + 512, 512);
    }
    uint32_t ka[K], ia[K], idx[K];
    top32(x, 0, HALF, ka, ia);
    TS(ta, 2);
    noc_semaphore_wait(sem(1), 1);
    TS(ta, 3);
    const volatile tt_l1_ptr uint32_t* kb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1 + S_KEY);
    const volatile tt_l1_ptr uint32_t* ib = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1 + S_IDX);
    for (uint32_t j = 0, a = 0, b = 0; j < K; j++) {  // merge: on equal keys the lower codes (half A) first
        if (ka[a] >= kb[b]) {
            idx[j] = ia[a++];
        } else {
            idx[j] = ib[b++];
        }
    }
    const uint32_t mbits = x[idx[0]];
    volatile tt_l1_ptr uint32_t* mw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m_l1);
    for (uint32_t i = 0; i < N; i++) {
        mw[i] = mbits;
    }
    TS(ta, 4);
    cx.push_back(1);
    cm.push_back(1);

    ce.wait_front(1);  // E = exp(X - m), same element order as X
    TS(ta, 5);
    const uint32_t* eb = reinterpret_cast<const uint32_t*>(ce.get_read_ptr());
    const uint64_t sa = fixed_sum(eb, HALF);
    TS(ta, 6);
    noc_semaphore_wait(sem(2), 1);
    TS(ta, 7);
    const uint32_t zb = fixed_to_f32(sa + *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(s_l1 + S_SUM));
    volatile tt_l1_ptr uint32_t* vw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cv.get_write_ptr());
    volatile tt_l1_ptr uint32_t* zw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(czt.get_write_ptr());
    for (uint32_t r = 0; r < K; r++) {  // V: e_r in row 0, Z: z in row 0 (the other rows are not read back)
        vw[row0(r)] = eb[idx[r]];
        zw[row0(r)] = zb;
    }
    ce.pop_front(1);
    cv.push_back(1);
    czt.push_back(1);
    cn.push_back(1);
    float top_p;
    const uint32_t tp = TOP_P_BITS;
    __builtin_memcpy(&top_p, &tp, 4);
    cp.wait_front(1);
    csc.wait_front(1);
    const uint32_t code = draw(
        reinterpret_cast<const float*>(cp.get_read_ptr()), reinterpret_cast<const float*>(csc.get_read_ptr()), idx, TOP_K, top_p);
    cp.pop_front(1);
    csc.pop_front(1);
    TS(ta, 8);

    noc_async_read_barrier();  // the zero code / one-hot tiles
    const float cf = static_cast<float>(code);
    __builtin_memcpy(reinterpret_cast<void*>(code_l1), &cf, 4);
#ifdef SAMPLER_PROFILE
    {
        volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(code_l1);
        const volatile tt_l1_ptr uint32_t* tbv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1 + S_TB);
        for (uint32_t i = 1; i < 9; i++) {
            cw[i] = ta[i] - ta[0];
        }
        for (uint32_t i = 0; i < 6; i++) {
            cw[16 * 16 + i] = tbv[i] - ta[0];  // face 1 row 0
        }
    }
#endif
    const uint32_t c32 = code % 32;
    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(h_l1)[(c32 / 16) * 256 + (c32 % 16)] = 0x3F80;  // bf16 1.0
    noc_async_write(code_l1, co.get_noc_addr(0), tb32);
    noc_semaphore_wait(sem(3), 1);  // B's zero tiles have landed: the one-hot tile goes over its page
    noc_async_write(h_l1, oh.get_noc_addr(code / 32), tb16);
    noc_async_write_barrier();
#endif
}
