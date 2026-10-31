// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// drama3 port10 bb12: the fused codebook draw (DRAMA3_SAMPLER_FUSED=1) over NC cores (DRAMA3_SAMPLER_FUSED_CORES=NC;
// the one-core version and the math: kernels/sampler_fused_reader.cpp). Core j owns logits [j W, (j + 1) W), W = N / NC:
//   A (NCRISC) reads them, takes the top-32 of the first half, merges with B's, E_j = exp(X_j - m_j) on its compute
//   core (m_j = its max), sums E_j's first half; B (BRISC) zeroes the one-hot tiles of its codes, takes the top-32 and
//   the E_j sum of the second half. A sends (its 32 keys + codes, m_j, S_j = sum E_j as 2^-40 fixed point) to slot j
//   of the central core (core 0) and bumps its semaphore 4; B bumps semaphore 5 once its zero tiles have landed.
//   Central A: merges the NC lists (equal keys: the lower code), m = the max; its compute core runs
//   G = exp([top-32 logits | m_j] - m) (e_r in row 0 = exp(x_r - m) exactly as the lean sampler; g_j = exp(m_j - m)),
//   z = sum_j g_j * S_j (fp32), then P = e / z and S = e / noise (div_binary_tile); the Kahan cum, keep and argmax as
//   the one-core kernel; writes the code tile and, after every core's zero tiles, the one-hot tile.
// Runtime args: logits, noise, code, one-hot addresses, j, central NOC x, y.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_x = 0, cb_m = 1, cb_e = 2, cb_code = 3, cb_oh = 4, cb_scr = 5, cb_v = 6, cb_z = 7, cb_n = 8,
                   cb_p = 9, cb_s = 10, cb_g = 11, cb_mg = 12;
constexpr uint32_t K = 32;
// scratch: B's top-32 (keys, codes), B's partial sum, then the central slots (one per core)
constexpr uint32_t S_KEY = 0, S_IDX = 128, S_SUM = 256, S_SLOT = 512, SLOT = 512;  // slot: keys 0, codes 128, S 256, m 264

static inline uint32_t row0(uint32_t r) { return r < 16 ? r : 256 + (r - 16); }

__attribute__((optimize("no-fast-math"))) static uint32_t draw(
    const float* P, const float* S, const uint32_t* idx, uint32_t top_k, float top_p) {
    float acc = 0.0f, c = 0.0f, best = -1.0f;
    uint32_t code = idx[0];
    for (uint32_t r = 0; r < K; r++) {
        const float y = P[row0(r)] - c;  // Kahan, as ttnn.cumsum
        const float t = acc + y;
        c = (t - acc) - y;
        acc = t;
        if (r > 0 && !(r < top_k && t <= top_p)) {
            break;
        }
        const float sc = S[row0(r)];
        if (sc > best) {
            best = sc;
            code = idx[r];
        }
    }
    return code;
}

// z = sum_j g_j * S_j in fp32, j in order (S_j: 2^-40 fixed point, converted with one rounding)
__attribute__((optimize("no-fast-math"))) static float zsum(const float* g, const uint64_t* s, uint32_t n) {
    float z = 0.0f;
    for (uint32_t j = 0; j < n; j++) {
        z = z + g[j] * (static_cast<float>(s[j]) * 9.094947017729282e-13f);  // 2^-40
    }
    return z;
}

static uint64_t fixed_sum(const uint32_t* v, uint32_t n) {
    uint64_t s = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t b = v[i];
        const uint32_t ex = b >> 23;
        const uint32_t mant = (b & 0x7FFFFFu) | 0x800000u;
        if (ex >= 110) {
            s += static_cast<uint64_t>(mant) << (ex - 110);
        } else if (ex > 78) {
            s += mant >> (110 - ex);
        }
    }
    return s;
}

static inline uint32_t key_of(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
static inline uint32_t bits_of(uint32_t k) { return (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; }
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

// top-32 (sorted, best first, ties: lower code) of x[i0 .. i0 + n), n >= 32; codes = base + position
static void top32(const uint32_t* x, uint32_t i0, uint32_t n, uint32_t base, uint32_t* key, uint32_t* idx) {
    uint32_t hk[K], hi[K];
    for (uint32_t j = 0; j < K; j++) {
        hk[j] = key_of(x[i0 + j]);
        hi[j] = base + i0 + j;
    }
    for (int p = K / 2 - 1; p >= 0; p--) {
        sift_down(hk, hi, p);
    }
    uint32_t kth = hk[0];
    for (uint32_t i = i0 + K; i < i0 + n; i++) {
        const uint32_t k = key_of(x[i]);
        if (k <= kth) {
            continue;
        }
        hk[0] = k;
        hi[0] = base + i;
        sift_down(hk, hi, 0);
        kth = hk[0];
    }
    for (uint32_t j = 0; j < K; j++) {
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
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr auto lg_a = TensorAccessorArgs<4>();
    constexpr auto nz_a = TensorAccessorArgs<lg_a.next_compile_time_args_offset()>();
    constexpr auto code_a = TensorAccessorArgs<nz_a.next_compile_time_args_offset()>();
    constexpr auto oh_a = TensorAccessorArgs<code_a.next_compile_time_args_offset()>();
    constexpr uint32_t W = N / NC, H = W / 2, TW = W / 32;
    static_assert(N == 1024 && N % NC == 0 && H >= K && TW >= 1, "N = 1024 over NC cores");
    const uint32_t tb32 = 4096, tb16 = 2048;
    const uint32_t j = get_arg_val<uint32_t>(4);
    const uint32_t cnx = get_arg_val<uint32_t>(5), cny = get_arg_val<uint32_t>(6);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);

    CircularBuffer cx(cb_x), cm(cb_m), ce(cb_e), cc(cb_code), coh(cb_oh), cs(cb_scr);
    const uint32_t x_l1 = cx.get_write_ptr(), m_l1 = cm.get_write_ptr(), code_l1 = cc.get_write_ptr();
    const uint32_t z_l1 = coh.get_write_ptr(), h_l1 = z_l1 + tb16, s_l1 = cs.get_write_ptr();
    const uint32_t* x = reinterpret_cast<const uint32_t*>(x_l1);

#ifdef SAMPLER_B
    const auto oh = TensorAccessor(oh_a, get_arg_val<uint32_t>(3), tb16);
    for (uint32_t k = 0; k < tb16; k += 512) {
        noc_async_read(zeros, z_l1 + k, 512);
    }
    noc_async_read_barrier();
    for (uint32_t t = j * TW; t < (j + 1) * TW; t++) {
        noc_async_write(z_l1, oh.get_noc_addr(t), tb16);
    }
    noc_semaphore_wait(sem(0), 1);
    top32(x, H, H, j * W, reinterpret_cast<uint32_t*>(s_l1 + S_KEY), reinterpret_cast<uint32_t*>(s_l1 + S_IDX));
    asm volatile("" ::: "memory");
    noc_semaphore_set(sem(1), 1);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(cnx, cny, get_semaphore(5)), 1);  // this core's zero tiles have landed
    ce.wait_front(1);
    *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(s_l1 + S_SUM) =
        fixed_sum(reinterpret_cast<const uint32_t*>(ce.get_read_ptr()) + H, H);
    noc_semaphore_set(sem(2), 1);
    noc_async_atomic_barrier();
#else
    const auto lg = TensorAccessor(lg_a, get_arg_val<uint32_t>(0), tb32);
    cx.reserve_back(1);
    cm.reserve_back(1);
    for (uint32_t t = 0; t < TW; t++) {
        noc_async_read(lg.get_noc_addr(j * TW + t, 0), x_l1 + (2 * t) * 64, 64);
        noc_async_read(lg.get_noc_addr(j * TW + t, 1024), x_l1 + (2 * t + 1) * 64, 64);
    }
    CircularBuffer cv(cb_v), czt(cb_z), cn(cb_n), cp(cb_p), csc(cb_s), cg(cb_g), cmg(cb_mg);
    if (j == 0) {
        const auto nz = TensorAccessor(nz_a, get_arg_val<uint32_t>(1), tb32);
        cn.reserve_back(1);
        noc_async_read(nz.get_noc_addr(0, 0), cn.get_write_ptr(), 64);
        noc_async_read(nz.get_noc_addr(0, 1024), cn.get_write_ptr() + 1024, 64);
    }
    noc_async_read_barrier();
    noc_semaphore_set(sem(0), 1);
    if (j == 0) {
        for (uint32_t k = 0; k < tb16; k += 512) {  // (lands while the top-32 runs)
            noc_async_read(zeros, h_l1 + k, 512);
            noc_async_read(zeros, code_l1 + 2 * k, 512);
            noc_async_read(zeros, code_l1 + 2 * k + 512, 512);
        }
    }
    uint32_t ka[K], ia[K];
    top32(x, 0, H, j * W, ka, ia);
    noc_semaphore_wait(sem(1), 1);
    const uint32_t slot = s_l1 + S_SLOT + j * SLOT;  // (the same address on every core of the program)
    volatile tt_l1_ptr uint32_t* sk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
    {
        const volatile tt_l1_ptr uint32_t* kb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1 + S_KEY);
        const volatile tt_l1_ptr uint32_t* ib = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1 + S_IDX);
        for (uint32_t r = 0, a = 0, b = 0; r < K; r++) {  // merge the halves into this core's own slot
            if (ka[a] >= kb[b]) {
                sk[r] = ka[a];
                sk[32 + r] = ia[a++];
            } else {
                sk[r] = kb[b];
                sk[32 + r] = ib[b++];
            }
        }
    }
    const uint32_t mj = bits_of(sk[0]);
    sk[66] = mj;
    volatile tt_l1_ptr uint32_t* mw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m_l1);
    for (uint32_t i = 0; i < W; i++) {
        mw[i] = mj;
    }
    cx.push_back(1);
    cm.push_back(1);
    ce.wait_front(1);
    const uint64_t sa = fixed_sum(reinterpret_cast<const uint32_t*>(ce.get_read_ptr()), H);
    noc_semaphore_wait(sem(2), 1);
    *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(slot + 256) = sa + *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(s_l1 + S_SUM);
    ce.pop_front(1);
    if (j != 0) {
        noc_async_write(slot, get_noc_addr(cnx, cny, slot), 272);
        noc_async_write_barrier();
    }
    noc_semaphore_inc(get_noc_addr(cnx, cny, get_semaphore(4)), 1);
    noc_async_atomic_barrier();
    if (j != 0) {
        return;
    }

    // ---- central
    noc_semaphore_wait(sem(4), NC);
    uint32_t idx[K], vb[K];
    {
        uint32_t pos[NC];
        for (uint32_t c = 0; c < NC; c++) {
            pos[c] = 0;
        }
        const volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1 + S_SLOT);
        for (uint32_t r = 0; r < K; r++) {  // NC-way merge; equal keys: the lower core (= the lower codes) first
            uint32_t bc = 0, bk = 0;
            bool any = false;
            for (uint32_t c = 0; c < NC; c++) {
                if (pos[c] < K) {
                    const uint32_t k = sl[c * (SLOT / 4) + pos[c]];
                    if (!any || k > bk) {
                        bk = k;
                        bc = c;
                        any = true;
                    }
                }
            }
            idx[r] = sl[bc * (SLOT / 4) + 32 + pos[bc]];
            vb[r] = bits_of(bk);
            pos[bc]++;
        }
    }
    // G = [top-32 logits in row 0 | m_j at row 1 of face 0], MG = m there: exp(G - MG) on the compute core
    cg.reserve_back(1);
    cmg.reserve_back(1);
    volatile tt_l1_ptr uint32_t* gw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cg.get_write_ptr());
    volatile tt_l1_ptr uint32_t* gm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cmg.get_write_ptr());
    const uint32_t mbits = vb[0];
    const volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1 + S_SLOT);
    for (uint32_t r = 0; r < K; r++) {
        gw[row0(r)] = vb[r];
        gm[row0(r)] = mbits;
    }
    for (uint32_t c = 0; c < NC; c++) {
        gw[16 + c] = sl[c * (SLOT / 4) + 66];
        gm[16 + c] = mbits;
    }
    cg.push_back(1);
    cmg.push_back(1);
    cv.wait_front(1);  // compute: V = exp(G - MG)
    const float* vf = reinterpret_cast<const float*>(cv.get_read_ptr());
    float g[NC];
    uint64_t sj[NC];
    for (uint32_t c = 0; c < NC; c++) {
        g[c] = vf[16 + c];
        sj[c] = *reinterpret_cast<const volatile tt_l1_ptr uint64_t*>(s_l1 + S_SLOT + c * SLOT + 256);
    }
    const float z = zsum(g, sj, NC);
    czt.reserve_back(1);
    volatile tt_l1_ptr float* zw = reinterpret_cast<volatile tt_l1_ptr float*>(czt.get_write_ptr());
    for (uint32_t r = 0; r < K; r++) {
        zw[row0(r)] = z;
    }
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
    csc.pop_front(1);  // (V is popped by its other consumer, the compute core's unpacker)

    const auto co = TensorAccessor(code_a, get_arg_val<uint32_t>(2), tb32);
    const auto oh = TensorAccessor(oh_a, get_arg_val<uint32_t>(3), tb16);
    noc_async_read_barrier();
    const float cf = static_cast<float>(code);
    __builtin_memcpy(reinterpret_cast<void*>(code_l1), &cf, 4);
    const uint32_t c32 = code % 32;
    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(h_l1)[(c32 / 16) * 256 + (c32 % 16)] = 0x3F80;  // bf16 1.0
    noc_async_write(code_l1, co.get_noc_addr(0), tb32);
    noc_semaphore_wait(sem(5), NC);  // every core's zero one-hot tiles have landed
    noc_async_write(h_l1, oh.get_noc_addr(code / 32), tb16);
    noc_async_write_barrier();
#endif
}
