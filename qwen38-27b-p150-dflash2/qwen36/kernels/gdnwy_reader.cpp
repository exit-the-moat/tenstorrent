// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ms4 (Exit the Moat, Episode 2): the DFlash2 verify GDN chain in block (WY) form, reader. One program per GDN layer
// for every live stream (core group per stream, CS cores per value head, rows as runtime args), the gdnq4c GDNQ_PRO
// prologue unchanged (conv of all NT slots as rows, q / k l2 norms), then instead of NT chained slots:
//   S0 = gamma_a * base + K_prev^T diag(w) U_prev   (the state at the previous step accepted slot a, recomputed from
//        the previous block; a >= NT: S0 = base), written back to base by the writer,
//   the block outputs from S0 in closed form (see gdnwy_compute.cpp), and this block K / U / R / gamma stored to blk
//   for the next step. The NT slot states are never written (no bufs).
// GDNWY_COMMIT: only S0 -> base (a commit before a non-WY step / the plain lane), nothing else.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#define XOFF(f, r) (((xrow0_ + (r)) / 16) * 1024 + (f) * 512 + ((xrow0_ + (r)) % 16) * 32)

constexpr uint32_t cb_z = 3, cb_sdst = 8, cb_onw = 9;
constexpr uint32_t cb_kp = 50, cb_up = 51, cb_wr = 52, cb_gu = 53, cb_cel = 62;

void kernel_main() {
    constexpr uint32_t NK = get_compile_time_arg_val(0);
    constexpr uint32_t NV = get_compile_time_arg_val(1);
    constexpr uint32_t ZT0 = get_compile_time_arg_val(2);
    constexpr uint32_t CS = get_compile_time_arg_val(3);
    constexpr uint32_t NC = 4 / CS;
    constexpr uint32_t RF = NV / NK;
    constexpr uint32_t NT = GDNQ_CHAIN;
    static_assert(NT <= 16, "WY block: the slots of a stream in one 16-row tile face");
    constexpr auto xa = TensorAccessorArgs<4>();
    constexpr auto xla = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto wpa = TensorAccessorArgs<xla.next_compile_time_args_offset()>();
    constexpr auto basea = TensorAccessorArgs<wpa.next_compile_time_args_offset()>();
    constexpr auto onwa = TensorAccessorArgs<basea.next_compile_time_args_offset()>();
    constexpr auto kva = TensorAccessorArgs<onwa.next_compile_time_args_offset()>();
    constexpr auto blka = TensorAccessorArgs<kva.next_compile_time_args_offset()>();
    constexpr uint32_t KT0 = 4 * NK, VT0 = 8 * NK;

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    const uint32_t ci = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t xl_addr = get_arg_val<uint32_t>(i++);
    const uint32_t wp_addr = get_arg_val<uint32_t>(i++);
    const uint32_t base_addr = get_arg_val<uint32_t>(i++);
    const uint32_t onw_addr = get_arg_val<uint32_t>(i++);
    const uint32_t kv_addr = get_arg_val<uint32_t>(i++);
    const uint32_t blk_addr = get_arg_val<uint32_t>(i++);
    const uint32_t xrow0_ = get_arg_val<uint32_t>(i++);
    (void)x_addr;
    (void)xl_addr;
    (void)wp_addr;
    (void)onw_addr;
    (void)xrow0_;

    const uint32_t hk = hv / RF;
    const uint32_t c0 = ci * NC;
    constexpr uint32_t tb = 2048, tf = 4096;
    const auto base = TensorAccessor(basea, base_addr, tf);
    const auto blk = TensorAccessor(blka, blk_addr, tf);
    Noc noc;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    auto zfill = [&](uint32_t addr, uint32_t bytes) {  // MEM_ZEROS is 512 bytes
        for (uint32_t o = 0; o < bytes; o += 512) {
            noc_async_read(zeros, addr + o, 512);
        }
    };
    auto f32 = [](uint32_t addr) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); };
    // element (r, c) of a fp32 tile: face (r / 16) * 2 + (c / 16), row r % 16, column c % 16
    auto eidx = [](uint32_t r, uint32_t c) -> uint32_t {
        return ((r / 16) * 2 + (c / 16)) * 256 + (r % 16) * 16 + (c % 16);
    };
    constexpr uint32_t ONEF = 0x3f800000u;

    // ---- the previous step accepted slot a (element NT * NT of the `a` input); a >= NT: start from base
    CircularBuffer ck(cb_kp), cu(cb_up), cw(cb_wr), cg(cb_gu);
    ck.reserve_back(4);
    cu.reserve_back(NC);
    cw.reserve_back(1);
    cg.reserve_back(1);
    const uint32_t pk = ck.get_write_ptr(), pu = cu.get_write_ptr(), pw = cw.get_write_ptr(), pg = cg.get_write_ptr();
    uint32_t a = 255;
    {
        const auto kvt = TensorAccessor(kva, kv_addr, (NT * NT + 16) * 4);
        noc_async_read(kvt.get_noc_addr(0, NT * NT * 4), pg, 64);
        noc_async_read_barrier();
        a = f32(pg)[0];
    }
    const bool rec = a < NT;

    // ---- base state (own columns) -> cb_sdst (unpack to dest: exact fp32 copies in the compute)
    {
        CircularBuffer cs(cb_sdst);
        cs.reserve_back(4 * NC);
        const uint32_t pd = cs.get_write_ptr();
        for (uint32_t ki = 0; ki < 4; ki++) {
            for (uint32_t j = 0; j < NC; j++) {
                noc_async_read(base.get_noc_addr(hv * 16 + ki * 4 + c0 + j), pd + (ki * NC + j) * tf, tf);
            }
        }
        noc.async_read_barrier();
        cs.push_back(4 * NC);
    }
    // ---- the previous block: K (4 tiles), U (own columns), and from R / gamma: the row-scale tile W (row j = w_j =
    // R[a, j], j <= a, on every column) and the uniform gamma_a tile. a >= NT: K = U = W = 0, gamma = 1 (S0 = base)
    if (rec) {
        for (uint32_t t = 0; t < 4; t++) {
            noc_async_read(blk.get_noc_addr(hv * 10 + t), pk + t * tf, tf);
        }
        for (uint32_t j = 0; j < NC; j++) {
            noc_async_read(blk.get_noc_addr(hv * 10 + 4 + c0 + j), pu + j * tf, tf);
        }
        noc_async_read(blk.get_noc_addr(hv * 10 + 8), pw, tf);  // R (staged in the W page)
        noc_async_read(blk.get_noc_addr(hv * 10 + 9), pg, tf);  // gamma rows (staged in the gamma page)
        noc.async_read_barrier();
        uint32_t wv[16];
        for (uint32_t j = 0; j < NT; j++) {
            wv[j] = j <= a ? f32(pw)[eidx(a, j)] : 0u;
        }
        const uint32_t ga = f32(pg)[eidx(a, 0)];
        zfill(pw, tf);
        noc.async_read_barrier();
        for (uint32_t j = 0; j < NT; j++) {
            for (uint32_t c = 0; c < 32; c++) {
                f32(pw)[eidx(j, c)] = wv[j];
            }
        }
        for (uint32_t e = 0; e < 1024; e++) {
            f32(pg)[e] = ga;
        }
    } else {
        zfill(pk, 4 * tf);
        zfill(pu, NC * tf);
        zfill(pw, tf);
        noc.async_read_barrier();
        for (uint32_t e = 0; e < 1024; e++) {
            f32(pg)[e] = ONEF;
        }
    }
    ck.push_back(4);
    cu.push_back(NC);
    cw.push_back(1);
    cg.push_back(1);
#ifndef GDNWY_COMMIT
    // ---- constants (eltwise, unpack to dest): cb_cel = [Ms (j < i < NT), Mi (j <= i < NT), RowMask (i < NT), I (32x32)]
    {
        CircularBuffer ce(cb_cel);
        ce.reserve_back(4);
        const uint32_t pe = ce.get_write_ptr();
        zfill(pe, 4 * tf);
        noc.async_read_barrier();
        for (uint32_t r = 0; r < NT; r++) {
            for (uint32_t c = 0; c <= r; c++) {
                f32(pe + tf)[eidx(r, c)] = ONEF;
                if (c < r) {
                    f32(pe)[eidx(r, c)] = ONEF;
                }
            }
            for (uint32_t c = 0; c < 32; c++) {
                f32(pe + 2 * tf)[eidx(r, c)] = ONEF;
            }
        }
        for (uint32_t c = 0; c < 32; c++) {
            f32(pe + 3 * tf)[eidx(c, c)] = ONEF;
        }
        ce.push_back(4);
    }
    // ---- GDNQ_PRO prologue inputs (as gdnq4c_reader.cpp): A[k][t] rows s = conv input rows of slot s (k = 0..2:
    // window rows s + k, k = 3: in-projection row s), W[k][t] rows s = tap k of channel tile t. Rows NT .. 31 of every
    // A / W page are zero (the 8-page rings are zeroed first): the conv / norm rows of the unused slots are exactly 0.
    const auto xq = TensorAccessor(xa, x_addr, tb);
    const auto xl = TensorAccessor(xla, xl_addr, tb);
    const auto wp = TensorAccessor(wpa, wp_addr, tb);
    const auto onw = TensorAccessor(onwa, onw_addr, tb);
    auto chan = [&](uint32_t t) -> uint32_t {
        return t < 4 ? 4 * hk + t : (t < 8 ? KT0 + 4 * hk + (t - 4) : VT0 + 4 * hv + (t - 8));
    };
    constexpr uint32_t NT9 = 8 + NC;
    {
        CircularBuffer cpa(37), cpw(38);
        zfill(cpa.get_write_ptr(), 8 * tb);
        zfill(cpw.get_write_ptr(), 8 * tb);
        noc.async_read_barrier();
        for (uint32_t t = 0; t < NT9; t++) {
            const uint32_t tt = t < 8 ? t : 8 + c0 + (t - 8);
            const uint32_t c = chan(tt);
            cpa.reserve_back(4);
            cpw.reserve_back(4);
            const uint32_t pa = cpa.get_write_ptr(), pwt = cpw.get_write_ptr();
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(xl.get_noc_addr(c, f * 512), pa + 0 * tb + f * 512, NT * 32);
                noc_async_read(xl.get_noc_addr(c, f * 512 + 32), pa + 1 * tb + f * 512, NT * 32);
                noc_async_read(xl.get_noc_addr(c, f * 512 + 64), pa + 2 * tb + f * 512, NT * 32);
                noc_async_read(xq.get_noc_addr(c, XOFF(f, 0)), pa + 3 * tb + f * 512, NT * 32);
                for (uint32_t k = 0; k < 4; k++) {
                    noc_async_read(wp.get_noc_addr((hv * 4 + k) * 12 + tt, f * 512), pwt + k * tb + f * 512, NT * 32);
                }
            }
            noc.async_read_barrier();
            cpa.push_back(4);
            cpw.push_back(4);
        }
        // ---- Gfull (row t = Gc_t = g_0 + .. + g_t), D (D[t, j] = g_{j+1} + .. + g_t, j <= t), betam (row t = beta_t),
        // all on every column of rows < NT, zero elsewhere: fp32 sums here (a tf32 matmul cumsum loses ~1e-3 of Gc)
        {
            CircularBuffer cgg(35), cbe(36), cgf(56), cdd(63), cbm(57);
            cgg.wait_front(1);
            cbe.wait_front(1);
            cgf.reserve_back(1);
            cdd.reserve_back(1);
            cbm.reserve_back(1);
            const uint32_t pgf = cgf.get_write_ptr(), pdd = cdd.get_write_ptr(), pbm = cbm.get_write_ptr();
            zfill(pgf, tf);
            zfill(pdd, tf);
            zfill(pbm, tf);
            noc.async_read_barrier();
            union FU { uint32_t u; float f; };
            float g[16], gc[16];
            float acc = 0.0f;
            for (uint32_t r = 0; r < NT; r++) {
                FU v;
                v.u = f32(cgg.get_read_ptr())[eidx(r, 0)];
                g[r] = v.f;
                acc = acc + v.f;
                gc[r] = acc;
            }
            for (uint32_t r = 0; r < NT; r++) {
                FU gv, bv;
                gv.f = gc[r];
                bv.u = f32(cbe.get_read_ptr())[eidx(r, 0)];
                for (uint32_t c = 0; c < 32; c++) {
                    f32(pgf)[eidx(r, c)] = gv.u;
                    f32(pbm)[eidx(r, c)] = bv.u;
                }
                float d = 0.0f;
                for (int32_t j = (int32_t)r; j >= 0; j--) {  // D[r, j] = g_{j+1} + .. + g_r
                    FU dv;
                    dv.f = d;
                    f32(pdd)[eidx(r, (uint32_t)j)] = dv.u;
                    d = d + g[j];
                }
            }
            cgf.push_back(1);
            cdd.push_back(1);
            cbm.push_back(1);
            cgg.pop_front(1);
            cbe.pop_front(1);
        }
        // the q / k norm scale of every slot: element (r, r) of the rsqrt'd gram -> row r of a full-row tile
        CircularBuffer c39(39), c40(40);
        for (uint32_t h = 0; h < 2; h++) {
            c39.wait_front(1);
            c40.reserve_back(1);
            const uint32_t src = c39.get_read_ptr(), dst = c40.get_write_ptr();
            zfill(dst, tf);
            noc.async_read_barrier();
            for (uint32_t r = 0; r < NT; r++) {
                const uint32_t v = f32(src)[r * 16 + r];
                for (uint32_t cc = 0; cc < 16; cc++) {
                    f32(dst)[r * 16 + cc] = v;
                    f32(dst)[256 + r * 16 + cc] = v;
                }
            }
            c40.push_back(1);
            c39.pop_front(1);
        }
    }
    // ---- z rows (rows = slots), the norm weight on every slot row, then the rms scale of every slot (gdnq4c RMSALL)
    {
        CircularBuffer cz(cb_z), conw(cb_onw);
        cz.reserve_back(NC);
        conw.reserve_back(NC);
        const uint32_t pz = cz.get_write_ptr(), pw2 = conw.get_write_ptr();
        for (uint32_t k = 0; k < NC; k++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(xq.get_noc_addr(ZT0 + 4 * hv + c0 + k, XOFF(f, 0)), pz + k * tb + f * 512, NT * 32);
            }
            noc.async_read(onw, conw, tb, {.page_id = c0 + k}, {.offset_bytes = k * tb});
        }
        noc.async_read_barrier();
        for (uint32_t k = 0; k < NC; k++) {
            for (uint32_t f = 0; f < 2; f++) {
                for (uint32_t r = 1; r < NT; r++) {
                    noc_async_read(get_noc_addr(pw2 + k * tb + f * 512), pw2 + k * tb + f * 512 + r * 32, 32);
                }
            }
        }
        noc.async_read_barrier();
        cz.push_back(NC);
        conw.push_back(NC);
        CircularBuffer c39(39), c40(40);
        c39.wait_front(1);
        c40.reserve_back(1);
        const uint32_t src = c39.get_read_ptr(), dst = c40.get_write_ptr();
        zfill(dst, tf);
        noc.async_read_barrier();
        for (uint32_t r = 0; r < NT; r++) {
            const uint32_t v = f32(src)[r * 16 + r];
            for (uint32_t cc = 0; cc < 16; cc++) {
                f32(dst)[r * 16 + cc] = v;
                f32(dst)[256 + r * 16 + cc] = v;
            }
        }
        c40.push_back(1);
        c39.pop_front(1);
    }
#endif
    noc_async_write_barrier();
}
