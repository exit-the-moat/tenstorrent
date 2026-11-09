// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 t3: the gdnq4 writer (B = 1) looped over the GDNQ_CHAIN slots of the DFlash2 verify (see gdnq4c_reader.cpp).
// dt_bias / A_neg once; per slot t: the gate inputs a / b of row t, the o exchange with the other CS - 1 cores of the
// head (cb_o slot t % 2, cumulative semaphore 1), the new state into bufs[t] (row 0) then local semaphore 3 + 1 (the
// reader reads it as the next slot's state), the gated output row t of the [1, T, NV*128] L1 output. The output tile
// rows T .. 31 are zeroed once.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// ms1o (multi-stream DFlash2 verify): the stream's rows sit at rows GDNQ_XROW0 .. + NT - 1 of the shared [1, R, W]
// in-projection and are written to rows GDNQ_OROW0 .. of a shared [1, R, Nv*128] output (no row slice / concat per
// stream); output rows GDNQ_OZ0 .. 31 are zeroed (the first stream's program only). Defaults = the single-stream code.
#ifndef GDNQ_XROW0
#define GDNQ_XROW0 0
#endif
#ifndef GDNQ_OROW0
#define GDNQ_OROW0 0
#endif
// (GDNQ_MSRT: the rows come as runtime args after the chain state addresses, one value per core group)
#define XOFF(f, r) (((xrow0_ + (r)) / 16) * 1024 + (f) * 512 + ((xrow0_ + (r)) % 16) * 32)
#define OOFF(f, r) (((orow0_ + (r)) / 16) * 1024 + (f) * 512 + ((orow0_ + (r)) % 16) * 32)
#ifndef GDNQ_OZ0
#define GDNQ_OZ0 NT
#endif

constexpr uint32_t cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_scr = 10, cb_out = 16, cb_state = 17, cb_o = 28,
                   cb_osig = 32, cb_orecv = 33;

void kernel_main() {
    constexpr uint32_t NV = get_compile_time_arg_val(0);
    constexpr uint32_t STF32 = get_compile_time_arg_val(1);
    constexpr uint32_t CS = get_compile_time_arg_val(2);
    constexpr uint32_t SEMO = get_compile_time_arg_val(3);
    constexpr uint32_t ACOL = get_compile_time_arg_val(4);
    constexpr uint32_t BCOL = get_compile_time_arg_val(5);
    constexpr uint32_t NC = 4 / CS;
    constexpr uint32_t NT = GDNQ_CHAIN;
    constexpr uint32_t SEMST = 3;
    constexpr auto reca = TensorAccessorArgs<6>();
    constexpr auto outa = TensorAccessorArgs<reca.next_compile_time_args_offset()>();
    constexpr auto xa = TensorAccessorArgs<outa.next_compile_time_args_offset()>();
    constexpr auto dtba = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto anega = TensorAccessorArgs<dtba.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    i++;  // p
    i++;  // B
    const uint32_t ci = get_arg_val<uint32_t>(i++);
    i++;  // last
    i++;  // rec (unused: the chain addresses below)
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t aneg_addr = get_arg_val<uint32_t>(i++);
    uint32_t peer[2 * (CS > 1 ? CS - 1 : 1)];
    for (uint32_t k = 0; k < 2 * (CS - 1); k++) {
        peer[k] = get_arg_val<uint32_t>(i++);
    }
    const uint32_t ca = i;  // bufs[0 .. NT - 1]
#ifdef GDNQ_MSRT
    const uint32_t xrow0_ = get_arg_val<uint32_t>(ca + NT);  // ms1o: this core group's stream rows (in / out / zero-from)
    const uint32_t orow0_ = get_arg_val<uint32_t>(ca + NT + 1);
    const uint32_t oz0_ = get_arg_val<uint32_t>(ca + NT + 2);
#else
    constexpr uint32_t xrow0_ = GDNQ_XROW0;
    constexpr uint32_t orow0_ = GDNQ_OROW0;
    constexpr uint32_t oz0_ = GDNQ_OZ0;
#endif
    const uint32_t c0 = ci * NC;

    constexpr uint32_t tb = 2048, tf = STF32 ? 4096 : 2048, to = 4096;
    const auto out = TensorAccessor(outa, out_addr, tb);
    const auto xq = TensorAccessor(xa, x_addr, tb);
    CircularBuffer cst(cb_state), cout(cb_out), co(cb_o), cos(cb_osig), cor(cb_orecv);
    CircularBuffer cdt(cb_dtb), can(cb_aneg), scr(cb_scr), cav(cb_a), cbv(cb_b);
    Noc noc;
    {
        const auto dtb = TensorAccessor(dtba, dtb_addr, tb);
        const auto aneg = TensorAccessor(anega, aneg_addr, tb);
        cdt.reserve_back(1);
        can.reserve_back(1);
        noc.async_read(dtb, cdt, tb, {.page_id = hv}, {.offset_bytes = 0});
        noc.async_read(aneg, can, tb, {.page_id = hv}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cdt.push_back(1);
        can.push_back(1);
    }
    // output tile rows GDNQ_OZ0 .. 31 (default NT .. 31): zeros (nothing depends on them)
    for (uint32_t k = 0; k < NC; k++) {
        const uint32_t pg = 4 * hv + c0 + k;
        const uint32_t oz = oz0_;
        if (oz < 16) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, f * 512 + oz * 32), (16 - oz) * 32);
            }
            noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, 1024), 512);
            noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, 1536), 512);
        } else if (oz < 32) {
            for (uint32_t f = 2; f < 4; f++) {
                noc_async_write(MEM_ZEROS_BASE, out.get_noc_addr(pg, f * 512 + (oz - 16) * 32), (32 - oz) * 32);
            }
        }
    }
    volatile tt_l1_ptr uint32_t* semo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEMO));
    const uint32_t obase = co.get_write_ptr();  // cb_o's base; slot s at + s * 4 tiles
#ifdef GDNQ_RMSALL
    CircularBuffer coall(46);
    coall.reserve_back(4);
    const uint32_t pall = coall.get_write_ptr();  // o_all: the same L1 address on every core of the program
#endif
    scr.reserve_back(1);
    const uint32_t ps = scr.get_write_ptr();
    const uint32_t ca_ = ACOL + hv, cbb = BCOL + hv;

#ifdef GDNQ_TS
    CircularBuffer cts(27);
    cts.wait_front(1);
    volatile tt_l1_ptr uint32_t* tsp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cts.get_read_ptr());
#define WTS(k) tsp[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
#else
#define WTS(k)
#endif
#ifdef GDNQ_GATES_ALL
    // ep2 t3b: slot r's gate inputs in row r (face 0) of one a / b tile, zeros elsewhere; the compute runs the gates
    // once; expand() copies slot tk's row of the results into the uniform per-slot decay / beta tiles
    {
        cav.reserve_back(1);
        cbv.reserve_back(1);
        const uint32_t ta0 = cav.get_write_ptr(), tb0 = cbv.get_write_ptr();
        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), ta0, tb);
        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), tb0, tb);
        for (uint32_t tk = 0; tk < NT; tk++) {
            noc_async_read(xq.get_noc_addr(ca_ / 32, XOFF((ca_ % 32) / 16, tk)), ps + tk * 128, 32);
            noc_async_read(xq.get_noc_addr(cbb / 32, XOFF((cbb % 32) / 16, tk)), ps + tk * 128 + 64, 32);
        }
        noc.async_read_barrier();
        for (uint32_t tk = 0; tk < NT; tk++) {
            volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ps + tk * 128);
            const uint32_t va = row[ca_ % 16], vb = row[32 + cbb % 16];
            volatile tt_l1_ptr uint32_t* ta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ta0 + tk * 32);
            volatile tt_l1_ptr uint32_t* tbp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb0 + tk * 32);
            const uint32_t wa = va | (va << 16), wb = vb | (vb << 16);
            for (uint32_t w = 0; w < 8; w++) {
                ta[w] = wa;
                tbp[w] = wb;
            }
        }
        cav.push_back(1);
        cbv.push_back(1);
    }
    CircularBuffer cdall(35), cball(36), cdec(30), cbet(31);
    cdall.wait_front(1);
    cball.wait_front(1);
    auto expand = [&](uint32_t tk) {
        cdec.reserve_back(1);
        cbet.reserve_back(1);
        const uint32_t td = cdec.get_write_ptr(), tbt = cbet.get_write_ptr();
        noc_async_read(get_noc_addr(cdall.get_read_ptr() + tk * 64), td, 64);
        noc_async_read(get_noc_addr(cball.get_read_ptr() + tk * 64), tbt, 64);
        noc.async_read_barrier();
        for (uint32_t sz = 64; sz < 4096; sz *= 2) {
            noc_async_read(get_noc_addr(td), td + sz, sz);
            noc_async_read(get_noc_addr(tbt), tbt + sz, sz);
            noc.async_read_barrier();
        }
        cdec.push_back(1);
        cbet.push_back(1);
    };
    expand(0);
#endif
#ifdef GDNQ_GATES_AHEAD
    // ep2 t3b: the gate inputs of every slot first (cb_a / cb_b hold NT tiles): the compute no longer waits ~4 us per
    // slot for the writer to finish the previous slot's state / output writes
    for (uint32_t tk = 0; tk < NT; tk++) {
        // ---- the gate inputs of row tk (as full tiles: the value in 64 bytes, then doubling local copies)
        cav.reserve_back(1);
        cbv.reserve_back(1);
        noc_async_read(xq.get_noc_addr(ca_ / 32, XOFF((ca_ % 32) / 16, tk)), ps, 32);
        noc_async_read(xq.get_noc_addr(cbb / 32, XOFF((cbb % 32) / 16, tk)), ps + 64, 32);
        noc.async_read_barrier();
        {
            volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ps);
            const uint32_t va = row[ca_ % 16], vb = row[32 + cbb % 16];
            volatile tt_l1_ptr uint32_t* ta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cav.get_write_ptr());
            volatile tt_l1_ptr uint32_t* tbp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbv.get_write_ptr());
            const uint32_t wa = va | (va << 16), wb = vb | (vb << 16);
            for (uint32_t w = 0; w < 16; w++) {
                ta[w] = wa;
                tbp[w] = wb;
            }
        }
        for (uint32_t sz = 64; sz < tb; sz *= 2) {
            const uint32_t ta = cav.get_write_ptr(), tbb = cbv.get_write_ptr();
            noc_async_read(get_noc_addr(ta), ta + sz, sz);
            noc_async_read(get_noc_addr(tbb), tbb + sz, sz);
            noc.async_read_barrier();
        }
        cav.push_back(1);
        cbv.push_back(1);

    }
#endif
    for (uint32_t tk = 0; tk < NT; tk++) {
#if !defined(GDNQ_GATES_AHEAD) && !defined(GDNQ_GATES_ALL)
        // ---- the gate inputs of row tk (as full tiles: the value in 64 bytes, then doubling local copies)
        cav.reserve_back(1);
        cbv.reserve_back(1);
        noc_async_read(xq.get_noc_addr(ca_ / 32, XOFF((ca_ % 32) / 16, tk)), ps, 32);
        noc_async_read(xq.get_noc_addr(cbb / 32, XOFF((cbb % 32) / 16, tk)), ps + 64, 32);
        noc.async_read_barrier();
        {
            volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ps);
            const uint32_t va = row[ca_ % 16], vb = row[32 + cbb % 16];
            volatile tt_l1_ptr uint32_t* ta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cav.get_write_ptr());
            volatile tt_l1_ptr uint32_t* tbp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbv.get_write_ptr());
            const uint32_t wa = va | (va << 16), wb = vb | (vb << 16);
            for (uint32_t w = 0; w < 16; w++) {
                ta[w] = wa;
                tbp[w] = wb;
            }
        }
        for (uint32_t sz = 64; sz < tb; sz *= 2) {
            const uint32_t ta = cav.get_write_ptr(), tbb = cbv.get_write_ptr();
            noc_async_read(get_noc_addr(ta), ta + sz, sz);
            noc_async_read(get_noc_addr(tbb), tbb + sz, sz);
            noc.async_read_barrier();
        }
        cav.push_back(1);
        cbv.push_back(1);

#endif
        // ---- o exchange
#ifdef GDNQ_RMSALL
        {
            // the own o row of slot tk -> row tk of o_all tile c0 + j on this core and on the other CS - 1 cores
            cos.wait_front(1);
            const uint32_t src = obase + ((tk % 2) * 4 + c0) * to;
            for (uint32_t j = 0; j < NC; j++) {
                for (uint32_t f = 0; f < 2; f++) {
                    const uint32_t so = src + j * to + f * 1024, d = pall + (c0 + j) * to + f * 1024 + tk * 64;
                    noc_async_write(so, get_noc_addr(d), 64);
                    for (uint32_t k = 0; k < CS - 1; k++) {
                        noc_async_write(so, get_noc_addr(peer[2 * k], peer[2 * k + 1], d), 64);
                    }
                }
            }
            noc_async_write_barrier();
            cos.pop_front(1);
#ifdef GDNQ_GATES_ALL
            if (tk + 1 < NT) {
                expand(tk + 1);
            }
#endif
        }
        if constexpr (false) {
#else
        if constexpr (CS > 1) {
#endif
            cos.wait_front(1);
            const uint32_t src = obase + ((tk % 2) * 4 + c0) * to;
            for (uint32_t k = 0; k < CS - 1; k++) {
                noc_async_write(src, get_noc_addr(peer[2 * k], peer[2 * k + 1], src), NC * to);
            }
            noc_async_write_barrier();
            for (uint32_t k = 0; k < CS - 1; k++) {
                noc_semaphore_inc(get_noc_addr(peer[2 * k], peer[2 * k + 1], get_semaphore(SEMO)), 1);
            }
            cos.pop_front(1);
#ifdef GDNQ_GATES_ALL
            if (tk + 1 < NT) {
                expand(tk + 1);
            }
#endif
            noc_semaphore_wait_min(semo, (CS - 1) * (tk + 1));
            cor.reserve_back(1);
            cor.push_back(1);
        }
        // ---- the new state -> bufs[tk] row 0, then tell the reader
        cst.wait_front(4 * NC);
#ifdef GDNQ_STL1
        if (tk + 1 < NT) {  // ms1o GDNQ_STL1: the reader copies the state from cb_state (L1), not from bufs[tk]
            noc_semaphore_inc(get_noc_addr(get_semaphore(SEMST)), 1);
            noc_async_atomic_barrier();
        }
#endif
        {
            const auto rec = TensorAccessor(reca, get_arg_val<uint32_t>(ca + tk), tf);
            const uint32_t pst = cst.get_read_ptr();
            for (uint32_t ki = 0; ki < 4; ki++) {
                for (uint32_t j = 0; j < NC; j++) {
                    noc_async_write(pst + (ki * NC + j) * tf, rec.get_noc_addr(hv * 16 + ki * 4 + c0 + j), tf);
                }
            }
        }
        noc_async_write_barrier();
        WTS(32 * tk + 21);
        cst.pop_front(4 * NC);
#ifndef GDNQ_STL1
        if (tk + 1 < NT) {
            noc_semaphore_inc(get_noc_addr(get_semaphore(SEMST)), 1);
            noc_async_atomic_barrier();
        }
#endif
#ifndef GDNQ_RMSALL
        // ---- the gated output row tk
        cout.wait_front(NC);
        const uint32_t po = cout.get_read_ptr();
        for (uint32_t k = 0; k < NC; k++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(po + k * tb + f * 512, out.get_noc_addr(4 * hv + c0 + k, OOFF(f, tk)), 32);
            }
        }
        noc_async_write_barrier();
        WTS(32 * tk + 22);
        cout.pop_front(NC);
#endif
    }
    noc_async_write_barrier();
#ifdef GDNQ_RMSALL
    {
        // all o rows of this core are on every core of the head: count in, wait for the others, release the compute
        for (uint32_t k = 0; k < CS - 1; k++) {
            noc_semaphore_inc(get_noc_addr(peer[2 * k], peer[2 * k + 1], get_semaphore(SEMO)), 1);
        }
        noc_semaphore_wait_min(semo, CS - 1);
        coall.push_back(4);
        cor.reserve_back(1);
        cor.push_back(1);
        // the gated output rows 0 .. NT-1 (rows = slots)
        cout.wait_front(NC);
        const uint32_t po = cout.get_read_ptr();
        for (uint32_t k = 0; k < NC; k++) {
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_write(po + k * tb + f * 512, out.get_noc_addr(4 * hv + c0 + k, OOFF(f, 0)), NT * 32);
            }
        }
        noc_async_write_barrier();
        cout.pop_front(NC);
    }
#endif
#ifdef GDNQ_TS
    {
        constexpr auto tsa = TensorAccessorArgs<anega.next_compile_time_args_offset()>();
        const uint32_t ti = ca + NT;
        const auto tsd = TensorAccessor(tsa, get_arg_val<uint32_t>(ti), 2048);
        noc_async_write(cts.get_read_ptr(), tsd.get_noc_addr(get_arg_val<uint32_t>(ti + 1)), 2048);
        noc_async_write_barrier();
    }
#endif
}
