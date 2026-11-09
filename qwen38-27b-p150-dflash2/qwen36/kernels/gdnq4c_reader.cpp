// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ep2 t3 (Exit the Moat, Episode 2): the DFlash2 verify's chained GDN slots as ONE program (DF_GDN_CHAIN1=1): the
// gdnq4 reader (QWEN36_GDN_FAST >= 4, B = 1) looped over the GDNQ_CHAIN slots of one sequence. Per slot t: the x / z
// rows are row t of the [1, T, W] in-projection, the conv history rows are rows t .. t + 2 of the L1 verify window
// X [hist(3); slots; zeros] (st1 = st2 = st3 = X, no conv-state shift), the state comes from the base state (t = 0)
// or from the state the writer stored for slot t - 1 (bufs[t - 1], after its local semaphore 3 reached t). The
// taps, selectors and norm weight are read once. Every slot computes exactly what one gdnq4 B = 1 call computes.

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

constexpr uint32_t cb_x = 0, cb_q = 1, cb_w = 2, cb_z = 3, cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_sdst = 8,
                   cb_onw = 9, cb_scr = 10, cb_sel = 11, cb_smm = 13;

void kernel_main() {
    constexpr uint32_t SEM = get_compile_time_arg_val(0);
    constexpr uint32_t NK = get_compile_time_arg_val(1);
    constexpr uint32_t NV = get_compile_time_arg_val(2);
    constexpr uint32_t ZT0 = get_compile_time_arg_val(3);
    constexpr uint32_t ACOL = get_compile_time_arg_val(4);
    constexpr uint32_t BCOL = get_compile_time_arg_val(5);
    constexpr uint32_t CMODE = get_compile_time_arg_val(6);
    static_assert(CMODE == 1, "v3: the TP 4-deep conv shift register only");
    constexpr uint32_t STF32 = get_compile_time_arg_val(7);
    constexpr uint32_t CS = get_compile_time_arg_val(8);
    constexpr uint32_t NC = 4 / CS;
    constexpr uint32_t RF = NV / NK;
    constexpr uint32_t NT = GDNQ_CHAIN;
    constexpr uint32_t SEMST = 3;  // writer -> reader: states stored so far
    constexpr auto xa = TensorAccessorArgs<9>();
    constexpr auto s0a = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto s1a = TensorAccessorArgs<s0a.next_compile_time_args_offset()>();
    constexpr auto s2a = TensorAccessorArgs<s1a.next_compile_time_args_offset()>();
    constexpr auto s3a = TensorAccessorArgs<s2a.next_compile_time_args_offset()>();
    constexpr auto wpa = TensorAccessorArgs<s3a.next_compile_time_args_offset()>();
    constexpr auto sela = TensorAccessorArgs<wpa.next_compile_time_args_offset()>();
    constexpr auto reca = TensorAccessorArgs<sela.next_compile_time_args_offset()>();
    constexpr auto dtba = TensorAccessorArgs<reca.next_compile_time_args_offset()>();
    constexpr auto anega = TensorAccessorArgs<dtba.next_compile_time_args_offset()>();
    constexpr auto onwa = TensorAccessorArgs<anega.next_compile_time_args_offset()>();
#ifdef GDNQ_AIDX
    // ep2 t3b DF_NOCOMMIT: the previous step's accepted slot a sits at element T*T of the KV-skip position input;
    // a < NT: slot 0 starts from bufs[a] (no commit copy), else from the base state
    constexpr auto kva = TensorAccessorArgs<onwa.next_compile_time_args_offset()>();
#endif
    constexpr uint32_t KT0 = 4 * NK, VT0 = 8 * NK;

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    i++;  // p (0)
    i++;  // B (1)
    const uint32_t ci = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    i++;  // st0 (unused: no shift)
    const uint32_t s1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s3_addr = get_arg_val<uint32_t>(i++);
    const uint32_t wp_addr = get_arg_val<uint32_t>(i++);
    i++;  // the selector tensor: generated here
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t aneg_addr = get_arg_val<uint32_t>(i++);
    const uint32_t onw_addr = get_arg_val<uint32_t>(i++);
#ifdef GDNQ_AIDX
    const uint32_t kv_addr = get_arg_val<uint32_t>(i++);
#endif
    i += 2;  // owner coordinates; the forward targets follow (owner cores only), then the chain state addresses

    const uint32_t hk = hv / RF;
    const uint32_t c0 = ci * NC;
    const bool owner = (hv % RF) == 0 && ci == 0;
    const uint32_t ca = i + ((owner && RF * CS > 1) ? 2 * (RF * CS - 1) : 0);  // bufs[0 .. NT - 2]
#ifdef GDNQ_MSRT
#ifndef GDNQ_AIDX
#error "GDNQ_MSRT needs GDNQ_AIDX (NT chain state addresses before the row arg)"
#endif
    const uint32_t xrow0_ = get_arg_val<uint32_t>(ca + NT);  // ms1o: this core group's stream rows
#else
    constexpr uint32_t xrow0_ = GDNQ_XROW0;
#endif
    (void)dtb_addr;
    (void)aneg_addr;

    constexpr uint32_t tb = 2048, tf = STF32 ? 4096 : 2048;
    const auto xq = TensorAccessor(xa, x_addr, tb);
    const auto st1 = TensorAccessor(s1a, s1_addr, tb);
    const auto st2 = TensorAccessor(s2a, s2_addr, tb);
    const auto st3 = TensorAccessor(s3a, s3_addr, tb);
    const auto wp = TensorAccessor(wpa, wp_addr, tb);
    const auto onw = TensorAccessor(onwa, onw_addr, tb);

    auto chan = [&](uint32_t t) -> uint32_t {
        return t < 4 ? 4 * hk + t : (t < 8 ? KT0 + 4 * hk + (t - 4) : VT0 + 4 * hv + (t - 8));
    };
    auto poff = [&](uint32_t t, uint32_t f) -> uint32_t { return (f + 2 * (t / 8)) * 512 + (t % 8) * 64; };

    Noc noc;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    CircularBuffer cx(cb_x), cq(cb_q), cw(cb_w), csel(cb_sel), csd(cb_sdst), csm(cb_smm), conw(cb_onw), cz(cb_z);
    constexpr uint32_t NSEL = 8 + NC;
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
    volatile tt_l1_ptr uint32_t* semst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEMST));
#ifdef GDNQ_STL1
#ifndef GDNQ_STRING
#define GDNQ_STRING 16
#endif
    static_assert(GDNQ_STRING % (4 * NC) == 0, "cb_state ring of GDNQ_STRING tiles");
    const uint32_t stl1_base = CircularBuffer(17).get_write_ptr();  // never pushed by this RISC: the CB's base
#endif

    uint32_t slot0_addr = rec_addr;
#ifdef GDNQ_AIDX
    {
        const auto kvt = TensorAccessor(kva, kv_addr, (NT * NT + 16) * 4);
        CircularBuffer csd0(cb_sdst);
        csd0.reserve_back(1);
        const uint32_t pa = csd0.get_write_ptr();
        noc_async_read(kvt.get_noc_addr(0, NT * NT * 4), pa, 64);
        noc_async_read_barrier();
        const uint32_t a = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pa);
        if (a < NT) {
            slot0_addr = get_arg_val<uint32_t>(ca + a);
        }
    }
#endif
#ifdef GDNQ_TS
    CircularBuffer cts(27);
    cts.reserve_back(1);
    volatile tt_l1_ptr uint32_t* tsp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cts.get_write_ptr());
#define RTS(k) tsp[k] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
    for (uint32_t z = 0; z < 512; z++) {
        tsp[z] = 0;
    }
    RTS(31);
    cts.push_back(1);
#else
#define RTS(k)
#endif
#ifdef GDNQ_PRO
    // ep2 t3b GDNQ_PRO (see gdnq4c_compute.cpp): A[k][t] rows s = conv input rows of slot s (k = 0..2: window rows
    // s + k, k = 3: in-projection row s), W[k][t] rows s = tap k of channel tile t (from the packed taps row)
    constexpr uint32_t NT9 = 8 + NC;
    auto zfill = [&](uint32_t addr, uint32_t bytes) {  // MEM_ZEROS is 512 bytes
        for (uint32_t o = 0; o < bytes; o += 512) {
            noc_async_read(zeros, addr + o, 512);
        }
    };
    {
        CircularBuffer cpa(37), cpw(38);
        cw.reserve_back(4);
        // per channel tile t: A[t][k] (k = 0..3) and W[t][k] as 4 + 4 tiles, pushed as soon as read (the compute's conv
        // of tile t overlaps the reads of t + 1); only rows 0 .. NT-1 are read (the other rows are never used)
#ifndef GDNQ_PRO_ONEBAR
        for (uint32_t t = 0; t < NT9; t++) {
            const uint32_t tt = t < 8 ? t : 8 + c0 + (t - 8);
            const uint32_t c = chan(tt);
            cpa.reserve_back(4);
            cpw.reserve_back(4);
            const uint32_t pa = cpa.get_write_ptr(), pwt = cpw.get_write_ptr();
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(st1.get_noc_addr(c, f * 512), pa + 0 * tb + f * 512, NT * 32);
                noc_async_read(st2.get_noc_addr(c, f * 512 + 32), pa + 1 * tb + f * 512, NT * 32);
                noc_async_read(st3.get_noc_addr(c, f * 512 + 64), pa + 2 * tb + f * 512, NT * 32);
                noc_async_read(xq.get_noc_addr(c, XOFF(f, 0)), pa + 3 * tb + f * 512, NT * 32);
                for (uint32_t k = 0; k < 4; k++) {  // W tiles prebuilt on the host (fused_decode wbat)
                    noc_async_read(wp.get_noc_addr((hv * 4 + k) * 12 + tt, f * 512), pwt + k * tb + f * 512, NT * 32);
                }
            }
            noc.async_read_barrier();
            cpa.push_back(4);
            cpw.push_back(4);
        }
#else
        // all reads in flight at once, one barrier (CB 37 / 38 hold all 4 * NT9 tiles)
        cpa.reserve_back(4 * NT9);
        cpw.reserve_back(4 * NT9);
        const uint32_t pa0 = cpa.get_write_ptr(), pw0 = cpw.get_write_ptr();
        for (uint32_t t = 0; t < NT9; t++) {
            const uint32_t tt = t < 8 ? t : 8 + c0 + (t - 8);
            const uint32_t c = chan(tt);
            const uint32_t pa = pa0 + 4 * t * tb, pwt = pw0 + 4 * t * tb;
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(st1.get_noc_addr(c, f * 512), pa + 0 * tb + f * 512, NT * 32);
                noc_async_read(st2.get_noc_addr(c, f * 512 + 32), pa + 1 * tb + f * 512, NT * 32);
                noc_async_read(st3.get_noc_addr(c, f * 512 + 64), pa + 2 * tb + f * 512, NT * 32);
                noc_async_read(xq.get_noc_addr(c, XOFF(f, 0)), pa + 3 * tb + f * 512, NT * 32);
                for (uint32_t k = 0; k < 4; k++) {
                    noc_async_read(wp.get_noc_addr((hv * 4 + k) * 12 + tt, f * 512), pwt + k * tb + f * 512, NT * 32);
                }
            }
        }
        noc.async_read_barrier();
        cpa.push_back(4 * NT9);
        cpw.push_back(4 * NT9);
#endif
        RTS(310);
        // the q / k norm scale of every slot: element (r, r) of the rsqrt'd gram -> (r, 0) of the column tile
        CircularBuffer c39(39), c40(40);
        for (uint32_t h = 0; h < 2; h++) {
            c39.wait_front(1);
            c40.reserve_back(1);
            const uint32_t src = c39.get_read_ptr(), dst = c40.get_write_ptr();
            zfill(dst, 4096);
            noc.async_read_barrier();
            volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
            volatile tt_l1_ptr uint32_t* dp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
            for (uint32_t r = 0; r < NT; r++) {  // row r of faces 0 / 1 = the scale of slot r (a full-row tile)
                const uint32_t v = sp[r * 16 + r];
                for (uint32_t cc = 0; cc < 16; cc++) {
                    dp[r * 16 + cc] = v;
                    dp[256 + r * 16 + cc] = v;
                }
            }
            c40.push_back(1);
            c39.pop_front(1);
        }
    }
    CircularBuffer c41(41), c42(42), c43(43), cyb(19), cqn(21), ckn(22);
    c41.wait_front(NT9);
    c42.wait_front(4);
    c43.wait_front(4);
    const uint32_t pyall = c41.get_read_ptr(), pqall = c42.get_read_ptr(), pkall = c43.get_read_ptr();
#ifdef GDNQ_PRO2
    CircularBuffer c44(44), c45(45), ckc(23), cqk(18);
    c44.wait_front(4);
    c45.wait_front(1);
    const uint32_t pkcall = c44.get_read_ptr(), pqkall = c45.get_read_ptr();
#endif
#endif
    for (uint32_t tk = 0; tk < NT; tk++) {
#ifdef GDNQ_PRO
        {
            // ---- this slot's rows -> row 0 of the y (v), qn, kn tiles (the rest stays zero)
            cyb.reserve_back(12);
            cqn.reserve_back(4);
            ckn.reserve_back(4);
            const uint32_t py = cyb.get_write_ptr(), pqn = cqn.get_write_ptr(), pkn = ckn.get_write_ptr();
            if (tk == 0) {
                zfill(py, 12 * tb);
                zfill(pqn, 4 * 4096);
                zfill(pkn, 4 * 4096);
                noc.async_read_barrier();
            }
            for (uint32_t j = 0; j < NC; j++) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_read(get_noc_addr(pyall + (8 + j) * tb + f * 512 + tk * 32), py + (8 + j) * tb + f * 512, 32);
                }
            }
            for (uint32_t q = 0; q < 4; q++) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_read(get_noc_addr(pqall + q * 4096 + f * 1024 + tk * 64), pqn + q * 4096 + f * 1024, 64);
                    noc_async_read(get_noc_addr(pkall + q * 4096 + f * 1024 + tk * 64), pkn + q * 4096 + f * 1024, 64);
                }
            }
            noc.async_read_barrier();
            cyb.push_back(12);
            cqn.push_back(4);
            ckn.push_back(4);
#ifdef GDNQ_PRO2
            {
                // k of slot tk as column 0 (fp32 tiles: column tk of faces 0 / 2 -> column 0), q . k at (0, 0)
                ckc.reserve_back(4);
                cqk.reserve_back(1);
                const uint32_t pkc = ckc.get_write_ptr(), pqk = cqk.get_write_ptr();
                if (tk == 0) {
                    zfill(pkc, 4 * 4096);
                    zfill(pqk, 4096);
                    noc.async_read_barrier();
                }
                for (uint32_t q = 0; q < 4; q++) {
                    volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pkcall + q * 4096);
                    volatile tt_l1_ptr uint32_t* dp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pkc + q * 4096);
                    for (uint32_t r = 0; r < 16; r++) {
                        dp[r * 16] = sp[r * 16 + tk];
                        dp[512 + r * 16] = sp[512 + r * 16 + tk];
                    }
                }
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pqk)[0] =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pqkall)[tk * 16 + tk];
                ckc.push_back(4);
                cqk.push_back(1);
            }
#endif
        }
#else
        // ---- phase A: conv rows (packed), taps + selectors (first slot)
        cx.reserve_back(1);
        cq.reserve_back(3);
        const uint32_t px = cx.get_write_ptr(), pq = cq.get_write_ptr();
        uint32_t psel = 0;
        if (tk == 0) {
            cw.reserve_back(4);
            csel.reserve_back(NSEL);
            psel = csel.get_write_ptr();
        }
        for (uint32_t t = owner ? 0 : 8; t < 12; t++) {
            if (t >= 8 && (t - 8 < c0 || t - 8 >= c0 + NC)) {
                continue;
            }
            const uint32_t c = chan(t);
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t d = poff(t, f);
                noc_async_read(xq.get_noc_addr(c, XOFF(f, tk)), px + d, 32);
                noc_async_read(st1.get_noc_addr(c, f * 512 + tk * 32), pq + d, 32);
                noc_async_read(st2.get_noc_addr(c, f * 512 + (tk + 1) * 32), pq + tb + d, 32);
                noc_async_read(st3.get_noc_addr(c, f * 512 + (tk + 2) * 32), pq + 2 * tb + d, 32);
            }
        }
        if (tk == 0) {
            for (uint32_t t = 0; t < 4; t++) {
                noc.async_read(wp, cw, tb, {.page_id = 4 * hv + t}, {.offset_bytes = t * tb});
            }
            for (uint32_t k = 0; k < NSEL; k++) {
                for (uint32_t q4 = 0; q4 < 4; q4++) {
                    noc_async_read(zeros, psel + k * tb + q4 * 512, 512);
                }
            }
        }
        for (uint32_t k = 0; k < 4; k++) {
            const uint32_t pt = k == 0 ? px : pq + (k - 1) * tb;
            for (uint32_t f = 2; f < 4; f++) {
                noc_async_read(zeros, pt + f * 512 + 256, 256);
                for (uint32_t v = 0; v < 4; v++) {
                    if (v < c0 || v >= c0 + NC) {
                        noc_async_read(zeros, pt + f * 512 + v * 64, 64);
                    }
                }
            }
        }
        noc.async_read_barrier();
        if (tk == 0) {
            for (uint32_t k = 0; k < NSEL; k++) {
                const uint32_t t = k < 8 ? k : 8 + c0 + (k - 8);
                const uint32_t R = (t / 8) * 16 + 2 * (t % 8);
                volatile tt_l1_ptr uint16_t* e =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(psel + k * tb + (R / 16) * 512);
                e[R % 16] = 0x3f80;
            }
        }
        // the packed partner rows (sequence 1 of the pair) of x and of the 3 history tiles: zero
        for (uint32_t t = owner ? 0 : 8; t < 12; t++) {
            if (t >= 8 && (t - 8 < c0 || t - 8 >= c0 + NC)) {
                continue;
            }
            for (uint32_t f = 0; f < 2; f++) {
                for (uint32_t m = 0; m < 4; m++) {
                    const uint32_t pt = m == 0 ? px : pq + (m - 1) * tb;
                    volatile tt_l1_ptr uint32_t* q32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt + poff(t, f) + 32);
                    for (uint32_t w = 0; w < 8; w++) {
                        q32[w] = 0;
                    }
                }
            }
        }
        if constexpr (RF * CS > 1) {
            if (owner) {  // forward the q / k rows (faces 0 / 1 of the 4 packed tiles) to the other cores of the key head
                for (uint32_t k = 0; k < 2 * (RF * CS - 1); k += 2) {
                    const uint32_t nx = get_arg_val<uint32_t>(i + k), ny = get_arg_val<uint32_t>(i + k + 1);
                    noc_async_write(px, get_noc_addr(nx, ny, px), 1024);
                    for (uint32_t m = 0; m < 3; m++) {
                        noc_async_write(pq + m * tb, get_noc_addr(nx, ny, pq + m * tb), 1024);
                    }
                }
                noc_async_write_barrier();
                for (uint32_t k = 0; k < 2 * (RF * CS - 1); k += 2) {
                    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(i + k), get_arg_val<uint32_t>(i + k + 1), get_semaphore(SEM)), 1);
                }
            } else {
                noc_semaphore_wait_min(sem, tk + 1);
            }
        }
        cx.push_back(1);
        cq.push_back(3);
        RTS(32 * tk + 23);
        if (tk == 0) {
            cw.push_back(4);
            csel.push_back(NSEL);
        }

#endif
        // ---- phase B: the state (base for slot 0, the stored state of slot tk - 1 after it), z, norm weight
        csd.reserve_back(4 * NC);
        csm.reserve_back(4 * NC);
        const uint32_t pm = csm.get_write_ptr(), pd = csd.get_write_ptr();
        uint32_t ra = slot0_addr;
        if (tk > 0) {
            ra = get_arg_val<uint32_t>(ca + tk - 1);
            RTS(32 * tk + 24);
            noc_semaphore_wait_min(semst, tk);
            RTS(32 * tk + 25);
        }
        const auto rec = TensorAccessor(reca, ra, tf);
#ifdef GDNQ_STL1
        if (tk > 0) {
            // ms1o GDNQ_STL1: the state of slot tk - 1 straight from the writer's cb_state pages on this core (a ring
            // of 16 tiles, 4 * NC per slot; the compute cannot write the next state before this copy is pushed), in
            // the order the writer stores it to bufs[tk - 1]: no DRAM read back of the state
            noc_async_read(get_noc_addr(stl1_base + (((tk - 1) * 4 * NC) % GDNQ_STRING) * tf), pm, 4 * NC * tf);
        } else
#endif
        for (uint32_t ki = 0; ki < 4; ki++) {
            for (uint32_t j = 0; j < NC; j++) {
                noc_async_read(rec.get_noc_addr(hv * 16 + ki * 4 + c0 + j), pm + (ki * NC + j) * tf, tf);
            }
        }
        if (tk == 0) {
            conw.reserve_back(NC);
        }
#ifndef GDNQ_RMSALL
        cz.reserve_back(NC);
        const uint32_t pz = cz.get_write_ptr();
        for (uint32_t k = 0; k < NC; k++) {  // z tile: row 0 = row tk of qkvzab tile ZT0 + 4hv + c0 + k, the rest 0
            const uint32_t pt = pz + k * tb;
            noc_async_read(zeros, pt + 32, 480);
            noc_async_read(zeros, pt + 544, 480);
            noc_async_read(zeros, pt + 1024, 512);
            noc_async_read(zeros, pt + 1536, 512);
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(xq.get_noc_addr(ZT0 + 4 * hv + c0 + k, XOFF(f, tk)), pt + f * 512, 32);
            }
        }
        if (tk == 0) {
            for (uint32_t t = 0; t < NC; t++) {
                noc.async_read(onw, conw, tb, {.page_id = c0 + t}, {.offset_bytes = t * tb});
            }
        }
#endif
        noc.async_read_barrier();
        csm.push_back(4 * NC);
        RTS(32 * tk + 20);
        noc_async_read(get_noc_addr(pm), pd, 4 * NC * tf);
        noc.async_read_barrier();
        csd.push_back(4 * NC);
#ifndef GDNQ_RMSALL
        cz.push_back(NC);
        if (tk == 0) {
            conw.push_back(NC);
        }
#endif
    }
#ifdef GDNQ_RMSALL
    {
        // z rows 0 .. NT-1 (rows = slots) and the norm weight on rows 0 .. NT-1, then the rms scale of every slot
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
        for (uint32_t o = 0; o < 4096; o += 512) {
            noc_async_read(zeros, dst + o, 512);
        }
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        volatile tt_l1_ptr uint32_t* dp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        for (uint32_t r = 0; r < NT; r++) {
            const uint32_t v = sp[r * 16 + r];
            for (uint32_t cc = 0; cc < 16; cc++) {
                dp[r * 16 + cc] = v;
                dp[256 + r * 16 + cc] = v;
            }
        }
        c40.push_back(1);
        c39.pop_front(1);
    }
#endif
    noc_async_write_barrier();
}
