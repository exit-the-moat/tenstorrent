// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ms4: the WY block GDN verify, writer (see gdnwy_reader.cpp). dt_bias / A_neg, the gate inputs of all slots (rows =
// slots, as gdnq4c GDNQ_GATES_ALL), S0 -> base, this block K / U / R / gamma -> blk (page hv * 10 + 0..3 K, 4..7 U column
// tiles, 8 R, 9 gamma), the o column tiles to o_all on every core of the head, the gated output rows.
// GDNWY_COMMIT: S0 -> base only.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#define XOFF(f, r) (((xrow0_ + (r)) / 16) * 1024 + (f) * 512 + ((xrow0_ + (r)) % 16) * 32)
#define OOFF(f, r) (((orow0_ + (r)) / 16) * 1024 + (f) * 512 + ((orow0_ + (r)) % 16) * 32)

constexpr uint32_t cb_a = 4, cb_b = 5, cb_dtb = 6, cb_aneg = 7, cb_scr = 10, cb_out = 16, cb_state = 17, cb_u = 26,
                   cb_o = 28, cb_orecv = 33, cb_kn = 43, cb_rm = 47, cb_gam = 48;

void kernel_main() {
    constexpr uint32_t NV = get_compile_time_arg_val(0);
    constexpr uint32_t CS = get_compile_time_arg_val(1);
    constexpr uint32_t SEMO = get_compile_time_arg_val(2);
    constexpr uint32_t ACOL = get_compile_time_arg_val(3);
    constexpr uint32_t BCOL = get_compile_time_arg_val(4);
    constexpr uint32_t NC = 4 / CS;
    constexpr uint32_t NT = GDNQ_CHAIN;
    constexpr auto basea = TensorAccessorArgs<5>();
    constexpr auto outa = TensorAccessorArgs<basea.next_compile_time_args_offset()>();
    constexpr auto xa = TensorAccessorArgs<outa.next_compile_time_args_offset()>();
    constexpr auto dtba = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto anega = TensorAccessorArgs<dtba.next_compile_time_args_offset()>();
    constexpr auto blka = TensorAccessorArgs<anega.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t hv = get_arg_val<uint32_t>(i++);
    const uint32_t ci = get_arg_val<uint32_t>(i++);
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t aneg_addr = get_arg_val<uint32_t>(i++);
    const uint32_t base_addr = get_arg_val<uint32_t>(i++);
    const uint32_t blk_addr = get_arg_val<uint32_t>(i++);
    const uint32_t xrow0_ = get_arg_val<uint32_t>(i++);
    const uint32_t orow0_ = get_arg_val<uint32_t>(i++);
    const uint32_t oz0_ = get_arg_val<uint32_t>(i++);
    uint32_t peer[2 * (CS > 1 ? CS - 1 : 1)];
    for (uint32_t k = 0; k < 2 * (CS - 1); k++) {
        peer[k] = get_arg_val<uint32_t>(i++);
    }
    (void)out_addr;
    (void)x_addr;
    (void)dtb_addr;
    (void)aneg_addr;
    (void)xrow0_;
    (void)orow0_;
    (void)oz0_;
    const uint32_t c0 = ci * NC;
    constexpr uint32_t tb = 2048, tf = 4096;
    const auto base = TensorAccessor(basea, base_addr, tf);
    const auto blk = TensorAccessor(blka, blk_addr, tf);
    Noc noc;

#ifndef GDNWY_COMMIT
    const auto out = TensorAccessor(outa, out_addr, tb);
    const auto xq = TensorAccessor(xa, x_addr, tb);
    CircularBuffer cdt(cb_dtb), can(cb_aneg), scr(cb_scr), cav(cb_a), cbv(cb_b);
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
    // the gate inputs of all slots: slot r value in row r (face 0, columns 0..15) of one a / b tile, zeros elsewhere
    scr.reserve_back(1);
    const uint32_t ps = scr.get_write_ptr();
    const uint32_t ca_ = ACOL + hv, cbb = BCOL + hv;
    {
        cav.reserve_back(1);
        cbv.reserve_back(1);
        const uint32_t ta0 = cav.get_write_ptr(), tb0 = cbv.get_write_ptr();
        for (uint32_t o = 0; o < tb; o += 512) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), ta0 + o, 512);
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), tb0 + o, 512);
        }
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
    // output tile rows oz0 .. 31: zeros (nothing depends on them)
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
    CircularBuffer coall(46);
    coall.reserve_back(4);
    const uint32_t pall = coall.get_write_ptr();  // o_all: the same L1 address on every core of the program
#endif
    // ---- S0 -> base (own columns)
    {
        CircularBuffer cst(cb_state);
        cst.wait_front(4 * NC);
        const uint32_t pst = cst.get_read_ptr();
        for (uint32_t ki = 0; ki < 4; ki++) {
            for (uint32_t j = 0; j < NC; j++) {
                noc_async_write(pst + (ki * NC + j) * tf, base.get_noc_addr(hv * 16 + ki * 4 + c0 + j), tf);
            }
        }
        noc_async_write_barrier();
        cst.pop_front(4 * NC);
    }
#ifndef GDNWY_COMMIT
    // ---- this block for the next step: K, R, gamma (core ci = 0 of the head), U (own columns)
    {
        CircularBuffer ckn(cb_kn), crm(cb_rm), cgm(cb_gam), cuu(cb_u);
        if (ci == 0) {
            ckn.wait_front(4);
            crm.wait_front(1);
            cgm.wait_front(1);
            for (uint32_t t = 0; t < 4; t++) {
                noc_async_write(ckn.get_read_ptr() + t * tf, blk.get_noc_addr(hv * 10 + t), tf);
            }
            noc_async_write(crm.get_read_ptr(), blk.get_noc_addr(hv * 10 + 8), tf);
            noc_async_write(cgm.get_read_ptr(), blk.get_noc_addr(hv * 10 + 9), tf);
        }
        cuu.wait_front(NC);
        for (uint32_t j = 0; j < NC; j++) {
            noc_async_write(cuu.get_read_ptr() + j * tf, blk.get_noc_addr(hv * 10 + 4 + c0 + j), tf);
        }
        noc_async_write_barrier();
    }
    // ---- o column tiles -> o_all (this core and the other CS - 1 cores of the head), then release the compute
    {
        CircularBuffer co(cb_o), cor(cb_orecv);
        co.wait_front(NC);
        const uint32_t po = co.get_read_ptr();
        for (uint32_t j = 0; j < NC; j++) {
            const uint32_t d = pall + (c0 + j) * tf;
            noc_async_write(po + j * tf, get_noc_addr(d), tf);
            for (uint32_t k = 0; k < CS - 1; k++) {
                noc_async_write(po + j * tf, get_noc_addr(peer[2 * k], peer[2 * k + 1], d), tf);
            }
        }
        noc_async_write_barrier();
        co.pop_front(NC);
        volatile tt_l1_ptr uint32_t* semo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEMO));
        for (uint32_t k = 0; k < CS - 1; k++) {
            noc_semaphore_inc(get_noc_addr(peer[2 * k], peer[2 * k + 1], get_semaphore(SEMO)), 1);
        }
        noc_semaphore_wait_min(semo, CS - 1);
        coall.push_back(4);
        cor.reserve_back(1);
        cor.push_back(1);
    }
    // ---- the gated output rows 0 .. NT-1 (rows = slots)
    {
        CircularBuffer cout(cb_out);
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
    noc_async_write_barrier();
}
