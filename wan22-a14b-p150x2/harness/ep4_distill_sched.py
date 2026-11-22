"""Episode 4: sampler for the lightx2v 4-step distill.

EP4_DISTILL_SAMPLER=ref (default): the reference sampler of LightX2V (configs/distill/wan22/wan_moe_i2v_distill_model.json
+ lightx2v/models/schedulers/wan/step_distill/scheduler.py, read 18 Nov): denoising_step_list [1000, 750, 500, 250],
sample_shift 5.0 -> sigmas 1.0, 0.9375, 0.8333, 0.625 (timesteps = sigma * 1000), flow Euler update
x_next = x + (sigma_next - sigma) * v (last step to sigma 0), boundary_step_index 2 = steps 0-1 high-noise expert,
steps 2-3 low-noise. In the tt pipeline the expert is chosen by t >= boundary_ratio * 1000, so boundary 0.875 gives
the same split (1000, 937.5 high; 833.3, 625 low).
EP4_DISTILL_SAMPLER=tt: tt_dit's experimental default (base I2V UniPC scheduler, flow_shift 12, boundary 0.5): at 4 steps
the timesteps are 999, 973, 923, 800, so the low-noise expert is never used.
"""
import os

import torch

from models.tt_dit.solvers.euler import EulerSolver

MODE = os.environ.get("EP4_DISTILL_SAMPLER", "ref")
STEP_LIST = [1000, 750, 500, 250]
SHIFT = 5.0


def ref_sigmas():
    s = [1.0 - (1000 - t) / 1000.0 for t in STEP_LIST]  # linspace(1, 0, 1001)[1000 - t]
    return [SHIFT * x / (1 + (SHIFT - 1) * x) for x in s]


BOUNDARY = 0.875 if MODE == "ref" else 0.5


def install(pipe):
    """Patch a built WanDistillPipelineI2V to sample like LightX2V (MODE == 'ref'); no-op for 'tt'."""
    if MODE != "ref":
        return
    sig = ref_sigmas()
    sched = pipe._scheduler

    def set_timesteps(n, *args, **kw):
        assert n == len(STEP_LIST), f"the distill reference sampler has {len(STEP_LIST)} steps, got {n}"
        sched.sigmas = torch.tensor(sig + [0.0], dtype=torch.float32)
        sched.timesteps = torch.tensor([x * 1000.0 for x in sig], dtype=torch.float32)

    sched.set_timesteps = set_timesteps
    pipe._solver = EulerSolver()
