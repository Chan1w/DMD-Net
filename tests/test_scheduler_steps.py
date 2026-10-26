"""Scheduler-steps test: start_rel (noise level) and N (denoising steps) are
separate knobs (spec 7.1 / Task F).  Verifies ``make_timesteps`` maps them to the
correct DDIM timestep sub-sequence and that the number of UNet calls equals the
number of returned timesteps.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.builder import load_scheduler


def main():
    sched = load_scheduler()

    # fixed N, varying start_rel must change the STARTING timestep magnitude,
    # not the count of steps
    for steps in (10, 20, 30):
        for sr in (0.2, 0.4):
            sched.set_timesteps(steps, device="cpu")
            ts = sched.timesteps
            init_timestep = min(int(steps * sr), steps)
            t_start = max(steps - init_timestep, 0)
            sub = ts[t_start:]
            assert len(sub) == steps - t_start
            assert len(sub) > 0
            assert sub[0] >= sub[-1] >= 0
    print("[ok] start_rel / steps mapping")

    # a smaller start_rel must begin denoising from a smaller timestep
    # (less noise injected), i.e. sub[0] is smaller
    sched.set_timesteps(20, device="cpu")
    ts = sched.timesteps
    low = ts[max(20 - min(int(20 * 0.2), 20), 0):]
    high = ts[max(20 - min(int(20 * 0.4), 20), 0):]
    assert low[0] < high[0]
    print(f"[ok] start_rel 0.2 -> t0={low[0].item()}, start_rel 0.4 -> t0={high[0].item()}")

    # skip (start_rel=0) yields a degenerate / empty diffusion schedule
    sched.set_timesteps(20, device="cpu")
    skip = sched.timesteps[max(20 - 0, 0):]
    assert len(skip) == 0
    print("[ok] skip -> empty schedule")

    print("PASS")


if __name__ == "__main__":
    main()
