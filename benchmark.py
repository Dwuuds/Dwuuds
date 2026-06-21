"""Within-life learning benchmark (acceptance evidence for Mode 1).

Proves the first demoable claim headlessly and reproducibly: a creature that can
learn measurably outperforms an otherwise-identical creature with plasticity
switched off.

Method. For each pair we draw one random born brain (a random genome). We run it
twice in matched worlds: once as a LEARNER (plasticity on) and once as a frozen
CONTROL -- the *same* born brain and the *same* exploration noise, with only the
synaptic learning disabled. So the single difference between the two is whether
the three-factor Hebbian rule is allowed to change the weights. Averaging over
many random born brains keeps the result honest (no cherry-picked wiring).

Creatures are born nearly blank (weak innate wiring), so there is real headroom
for learning to fill in -- the brain is born from the genome and then goes
plastic, exactly as in the brief.

To isolate *foraging skill* from mortality, the probe holds glucose at a mildly
hungry level so the creature neither starves nor ages out; we measure food eaten
per time window over its life.

Run:  python benchmark.py [--pairs 40] [--ticks 24000] [--csv learning.csv]
"""

from __future__ import annotations

import argparse
import csv as csvmod

import numpy as np

from config import Config
from genome import Genome
from world import World

PROBE_GLUCOSE = 45.0  # mildly hungry: keeps the appetitive drive engaged


def run_individual(pair: int, learn: bool, ticks: int, n_windows: int):
    cfg = Config()
    cfg.max_age_ticks = 10**9  # isolate learning from age-death during the probe
    world = World(cfg, np.random.default_rng(pair))
    genome = Genome.random(np.random.default_rng(7000 + pair))
    c = world.add_creature(genome)
    if not learn:
        c.brain.freeze()

    window = max(1, ticks // n_windows)
    windows, last = [], 0
    for t in range(ticks):
        world.step()
        c.biochem.glucose = PROBE_GLUCOSE  # never starve, stay mildly hungry
        c.alive = True
        if (t + 1) % window == 0:
            windows.append(c.food_eaten - last)
            last = c.food_eaten
    return c.food_eaten, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--ticks", type=int, default=24000)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    lwin = np.zeros(args.windows)
    cwin = np.zeros(args.windows)
    lt = ct = wins = 0
    rows = []
    for i in range(args.pairs):
        a, lw = run_individual(i, True, args.ticks, args.windows)
        b, cw = run_individual(i, False, args.ticks, args.windows)
        lwin += lw
        cwin += cw
        lt += a
        ct += b
        wins += int(a > b)
        rows.append((i, a, b))

    lwin /= args.pairs
    cwin /= args.pairs
    print(f"pairs={args.pairs} ticks={args.ticks}  (each pair: one random born "
          f"brain, learner vs frozen twin)")
    print(f"learner food/life (mean): {lt/args.pairs:.1f}")
    print(f"control food/life (mean): {ct/args.pairs:.1f}")
    print(f"learner / control:        {lt/max(1,ct):.2f}x")
    print(f"learner beats control:    {wins}/{args.pairs} brains")
    print("learner food per window (early -> late): " +
          " ".join(f"{v:.1f}" for v in lwin))
    print("control food per window (early -> late): " +
          " ".join(f"{v:.1f}" for v in cwin))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csvmod.writer(f)
            w.writerow(["pair", "learner_total", "control_total"])
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
