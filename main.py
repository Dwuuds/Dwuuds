"""Entry point for the A-Life creature sim proof of concept.

  python main.py --mode interactive
      Open the pygame viewer. Click a creature to select it, hold R to reward and
      P to punish. Teach it to approach food / avoid the hazard and watch its
      food-rate sparkline pull ahead of the frozen control.

  python main.py --mode evolve --generations 200 --pop 40
      Run headless evolution and write per-generation statistics to stats.csv.

  python main.py --mode benchmark
      Headless within-life-learning benchmark (learner vs frozen control).
"""

from __future__ import annotations

import argparse

from config import Config


def build_config(args) -> Config:
    cfg = Config()
    if args.pop is not None:
        cfg.start_population = args.pop
        cfg.max_population = args.pop
    if args.seed is not None:
        cfg.seed = args.seed
    if args.ticks_per_gen is not None:
        cfg.ticks_per_generation = args.ticks_per_gen
    if args.mode == "evolve":
        # finite-but-renewing food creates competition; foraging skill, speed and
        # metabolic efficiency all bear on who eats enough to out-reproduce.
        cfg.n_food = 35
    return cfg


def main():
    ap = argparse.ArgumentParser(description="A-Life creature sim PoC")
    ap.add_argument("--mode", choices=["interactive", "evolve", "benchmark"],
                    default="interactive")
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--ticks-per-gen", dest="ticks_per_gen", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--csv", type=str, default="stats.csv")
    args = ap.parse_args()

    cfg = build_config(args)

    if args.mode == "interactive":
        from render import run_interactive
        run_interactive(cfg, seed=cfg.seed)
    elif args.mode == "evolve":
        from evolve import run_evolution
        run_evolution(cfg, args.generations, args.csv)
    elif args.mode == "benchmark":
        import benchmark
        import sys
        sys.argv = ["benchmark.py"]
        benchmark.main()


if __name__ == "__main__":
    main()
