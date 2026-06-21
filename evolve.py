"""Mode 2: headless evolution (proves heritable, evolvable genetics).

No rendering, accelerated ticks. A founder population of random genomes is run in
a world with finite food and hazards; foraging skill, speed and metabolic
efficiency all bear on how much a creature eats and how long it survives. At the
end of each generation the most successful creatures reproduce sexually
(crossover + mutation) to seed the next generation, and per-generation statistics
are written to CSV.

Plasticity still operates *within* each life (the brain learns while the
generation runs) but learned weights are never inherited -- only the genome
crosses the generation boundary. Evolution (between lives) and learning (within a
life) are deliberately separate systems; that separation is the whole point.

The acceptance criterion is a logged, measurable change in population statistics
across generations under the selection pressure.
"""

from __future__ import annotations

import argparse
import csv as csvmod

import numpy as np

from config import Config
from genome import Genome, GENE_SPECS
from world import World

# traits we log each generation so drift is visible in the CSV
TRACKED_TRAITS = [
    "max_speed", "metabolic_rate", "radius", "turn_rate",
    "learning_rate", "trace_decay", "hidden_size", "innate_food",
]


class Evolution:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # founder genomes
        self.genomes = [Genome.random(self.rng) for _ in range(cfg.start_population)]

    # ---- one generation ----------------------------------------------
    def run_generation(self) -> tuple[list, dict]:
        cfg = self.cfg
        # a fresh world; finite food so foraging actually matters
        world = World(cfg, self.rng)
        world.cfg = cfg
        creatures = [world.add_creature(g.copy()) for g in self.genomes]

        for _ in range(cfg.ticks_per_generation):
            world.step()
            if world.n_alive == 0:
                break

        # every creature object retains its final stats even after being culled
        return creatures, self._stats(creatures)

    def _stats(self, creatures: list) -> dict:
        foods = np.array([c.food_eaten for c in creatures], dtype=float)
        ages = np.array([c.age_ticks for c in creatures], dtype=float)
        fits = np.array([c.fitness() for c in creatures], dtype=float)
        survived = sum(1 for c in creatures if c.alive)
        stats = {
            "pop": len(creatures),
            "survivors": survived,
            "mean_food": float(foods.mean()),
            "max_food": float(foods.max()),
            "mean_age": float(ages.mean()),
            "mean_fitness": float(fits.mean()),
        }
        for t in TRACKED_TRAITS:
            vals = np.array([c.pheno[t] for c in creatures], dtype=float)
            stats[f"mean_{t}"] = float(vals.mean())
        return stats

    # ---- selection + breeding ----------------------------------------
    def breed(self, creatures: list) -> None:
        cfg = self.cfg
        # rank by fitness; the top fraction become the breeding pool
        ranked = sorted(creatures, key=lambda c: c.fitness(), reverse=True)
        n_survivors = max(2, int(len(ranked) * cfg.survivors_fraction))
        pool = ranked[:n_survivors]

        # elitism: carry the single best genome through unchanged so good
        # solutions are not lost to unlucky recombination
        next_genomes = [pool[0].genome.copy()]

        while len(next_genomes) < cfg.max_population:
            p1, p2 = self.rng.choice(len(pool), size=2, replace=len(pool) < 2)
            child = Genome.crossover(pool[p1].genome, pool[p2].genome,
                                     self.rng, cfg.crossover_rate)
            child.mutate(self.rng, cfg.p_point, cfg.p_struct)
            if cfg.lamarckian:
                # OFF by default: write learned weights back is NOT modelled here;
                # Lamarckism would require encoding the trained brain into genes.
                pass
            next_genomes.append(child)

        self.genomes = next_genomes

    # ---- full run -----------------------------------------------------
    def run(self, generations: int, csv_path: str = "stats.csv",
            verbose: bool = True) -> None:
        fieldnames = ["generation", "pop", "survivors", "mean_food", "max_food",
                      "mean_age", "mean_fitness"] + [f"mean_{t}" for t in TRACKED_TRAITS]
        with open(csv_path, "w", newline="") as f:
            writer = csvmod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for gen in range(generations):
                creatures, stats = self.run_generation()
                row = {"generation": gen, **stats}
                writer.writerow(row)
                f.flush()
                if verbose:
                    print(f"gen {gen:3d}  pop {stats['pop']:3d}  "
                          f"surv {stats['survivors']:3d}  "
                          f"food mean {stats['mean_food']:5.1f} max {stats['max_food']:4.0f}  "
                          f"speed {stats['mean_max_speed']:.2f}  "
                          f"metab {stats['mean_metabolic_rate']:.3f}  "
                          f"fit {stats['mean_fitness']:6.1f}")
                self.breed(creatures)
        if verbose:
            print(f"\nwrote per-generation stats to {csv_path}")


def run_evolution(cfg: Config, generations: int, csv_path: str = "stats.csv") -> None:
    Evolution(cfg).run(generations, csv_path)
