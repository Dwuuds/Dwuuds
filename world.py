"""The 2D world: food, hazards, creatures, and the simulation step.

No rendering happens here. The world owns the spatial state and advances it one
tick at a time so the exact same core can run headless-and-fast for evolution or
real-time under the pygame viewer.
"""

from __future__ import annotations

import math

import numpy as np

from config import Config
from genome import Genome
from creature import Creature


class World:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.tick = 0

        self.creatures: list[Creature] = []
        self.food: list[list[float]] = []      # [x, y]
        self.hazards: list[list[float]] = []   # [x, y]

        for _ in range(cfg.n_food):
            self._spawn_food()
        for _ in range(cfg.n_hazards):
            self.hazards.append(self._random_point(margin=cfg.hazard_radius * 2))

    # ---- spawning -----------------------------------------------------
    def _random_point(self, margin: float = 10.0) -> list[float]:
        return [
            float(self.rng.uniform(margin, self.cfg.world_width - margin)),
            float(self.rng.uniform(margin, self.cfg.world_height - margin)),
        ]

    def _spawn_food(self) -> None:
        self.food.append(self._random_point(self.cfg.food_radius))

    def add_creature(self, genome: Genome, x=None, y=None, generation=0) -> Creature:
        if x is None:
            x, y = self._random_point()
        c = Creature(genome, self.cfg, self.rng, x, y, generation=generation)
        self.creatures.append(c)
        return c

    def populate_random(self, n: int) -> None:
        for _ in range(n):
            self.add_creature(Genome.random(self.rng))

    # ---- nearest-object queries --------------------------------------
    def _nearest(self, x, y, points, exclude_idx=None):
        best = None
        best_d2 = float("inf")
        for i, p in enumerate(points):
            if exclude_idx is not None and i == exclude_idx:
                continue
            dx = p[0] - x
            dy = p[1] - y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = (p[0], p[1])
        return best

    def _nearest_creature(self, c: Creature):
        best = None
        best_d2 = float("inf")
        for other in self.creatures:
            if other is c or not other.alive:
                continue
            dx = other.x - c.x
            dy = other.y - c.y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = (other.x, other.y)
        return best

    # ---- the sim step -------------------------------------------------
    def step(self) -> None:
        cfg = self.cfg
        dt = cfg.dt

        for c in self.creatures:
            if not c.alive:
                continue

            nf = self._nearest(c.x, c.y, self.food)
            nh = self._nearest(c.x, c.y, self.hazards)
            nm = self._nearest_creature(c)

            c.sense(nf, nh, nm)
            motors = c.think()
            trying_to_eat = c.act(motors, dt)

            did_eat = self._handle_food(c, trying_to_eat)
            self._handle_hazards(c, dt)
            self._appetitive_shaping(c, nf, nh)
            c.metabolize(did_eat, dt)

        # cull the dead
        self.creatures = [c for c in self.creatures if c.alive]

        # keep the food supply topped up
        if cfg.food_respawn:
            while len(self.food) < cfg.n_food:
                self._spawn_food()

        self.tick += 1

    def _handle_food(self, c: Creature, trying_to_eat: bool) -> bool:
        """Eat the nearest pellet if the creature overlaps it and wants to eat."""
        if not trying_to_eat:
            return False
        eat_reach = c.radius + self.cfg.food_radius
        for i, p in enumerate(self.food):
            dx = p[0] - c.x
            dy = p[1] - c.y
            if dx * dx + dy * dy <= eat_reach * eat_reach:
                c.on_eat()
                self.food.pop(i)
                if not self.cfg.food_respawn:
                    pass  # consumed; finite-food pressure in evolve mode
                return True
        return False

    def _appetitive_shaping(self, c: Creature, nf, nh) -> None:
        """Dense, signed intrinsic learning signal. Closing on food (when hungry)
        emits a little reward; drifting away emits a little punishment; closing on
        a hazard emits punishment. This is what makes feeding learnable from
        scratch -- the rare event of eating is too sparse on its own."""
        cfg = self.cfg
        if cfg.appetite_reward > 0.0 and nf is not None:
            d = math.hypot(nf[0] - c.x, nf[1] - c.y)
            last = getattr(c, "_last_food_dist", d)
            c._last_food_dist = d
            approach = last - d  # +ve = got closer
            drive = max(cfg.appetite_floor, min(1.0, c.biochem.hunger))
            if approach >= 0:
                c.biochem.emit_reward(approach * cfg.appetite_reward * drive)
            else:
                c.biochem.emit_punishment(-approach * cfg.appetite_reward * drive)
        if cfg.hazard_aversion > 0.0 and nh is not None:
            d = math.hypot(nh[0] - c.x, nh[1] - c.y)
            last = getattr(c, "_last_hazard_dist", d)
            c._last_hazard_dist = d
            closing = last - d
            if closing > 0 and d < 150.0:
                c.biochem.emit_punishment(closing * cfg.hazard_aversion)

    def _handle_hazards(self, c: Creature, dt: float) -> None:
        for h in self.hazards:
            dx = h[0] - c.x
            dy = h[1] - c.y
            reach = c.radius + self.cfg.hazard_radius
            if dx * dx + dy * dy <= reach * reach:
                c.biochem.take_damage(self.cfg.hazard_damage * dt)

    # ---- helpers ------------------------------------------------------
    def creature_at(self, x: float, y: float):
        """Topmost living creature whose body contains (x, y), or None."""
        for c in reversed(self.creatures):
            if not c.alive:
                continue
            if (c.x - x) ** 2 + (c.y - y) ** 2 <= c.radius ** 2:
                return c
        return None

    @property
    def n_alive(self) -> int:
        return sum(1 for c in self.creatures if c.alive)
