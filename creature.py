"""A creature: genome -> phenotype, bound to a brain, a biochemistry and a body.

Owns its sensors (what it perceives of the world), its motors (differential drive
plus an eat action), and its life state (energy, age, death). The world calls
`sense`, `think`, and `act` each tick and handles food/hazard interactions.

The brain is born from the genome and then goes plastic during life; the genome
is the heritable part, the learned weights are not.
"""

from __future__ import annotations

import math

import numpy as np

from config import Config
from genome import Genome
from brain import Brain, N_SENSORS, SENSOR_NAMES, MOTOR_NAMES
from biochem import Biochem

# sensing range over which proximity falls off
SENSE_RANGE = 250.0


class Creature:
    _next_id = 0

    def __init__(self, genome: Genome, cfg: Config, rng: np.random.Generator,
                 x: float, y: float, heading: float | None = None,
                 generation: int = 0):
        self.id = Creature._next_id
        Creature._next_id += 1

        self.genome = genome
        self.cfg = cfg
        self.rng = rng
        self.pheno = genome.express()

        self.brain = Brain(self.pheno, rng)
        self.biochem = Biochem(self.pheno, cfg)

        # body
        self.radius = float(self.pheno["radius"])
        self.max_speed = float(self.pheno["max_speed"])
        self.turn_rate = float(self.pheno["turn_rate"])
        self.color = (
            int(255 * self.pheno["color_r"]),
            int(255 * self.pheno["color_g"]),
            int(255 * self.pheno["color_b"]),
        )

        self.x = x
        self.y = y
        self.heading = heading if heading is not None else float(rng.uniform(0, 2 * math.pi))
        self.vx = 0.0
        self.vy = 0.0

        self.alive = True
        self.generation = generation
        self.age_ticks = 0
        self.food_eaten = 0          # lifetime food count (fitness signal)
        self.distance_travelled = 0.0
        self._sensors = np.zeros(N_SENSORS)
        self._last_eat_output = 0.0

    # ---- perception ---------------------------------------------------
    def _relative(self, tx: float, ty: float):
        """Return (forward, side, proximity) of a target relative to heading.
        forward/side are unit-ish components in the creature's frame; proximity
        falls off with distance."""
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0, 1.0
        # rotate world delta into creature frame
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        fwd = (dx * ch + dy * sh) / dist     # +1 straight ahead
        side = (-dx * sh + dy * ch) / dist    # +1 to the left
        prox = max(0.0, 1.0 - dist / SENSE_RANGE)
        return fwd, side, prox

    def sense(self, nearest_food, nearest_hazard, nearest_mate) -> np.ndarray:
        """Build the sensor vector from nearest world objects plus internal drives.
        Each nearest_* is an (x, y) tuple or None."""
        s = np.zeros(N_SENSORS)

        def fill(prefix, target):
            if target is None:
                return
            fwd, side, prox = self._relative(target[0], target[1])
            s[SENSOR_NAMES.index(f"{prefix}_fwd")] = fwd * prox
            s[SENSOR_NAMES.index(f"{prefix}_side")] = side * prox
            s[SENSOR_NAMES.index(f"{prefix}_prox")] = prox

        fill("food", nearest_food)
        fill("hazard", nearest_hazard)
        fill("mate", nearest_mate)

        s[SENSOR_NAMES.index("hunger")] = self.biochem.hunger
        s[SENSOR_NAMES.index("pain")] = min(1.0, self.biochem.pain)
        s[SENSOR_NAMES.index("reward")] = min(1.0, self.biochem.reward)
        s[SENSOR_NAMES.index("bias")] = 1.0

        self._sensors = s
        return s

    # ---- cognition + action ------------------------------------------
    def think(self) -> np.ndarray:
        return self.brain.forward(self._sensors)

    def act(self, motors: np.ndarray, dt: float) -> bool:
        """Apply motor outputs to the body. Returns whether the creature is trying
        to eat this tick (eat output above threshold)."""
        left = float(motors[MOTOR_NAMES.index("wheel_left")])
        right = float(motors[MOTOR_NAMES.index("wheel_right")])
        eat_out = float(motors[MOTOR_NAMES.index("eat")])
        self._last_eat_output = eat_out

        # differential drive: average -> forward speed, difference -> turning.
        # A small innate base drive keeps the creature wandering so it can find
        # food and bootstrap learning; steering is the learned part.
        drive = 0.5 * (left + right) + self.cfg.base_drive
        forward = drive * self.max_speed
        turn = 0.5 * (right - left) * self.turn_rate * math.pi
        self.heading = (self.heading + turn * dt) % (2 * math.pi)

        self.vx = math.cos(self.heading) * forward
        self.vy = math.sin(self.heading) * forward
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.distance_travelled += abs(forward) * dt

        # toroidal world wrap keeps creatures in play
        self.x %= self.cfg.world_width
        self.y %= self.cfg.world_height

        effort = min(1.0, (abs(left) + abs(right)) * 0.5)
        self._effort = effort
        return eat_out > 0.0

    # ---- metabolism / learning / death -------------------------------
    def metabolize(self, did_eat: bool, dt: float) -> None:
        effort = getattr(self, "_effort", 0.0)
        self.biochem.step(effort, did_eat, dt)
        # within-life plasticity: one local Hebbian update modulated by biochem
        self.brain.learn(self.biochem.modulator)

        self.age_ticks += 1
        if self.biochem.is_dead_of_starvation():
            self.alive = False
        elif self.rng.random() < self.biochem.death_probability():
            self.alive = False

    def on_eat(self) -> None:
        """Called by the world when this creature successfully eats a pellet."""
        self.biochem.add_glucose(self.cfg.food_energy)
        self.food_eaten += 1

    def reward(self, amount: float) -> None:
        self.biochem.emit_reward(amount)

    def punish(self, amount: float) -> None:
        self.biochem.emit_punishment(amount)

    # ---- fitness / reproduction --------------------------------------
    @property
    def position(self):
        return (self.x, self.y)

    def fitness(self) -> float:
        """A simple lifetime fitness proxy used for selection in evolve mode."""
        return self.food_eaten * 10.0 + self.age_ticks * 0.01
