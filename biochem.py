"""Creature biochemistry: a small set of chemical concentrations integrated each
tick with simple Euler steps.

Chemicals (all float concentrations):
  glucose      - energy store; metabolism burns it, food refills it, zero = death
  hunger       - drive that rises as glucose falls below a genome-set threshold
  pain         - rises on hazard contact, decays with a half-life
  reward       - neuromodulator (positive); teaching and intrinsic eating inject it
  punishment   - neuromodulator (negative side); teaching and damage inject it
  fatigue      - rises with movement effort, decays with rest
  age          - slow monotonic clock; past a threshold it raises death probability

hunger, pain and reward are fed to the brain as inputs. reward/punishment are
both the teaching channel (player keypress) and the intrinsic-learning channel
(eating while hungry -> reward, taking damage -> punishment). All rates,
thresholds, half-lives and emitter strengths come from the genome, so the
biochemistry is heritable and evolvable.
"""

from __future__ import annotations

import math

from config import Config


def halflife_to_decay(halflife: float) -> float:
    """Per-tick multiplicative decay that yields the given half-life in ticks."""
    if halflife <= 0:
        return 0.0
    return 0.5 ** (1.0 / halflife)


class Biochem:
    def __init__(self, pheno: dict, cfg: Config):
        self.cfg = cfg
        self.p = pheno

        self.glucose = cfg.glucose_start
        self.hunger = 0.0
        self.pain = 0.0
        self.reward = 0.0
        self.punishment = 0.0
        self.fatigue = 0.0
        self.age = 0.0

        # precompute decays from genome half-lives
        self.reward_decay = halflife_to_decay(pheno["reward_halflife"])
        self.punish_decay = halflife_to_decay(pheno["punish_halflife"])
        self.pain_decay = halflife_to_decay(pheno["pain_halflife"])
        self.fatigue_decay = halflife_to_decay(pheno["fatigue_halflife"])

    # ---- emitters (called by the creature/world) ----------------------
    def emit_reward(self, amount: float) -> None:
        self.reward = min(self.cfg.chem_max, self.reward + amount)

    def emit_punishment(self, amount: float) -> None:
        self.punishment = min(self.cfg.chem_max, self.punishment + amount)

    def emit_pain(self, amount: float) -> None:
        self.pain = min(self.cfg.chem_max, self.pain + amount)

    def add_glucose(self, amount: float) -> None:
        self.glucose = min(self.cfg.glucose_max, self.glucose + amount)

    def take_damage(self, amount: float) -> None:
        self.glucose = max(0.0, self.glucose - amount)
        self.emit_pain(amount * 0.2)
        # intrinsic punishment proportional to damage -> drives learning to avoid
        self.emit_punishment(self.p["damage_punish"] * 0.4)

    # ---- the per-tick update -----------------------------------------
    def step(self, effort: float, did_eat: bool, dt: float) -> None:
        """Integrate one tick.

        effort   - normalized movement effort in [0,1], drives metabolism+fatigue
        did_eat  - whether the creature successfully ate this tick
        """
        # metabolism: a base burn plus an effort/size dependent burn
        burn = self.p["metabolic_rate"] * (1.0 + 0.5 * self.fatigue + effort)
        self.glucose = max(0.0, self.glucose - burn * dt)

        # intrinsic reward: eating while hungry feels good and teaches feeding
        if did_eat:
            hunger_factor = min(1.0, self.hunger)
            self.emit_reward(self.p["eat_reward"] * (0.3 + 0.7 * hunger_factor))

        # hunger rises as glucose drops below threshold, relaxes otherwise
        deficit = self.p["hunger_threshold"] - self.glucose
        if deficit > 0:
            self.hunger += self.p["hunger_gain"] * (deficit / self.p["hunger_threshold"]) * dt
        else:
            self.hunger -= self.p["hunger_gain"] * dt
        self.hunger = float(min(1.5, max(0.0, self.hunger)))

        # fatigue accumulates with effort, decays toward zero
        self.fatigue += self.p["fatigue_rate"] * effort * dt
        self.fatigue *= self.fatigue_decay ** dt

        # neuromodulators and pain decay with their half-lives
        self.reward *= self.reward_decay ** dt
        self.punishment *= self.punish_decay ** dt
        self.pain *= self.pain_decay ** dt

        # aging clock
        self.age += self.p["aging_rate"] * dt

    # ---- readouts -----------------------------------------------------
    @property
    def modulator(self) -> float:
        """Global neuromodulator handed to the brain's learning rule."""
        return self.reward - self.punishment

    def is_dead_of_starvation(self) -> bool:
        return self.glucose <= 0.0

    def death_probability(self) -> float:
        """Per-tick probability of death from old age, ramping past the soft cap."""
        if self.age < self.cfg.max_age_ticks:
            return 0.0
        over = (self.age - self.cfg.max_age_ticks) / self.cfg.max_age_ticks
        return min(1.0, 0.0005 + 0.01 * over)
