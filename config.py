"""Central configuration for the A-Life creature sim PoC.

Every tunable knob lives here. No magic numbers scattered through the code.
The simulation core reads from a single `Config` instance so that interactive
and headless modes share identical physics and only differ in presentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- World ----------------------------------------------------------
    world_width: float = 800.0
    world_height: float = 600.0
    dt: float = 1.0  # simulation timestep in "ticks"; physics scaled by this
    tick_rate: int = 60  # target ticks/sec for the interactive viewer

    n_food: int = 60  # food pellets kept alive in the world
    food_energy: float = 40.0  # glucose released when a pellet is eaten
    food_radius: float = 5.0
    food_respawn: bool = True  # keep food count topped up (interactive/learning)

    n_hazards: int = 6
    hazard_radius: float = 16.0
    hazard_damage: float = 6.0  # glucose lost per tick touching a hazard

    # ---- Population -----------------------------------------------------
    start_population: int = 20
    max_population: int = 60

    # ---- Brain ----------------------------------------------------------
    # The brain is a small feed-forward net. These are the *bounds*; the actual
    # per-creature values come from the genome and are clipped into these ranges.
    hidden_min: int = 6
    hidden_max: int = 18
    learning_rate_max: float = 0.025
    trace_decay_min: float = 0.90
    trace_decay_max: float = 0.995
    weight_bound_max: float = 4.0
    explore_noise_max: float = 0.6
    # A small innate forward crawl so creatures always wander and can stumble onto
    # food to bootstrap intrinsic-reward learning. Steering on top is learned.
    base_drive: float = 0.3
    # If True, learned weights are written back into the genome at reproduction
    # (Lamarckism). OFF by default and only for experiments; the PoC's whole
    # point is that plasticity (within-life) and evolution (between-life) are
    # separate systems.
    lamarckian: bool = False

    # ---- Biochemistry ---------------------------------------------------
    glucose_start: float = 100.0
    glucose_max: float = 200.0
    # Reward/punishment are clamped so a single teaching keypress is a nudge,
    # not a sledgehammer.
    chem_max: float = 4.0
    reward_key_dose: float = 1.5  # injected per frame the R key is held
    punish_key_dose: float = 1.5  # injected per frame the P key is held
    # Intrinsic appetitive drive: closing distance to the nearest food emits a
    # little reward, opening it emits a little punishment, scaled by hunger. This
    # is the dense, signed intrinsic signal that lets a creature learn to feed
    # itself with no teacher. Set to 0.0 to fall back to eating-only reward.
    appetite_reward: float = 0.08
    appetite_floor: float = 0.25  # food stays a bit appetitive even when full
    hazard_aversion: float = 0.06  # closing on a hazard emits punishment

    # ---- Aging / death --------------------------------------------------
    max_age_ticks: int = 6000  # soft cap; death probability ramps past threshold

    # ---- Genetics -------------------------------------------------------
    p_point: float = 0.15  # per-gene gaussian mutation probability
    p_struct: float = 0.01  # structural mutation (gene dup/deletion) probability
    crossover_rate: float = 0.04  # per-locus probability of switching homolog

    # ---- Evolution run --------------------------------------------------
    generations: int = 100
    ticks_per_generation: int = 4000
    survivors_fraction: float = 0.4  # top fraction that get to reproduce

    # ---- Reproducibility ------------------------------------------------
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
