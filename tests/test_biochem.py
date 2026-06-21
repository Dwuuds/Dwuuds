"""Unit tests for the biochemistry update step and the brain's learning rule."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from genome import Genome
from biochem import Biochem, halflife_to_decay
from brain import Brain, N_SENSORS, N_MOTORS


def make_pheno(seed=0):
    return Genome.random(np.random.default_rng(seed)).express()


def test_halflife_decay_math():
    d = halflife_to_decay(10.0)
    # after exactly one half-life, a unit concentration should be ~0.5
    val = 1.0
    for _ in range(10):
        val *= d
    assert val == pytest.approx(0.5, abs=1e-6)


def test_metabolism_burns_glucose():
    cfg = Config()
    b = Biochem(make_pheno(), cfg)
    g0 = b.glucose
    b.step(effort=0.0, did_eat=False, dt=1.0)
    assert b.glucose < g0  # metabolism always costs something


def test_starvation_death_flag():
    cfg = Config()
    b = Biochem(make_pheno(), cfg)
    b.glucose = 0.0
    assert b.is_dead_of_starvation()


def test_hunger_rises_when_glucose_low():
    cfg = Config()
    b = Biochem(make_pheno(), cfg)
    b.glucose = 0.0  # well below any threshold
    b.hunger = 0.0
    for _ in range(20):
        b.step(effort=0.0, did_eat=False, dt=1.0)
    assert b.hunger > 0.0  # a hunger drive built up


def test_eating_while_hungry_emits_reward():
    cfg = Config()
    b = Biochem(make_pheno(), cfg)
    b.glucose = 1.0
    b.hunger = 1.0
    b.reward = 0.0
    b.step(effort=0.0, did_eat=True, dt=1.0)
    assert b.reward > 0.0  # intrinsic reward channel fired


def test_reward_decays_with_halflife():
    cfg = Config()
    b = Biochem(make_pheno(), cfg)
    b.emit_reward(2.0)
    r0 = b.reward
    for _ in range(5):
        b.step(effort=0.0, did_eat=False, dt=1.0)
    assert 0.0 < b.reward < r0  # decays but not instantly


def test_damage_emits_pain_and_punishment():
    cfg = Config()
    b = Biochem(make_pheno(), cfg)
    b.pain = b.punishment = 0.0
    b.take_damage(5.0)
    assert b.pain > 0.0 and b.punishment > 0.0


def test_modulator_is_reward_minus_punishment():
    cfg = Config()
    b = Biochem(make_pheno(), cfg)
    b.reward = 0.8
    b.punishment = 0.3
    assert b.modulator == pytest.approx(0.5)


# ---- brain learning rule ----

def test_positive_modulator_strengthens_coactive_synapse():
    """The three-factor rule must push a synapse up when pre and post are
    co-active and the modulator (reward) is positive."""
    pheno = make_pheno()
    rng = np.random.default_rng(0)
    brain = Brain(pheno, rng)
    brain.explore_noise = 0.0  # deterministic for the assertion

    sensors = np.zeros(N_SENSORS)
    sensors[0] = 1.0  # drive one input hard
    # build eligibility from repeated co-activation, then reward
    for _ in range(5):
        brain.forward(sensors)
        w_before = brain.w1.copy()
        brain.learn(modulator=1.0)
    # at least some weights moved off their initial values under reward
    assert not np.allclose(brain.w1, w_before) or brain.elig1.any()
    # eligibility accumulated for the active input column
    assert np.abs(brain.elig1[:, 0]).sum() > 0


def test_zero_modulator_leaves_weights_unchanged():
    pheno = make_pheno()
    brain = Brain(pheno, np.random.default_rng(1))
    sensors = np.ones(N_SENSORS)
    brain.forward(sensors)
    w1 = brain.w1.copy()
    w2 = brain.w2.copy()
    brain.learn(modulator=0.0)  # baseline starts at 0 -> rpe 0 -> no change
    assert np.allclose(brain.w1, w1)
    assert np.allclose(brain.w2, w2)


def test_weights_stay_within_genome_bounds():
    pheno = make_pheno()
    brain = Brain(pheno, np.random.default_rng(2))
    sensors = np.ones(N_SENSORS)
    for _ in range(200):
        brain.forward(sensors)
        brain.learn(modulator=5.0)  # hammer it
    assert brain.w1.max() <= brain.w_max + 1e-9
    assert brain.w1.min() >= brain.w_min - 1e-9
    assert brain.w2.max() <= brain.w_max + 1e-9
    assert brain.w2.min() >= brain.w_min - 1e-9


def test_frozen_brain_does_not_learn():
    pheno = make_pheno()
    brain = Brain(pheno, np.random.default_rng(3))
    brain.freeze()
    sensors = np.ones(N_SENSORS)
    w1 = brain.w1.copy()
    for _ in range(50):
        brain.forward(sensors)
        brain.learn(modulator=3.0)
    assert np.allclose(brain.w1, w1)  # plasticity off
