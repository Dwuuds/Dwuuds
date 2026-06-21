"""The brain: a small feed-forward net with an online, local learning rule.

Architecture is a two-layer feed-forward net (inputs -> hidden -> outputs) whose
hidden width is set by the genome. Inputs are external sensors *plus* internal
drive signals from the biochemistry (hunger, pain, being-rewarded) -- feeding the
drives into cognition as inputs is exactly how Creatures worked.

Learning is NOT backprop. It is a three-factor Hebbian rule with eligibility
traces and a global neuromodulator:

    elig[i][j] = elig[i][j] * trace_decay + pre[i] * post[j]
    modulator  = reward_chem - punishment_chem          # can be negative
    w[i][j]   += learning_rate * modulator * elig[i][j]
    w[i][j]    = clip(w[i][j], w_min, w_max)

The eligibility trace remembers which synapses were co-active over the last
fraction of a second, so a reward that arrives *after* the useful action (eating
a moment after steering toward food) is still credited to the synapses that
caused it. Naive Hebbian cannot do this; that delayed-credit bridge is the whole
reason for the trace.

Exploration noise is added to the motor outputs and captured in the eligibility
trace. Combined with the reward modulator this makes the rule a reward-modulated
exploratory learner: lucky noisy actions that precede reward get reinforced. That
is what lets a creature learn to feed itself with no teacher at all, purely from
intrinsic reward.

Pure NumPy. The genome encodes *how* the brain learns (rate, trace decay, weight
bounds, initial wiring); the learned weights themselves are never inherited.
"""

from __future__ import annotations

import numpy as np


# Fixed sensor/motor layout. Hidden width varies per genome; the input and output
# widths are constant so brains share a comparable interface.
SENSOR_NAMES = [
    "food_fwd", "food_side", "food_prox",      # nearest food
    "hazard_fwd", "hazard_side", "hazard_prox",  # nearest hazard
    "mate_fwd", "mate_side", "mate_prox",      # nearest other creature
    "hunger", "pain", "reward",                # internal drives from biochem
    "bias",                                    # constant 1.0
]
N_SENSORS = len(SENSOR_NAMES)

MOTOR_NAMES = ["wheel_left", "wheel_right", "eat"]
N_MOTORS = len(MOTOR_NAMES)


class Brain:
    def __init__(self, pheno: dict, rng: np.random.Generator):
        self.lr = float(pheno["learning_rate"])
        self.trace_decay = float(pheno["trace_decay"])
        bound = float(pheno["w_bound"])
        self.w_min, self.w_max = -bound, bound
        self.explore_noise = float(pheno["explore_noise"])

        h = int(pheno["hidden_size"])
        self.h = h

        # Initial wiring is born from the genome: a deterministic seed makes
        # development reproducible, w_scale sets the random magnitude.
        init_rng = np.random.default_rng(int(pheno["wiring_seed"]))
        scale = float(pheno["w_scale"])
        self.w1 = init_rng.normal(0.0, scale, size=(h, N_SENSORS))      # hidden<-input
        self.w2 = init_rng.normal(0.0, scale, size=(N_MOTORS, h))       # motor<-hidden

        # Innate wiring bias so a creature is not born totally blank: a mild
        # predisposition for "food ahead -> drive forward" that learning can
        # amplify, suppress, or override. This is heritable (it's a gene); the
        # *learned* deltas on top of it are not.
        innate = float(pheno["innate_food"])
        i_fwd = SENSOR_NAMES.index("food_fwd")
        i_prox = SENSOR_NAMES.index("food_prox")
        # route innate food signal into the first hidden unit, then to both wheels
        self.w1[0, i_fwd] += innate
        self.w1[0, i_prox] += innate
        self.w2[MOTOR_NAMES.index("wheel_left"), 0] += innate
        self.w2[MOTOR_NAMES.index("wheel_right"), 0] += innate
        np.clip(self.w1, self.w_min, self.w_max, out=self.w1)
        np.clip(self.w2, self.w_min, self.w_max, out=self.w2)

        # Per-sensor input gains, grouped by modality (heritable receptor strength).
        self.gain = np.ones(N_SENSORS)
        for k in ("food_fwd", "food_side", "food_prox"):
            self.gain[SENSOR_NAMES.index(k)] = float(pheno["gain_food"])
        for k in ("hazard_fwd", "hazard_side", "hazard_prox"):
            self.gain[SENSOR_NAMES.index(k)] = float(pheno["gain_hazard"])
        for k in ("mate_fwd", "mate_side", "mate_prox"):
            self.gain[SENSOR_NAMES.index(k)] = float(pheno["gain_mate"])
        for k in ("hunger", "pain", "reward"):
            self.gain[SENSOR_NAMES.index(k)] = float(pheno["gain_drive"])

        # Eligibility traces, one per synapse.
        self.elig1 = np.zeros_like(self.w1)
        self.elig2 = np.zeros_like(self.w2)

        # Slow running averages of activity, used to center the eligibility trace
        # on activity *deviations* (exploration) rather than on whatever fires
        # constantly. This focuses credit assignment on the exploratory actions
        # that actually preceded reward and keeps the rule from reinforcing a
        # constant-output degenerate policy. It is a standard, well-understood
        # refinement of the bare three-factor rule.
        self._hidden_avg = np.zeros(h)
        self._out_avg = np.zeros(N_MOTORS)
        self._act_avg_rate = 0.02

        # Neuromodulator baseline (reward-prediction error). The intrinsic
        # appetitive reward is mostly positive -- you usually drift a little
        # closer to *some* food just by moving -- so the raw modulator would
        # inflate every co-active synapse to saturation. Subtracting a slow
        # running baseline makes only better-than-expected reward drive learning,
        # which is what makes the signal discriminative instead of a constant
        # push, and is what keeps the weights stable without any decay term.
        self._mod_baseline = 0.0
        self._baseline_rate = 0.0008

        self.rng = rng
        # cached activations from the last forward pass (used by learning)
        self._x = np.zeros(N_SENSORS)
        self._hidden = np.zeros(h)
        self._out = np.zeros(N_MOTORS)

    # ---- inference ----------------------------------------------------
    def forward(self, sensors: np.ndarray) -> np.ndarray:
        """Run the net and return motor activations in [-1, 1], with exploration
        noise added. The noisy outputs are what actually drive the body and what
        the eligibility trace records, so exploration is part of credit
        assignment."""
        x = sensors * self.gain
        hidden = np.tanh(self.w1 @ x)
        raw = self.w2 @ hidden
        if self.explore_noise > 0.0:
            raw = raw + self.rng.normal(0.0, self.explore_noise, size=N_MOTORS)
        out = np.tanh(raw)

        self._x = x
        self._hidden = hidden
        self._out = out
        return out

    # ---- learning -----------------------------------------------------
    def learn(self, modulator: float) -> None:
        """Apply one three-factor Hebbian update using the last forward pass.

        `modulator` is the global neuromodulator (reward - punishment) supplied by
        the biochemistry. The eligibility trace accumulates pre*post coincidences
        and decays, so even a delayed modulator reaches the synapses that earned
        it.
        """
        # update slow activity averages, then center the post activations so the
        # trace captures *deviations* (exploration) rather than constant drive
        self._hidden_avg += self._act_avg_rate * (self._hidden - self._hidden_avg)
        self._out_avg += self._act_avg_rate * (self._out - self._out_avg)
        hidden_dev = self._hidden - self._hidden_avg
        out_dev = self._out - self._out_avg

        # accumulate-and-decay eligibility for both layers
        self.elig1 *= self.trace_decay
        self.elig1 += np.outer(hidden_dev, self._x)      # post(hidden) x pre(input)
        self.elig2 *= self.trace_decay
        self.elig2 += np.outer(out_dev, self._hidden)    # post(motor) x pre(hidden)

        # reward-prediction error: only surprise relative to the running baseline
        # drives change (the appetitive signal is mostly positive otherwise)
        rpe = modulator - self._mod_baseline
        self._mod_baseline += self._baseline_rate * (modulator - self._mod_baseline)

        # three-factor update: the global neuromodulator (here the reward
        # prediction error) gates the local eligibility trace
        if self.lr > 0.0:
            self.w1 += self.lr * rpe * self.elig1
            self.w2 += self.lr * rpe * self.elig2
            np.clip(self.w1, self.w_min, self.w_max, out=self.w1)
            np.clip(self.w2, self.w_min, self.w_max, out=self.w2)

    def freeze(self) -> None:
        """Disable plasticity (used for untaught control creatures). Exploration
        noise is kept so the control is the *same* born stochastic policy with
        only learning switched off -- the cleanest possible comparison."""
        self.lr = 0.0
