"""Diploid, evolvable genome.

A genome is two homologous chromosomes (diploid). Each chromosome is an ordered
list of `Gene`s aligned by locus, so crossover is a simple walk that switches
between the two homologs. Expression resolves the two alleles at each locus by a
dominance rule (complete dominance, codominant blend on ties) and emits a flat
phenotype dict that `creature.py` consumes.

Inheritance is Mendelian: gametes are formed by independent assortment plus
crossover, the child is the union of one gamete from each parent, and mutation
is applied afterwards. Learned brain weights are NEVER stored here -- evolution
operates on this genome between lives, plasticity operates on the brain within a
life, and the two are kept strictly separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Gene blueprint. Each entry defines one locus: its name (globally unique so the
# expressed phenotype is a flat dict), which functional group it belongs to, a
# default value, the legal range, the per-gene mutation sigma, and whether the
# value is conceptually an integer.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeneSpec:
    name: str
    group: str  # "brain" | "biochem" | "body"
    default: float
    vmin: float
    vmax: float
    sigma: float
    is_int: bool = False


GENE_SPECS: list[GeneSpec] = [
    # ---- brain ----
    GeneSpec("hidden_size", "brain", 10, 6, 18, 2.0, is_int=True),
    GeneSpec("learning_rate", "brain", 0.012, 0.003, 0.025, 0.004),
    GeneSpec("trace_decay", "brain", 0.98, 0.90, 0.995, 0.02),
    GeneSpec("w_scale", "brain", 0.6, 0.05, 2.0, 0.15),
    GeneSpec("w_bound", "brain", 3.0, 1.0, 4.0, 0.3),
    GeneSpec("wiring_seed", "brain", 12345, 0, 1_000_000, 50_000, is_int=True),
    GeneSpec("innate_food", "brain", 0.15, 0.0, 1.5, 0.2),
    GeneSpec("explore_noise", "brain", 0.4, 0.10, 0.6, 0.08),
    GeneSpec("gain_food", "brain", 1.0, 0.1, 3.0, 0.25),
    GeneSpec("gain_hazard", "brain", 1.0, 0.1, 3.0, 0.25),
    GeneSpec("gain_mate", "brain", 1.0, 0.1, 3.0, 0.25),
    GeneSpec("gain_drive", "brain", 1.0, 0.1, 3.0, 0.25),
    # ---- biochem ----
    GeneSpec("metabolic_rate", "biochem", 0.05, 0.01, 0.25, 0.02),
    GeneSpec("hunger_threshold", "biochem", 60.0, 20.0, 120.0, 8.0),
    GeneSpec("hunger_gain", "biochem", 0.05, 0.005, 0.2, 0.02),
    GeneSpec("reward_halflife", "biochem", 20.0, 4.0, 80.0, 8.0),
    GeneSpec("punish_halflife", "biochem", 20.0, 4.0, 80.0, 8.0),
    GeneSpec("pain_halflife", "biochem", 12.0, 4.0, 60.0, 6.0),
    GeneSpec("fatigue_rate", "biochem", 0.01, 0.0, 0.08, 0.01),
    GeneSpec("fatigue_halflife", "biochem", 40.0, 10.0, 120.0, 12.0),
    GeneSpec("eat_reward", "biochem", 1.2, 0.0, 3.0, 0.3),
    GeneSpec("damage_punish", "biochem", 1.0, 0.0, 3.0, 0.3),
    GeneSpec("aging_rate", "biochem", 1.0, 0.5, 2.0, 0.15),
    # ---- body ----
    GeneSpec("radius", "body", 8.0, 4.0, 16.0, 1.5),
    GeneSpec("max_speed", "body", 2.2, 0.8, 4.0, 0.4),
    GeneSpec("turn_rate", "body", 0.12, 0.04, 0.30, 0.03),
    GeneSpec("color_r", "body", 0.4, 0.0, 1.0, 0.12),
    GeneSpec("color_g", "body", 0.6, 0.0, 1.0, 0.12),
    GeneSpec("color_b", "body", 0.9, 0.0, 1.0, 0.12),
]

SPEC_BY_NAME: dict[str, GeneSpec] = {s.name: s for s in GENE_SPECS}


@dataclass
class Gene:
    """One allele at one locus."""

    name: str          # locus identifier (matches a GeneSpec.name for core genes)
    group: str
    value: float
    dominance: float   # [0,1]; higher dominance is expressed under conflict

    def copy(self) -> "Gene":
        return Gene(self.name, self.group, self.value, self.dominance)

    def to_dict(self) -> dict:
        return {"name": self.name, "group": self.group,
                "value": self.value, "dominance": self.dominance}

    @staticmethod
    def from_dict(d: dict) -> "Gene":
        return Gene(d["name"], d["group"], float(d["value"]), float(d["dominance"]))


def _clip_spec(spec: GeneSpec, value: float) -> float:
    value = float(np.clip(value, spec.vmin, spec.vmax))
    if spec.is_int:
        value = float(int(round(value)))
    return value


class Genome:
    """Two homologous chromosomes of aligned genes (diploid)."""

    def __init__(self, chrom_a: list[Gene], chrom_b: list[Gene]):
        self.chrom_a = chrom_a
        self.chrom_b = chrom_b

    # ---- construction -------------------------------------------------
    @staticmethod
    def random(rng: np.random.Generator) -> "Genome":
        """A fresh founder genome: each allele jittered around its default so the
        starting population has real genetic diversity to select on."""

        def make_chrom() -> list[Gene]:
            genes = []
            for spec in GENE_SPECS:
                jitter = rng.normal(0.0, spec.sigma)
                value = _clip_spec(spec, spec.default + jitter)
                dominance = float(rng.random())
                genes.append(Gene(spec.name, spec.group, value, dominance))
            return genes

        return Genome(make_chrom(), make_chrom())

    # ---- expression ---------------------------------------------------
    def express(self) -> dict[str, float]:
        """Resolve alleles to a flat phenotype dict.

        Complete dominance: the allele with higher dominance wins. On a near-tie
        the two values are blended (codominance), which gives smooth heritable
        variation while still producing Mendelian segregation across a
        population. Any core locus missing (possible after a structural deletion)
        falls back to its spec default.
        """
        # Index alleles by locus name. Structural duplicates land on novel locus
        # names (suffixed) and simply ride along without overriding core loci.
        a = {g.name: g for g in self.chrom_a}
        b = {g.name: g for g in self.chrom_b}

        pheno: dict[str, float] = {}
        for spec in GENE_SPECS:
            ga = a.get(spec.name)
            gb = b.get(spec.name)
            if ga is None and gb is None:
                pheno[spec.name] = spec.default
                continue
            if ga is None:
                value = gb.value
            elif gb is None:
                value = ga.value
            elif abs(ga.dominance - gb.dominance) < 0.1:
                value = 0.5 * (ga.value + gb.value)  # codominant blend
            else:
                value = ga.value if ga.dominance > gb.dominance else gb.value
            pheno[spec.name] = _clip_spec(spec, value)
        return pheno

    # ---- reproduction -------------------------------------------------
    def _gamete(self, rng: np.random.Generator, crossover_rate: float) -> list[Gene]:
        """Form a haploid gamete by walking the aligned chromosomes, starting on a
        random homolog (independent assortment) and switching homologs at
        crossover points."""
        # Align by locus order of chrom_a; pull the matching allele from whichever
        # homolog is "active". Loci unique to one homolog (post-duplication) are
        # carried through from that homolog.
        by_name_b = {g.name: g for g in self.chrom_b}
        active_a = bool(rng.random() < 0.5)
        gamete: list[Gene] = []
        for ga in self.chrom_a:
            if rng.random() < crossover_rate:
                active_a = not active_a
            gb = by_name_b.get(ga.name, ga)
            chosen = ga if active_a else gb
            gamete.append(chosen.copy())
        return gamete

    @staticmethod
    def crossover(p1: "Genome", p2: "Genome", rng: np.random.Generator,
                  crossover_rate: float) -> "Genome":
        """Sexual reproduction: one gamete from each parent forms the diploid child."""
        g1 = p1._gamete(rng, crossover_rate)
        g2 = p2._gamete(rng, crossover_rate)
        return Genome(g1, g2)

    def mutate(self, rng: np.random.Generator, p_point: float,
               p_struct: float) -> "Genome":
        """Point mutation (per-gene gaussian) plus rare structural mutation.

        Mutations are not pre-judged: selection sorts them out. Returns self for
        chaining; mutates in place."""
        for chrom in (self.chrom_a, self.chrom_b):
            for g in chrom:
                spec = SPEC_BY_NAME.get(g.name)
                if rng.random() < p_point:
                    sigma = spec.sigma if spec else max(1e-6, abs(g.value) * 0.1)
                    g.value = g.value + rng.normal(0.0, sigma)
                    if spec is not None:
                        g.value = _clip_spec(spec, g.value)
                if rng.random() < 0.5 * p_point:
                    g.dominance = float(np.clip(g.dominance + rng.normal(0, 0.1), 0, 1))

        if rng.random() < p_struct:
            self._structural_mutation(rng)
        return self

    def _structural_mutation(self, rng: np.random.Generator) -> None:
        """Gene duplication or deletion, applied symmetrically to both homologs so
        they stay aligned. Duplicates target a novel locus name so core
        expression is never corrupted; deletions only remove such extras."""
        extras_a = [i for i, g in enumerate(self.chrom_a)
                    if g.name not in SPEC_BY_NAME]
        if extras_a and rng.random() < 0.5:
            # deletion of an existing extra (by name, to keep homologs aligned)
            name = self.chrom_a[extras_a[int(rng.integers(len(extras_a)))]].name
            self.chrom_a = [g for g in self.chrom_a if g.name != name]
            self.chrom_b = [g for g in self.chrom_b if g.name != name]
        else:
            # duplication of a random core gene onto a fresh locus
            src = self.chrom_a[rng.integers(len(self.chrom_a))]
            new_name = f"{src.name}#dup{int(rng.integers(1_000_000))}"
            for chrom in (self.chrom_a, self.chrom_b):
                # find the matching source allele on this homolog
                base = next((g for g in chrom if g.name == src.name), src)
                clone = base.copy()
                clone.name = new_name
                spec = SPEC_BY_NAME.get(base.name)
                if spec is not None:
                    clone.value = _clip_spec(spec, clone.value + rng.normal(0, spec.sigma))
                chrom.append(clone)

    # ---- serialization ------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({
            "chrom_a": [g.to_dict() for g in self.chrom_a],
            "chrom_b": [g.to_dict() for g in self.chrom_b],
        })

    @staticmethod
    def from_json(s: str) -> "Genome":
        d = json.loads(s)
        return Genome(
            [Gene.from_dict(g) for g in d["chrom_a"]],
            [Gene.from_dict(g) for g in d["chrom_b"]],
        )

    def copy(self) -> "Genome":
        return Genome([g.copy() for g in self.chrom_a],
                      [g.copy() for g in self.chrom_b])
