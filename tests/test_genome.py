"""Unit tests for genome crossover, mutation, expression and serialization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genome import Genome, Gene, GENE_SPECS, SPEC_BY_NAME


def test_random_genome_is_diploid_and_complete():
    rng = np.random.default_rng(0)
    g = Genome.random(rng)
    assert len(g.chrom_a) == len(GENE_SPECS)
    assert len(g.chrom_b) == len(GENE_SPECS)
    # every core locus present on both homologs
    names_a = {gene.name for gene in g.chrom_a}
    names_b = {gene.name for gene in g.chrom_b}
    for spec in GENE_SPECS:
        assert spec.name in names_a and spec.name in names_b


def test_expression_is_flat_and_in_bounds():
    rng = np.random.default_rng(1)
    pheno = Genome.random(rng).express()
    for spec in GENE_SPECS:
        assert spec.name in pheno
        assert spec.vmin - 1e-9 <= pheno[spec.name] <= spec.vmax + 1e-9
    # integer-typed genes express as whole numbers
    assert float(pheno["hidden_size"]).is_integer()


def test_dominance_resolution():
    # build a genome where one allele clearly dominates and check it wins
    a = [Gene("metabolic_rate", "biochem", 0.10, dominance=0.9)]
    b = [Gene("metabolic_rate", "biochem", 0.20, dominance=0.1)]
    g = Genome(a, b)
    assert g.express()["metabolic_rate"] == pytest.approx(0.10)
    # codominant (near-equal dominance) blends
    a2 = [Gene("metabolic_rate", "biochem", 0.10, dominance=0.50)]
    b2 = [Gene("metabolic_rate", "biochem", 0.20, dominance=0.52)]
    assert Genome(a2, b2).express()["metabolic_rate"] == pytest.approx(0.15)


def test_crossover_child_is_diploid_from_both_parents():
    rng = np.random.default_rng(2)
    p1 = Genome.random(rng)
    p2 = Genome.random(rng)
    child = Genome.crossover(p1, p2, rng, crossover_rate=0.1)
    assert len(child.chrom_a) == len(p1.chrom_a)  # gamete from p1
    assert len(child.chrom_b) == len(p2.chrom_a)  # gamete from p2
    # each child allele value must come from one of the parents at that locus
    p1_vals = {(gene.name): {gene.value for gene in p1.chrom_a} |
               {gene.value for gene in p1.chrom_b} for gene in p1.chrom_a}
    for gene in child.chrom_a:
        assert gene.value in p1_vals[gene.name]


def test_crossover_recombines():
    # With a nonzero crossover rate the gamete should mix both homologs across many
    # loci (independent assortment + crossover), not just copy one homolog.
    rng = np.random.default_rng(3)
    # parent with distinguishable homologs: chrom_a all dominance 0.0, b all 1.0
    a = [Gene(s.name, s.group, s.default, 0.0) for s in GENE_SPECS]
    b = [Gene(s.name, s.group, s.default, 1.0) for s in GENE_SPECS]
    parent = Genome(a, b)
    saw_a = saw_b = False
    for _ in range(50):
        gam = parent._gamete(rng, crossover_rate=0.3)
        doms = [gene.dominance for gene in gam]
        saw_a |= any(d == 0.0 for d in doms)
        saw_b |= any(d == 1.0 for d in doms)
    assert saw_a and saw_b  # gametes draw from both homologs


def test_point_mutation_changes_values_within_bounds():
    rng = np.random.default_rng(4)
    g = Genome.random(rng)
    before = [gene.value for gene in g.chrom_a]
    # force heavy mutation
    g.mutate(rng, p_point=1.0, p_struct=0.0)
    after = [gene.value for gene in g.chrom_a]
    assert any(b != a for b, a in zip(before, after))  # something changed
    for gene in g.chrom_a:
        spec = SPEC_BY_NAME.get(gene.name)
        if spec:
            assert spec.vmin - 1e-9 <= gene.value <= spec.vmax + 1e-9


def test_structural_mutation_keeps_homologs_aligned():
    rng = np.random.default_rng(5)
    g = Genome.random(rng)
    for _ in range(40):
        g._structural_mutation(rng)
    # duplication/deletion is applied symmetrically -> homologs stay same length
    assert len(g.chrom_a) == len(g.chrom_b)
    # core expression still works and stays complete
    pheno = g.express()
    for spec in GENE_SPECS:
        assert spec.name in pheno


def test_json_roundtrip():
    rng = np.random.default_rng(6)
    g = Genome.random(rng)
    g2 = Genome.from_json(g.to_json())
    assert g.express() == g2.express()
