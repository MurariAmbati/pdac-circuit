from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..core.paths import REGISTRY_JSON
from . import codon as _codon
from . import gc as _gc
from . import restriction as _restr
from . import splice as _splice
from . import structure as _struct
from .types import CircuitSeq, OptReport

def _load_prereg() -> dict:
    with open(REGISTRY_JSON, encoding="utf-8") as fh:
        reg=json.load(fh)
    return reg["prereg"]["module_IV"]

@dataclass
class OptConfig:

    gc_band: tuple[float, float] = (0.40, 0.60)
    cai_floor: float = 0.80
    structure_target_mfe: float = -10.0
    forbidden_enzymes: tuple[str, ...] = _restr.DEFAULT_ENZYMES
    donor_thr: float = _splice.DEFAULT_DONOR_THR
    acceptor_thr: float = _splice.DEFAULT_ACCEPTOR_THR
    gc_window: int = 50
    gc_step: int = 10
    structure_up: int = 30
    structure_down: int = 30
    max_homopolymer: int = 6
    sa_iters: int = 400
    sa_t0: float = 1.0
    sa_t1: float = 0.05
    seed: int = 20260620
    intended_sites: tuple[tuple[int, int], ...] = ()
    weights: dict = field(default_factory=lambda: _codon.HUMAN_WEIGHTS)

    @classmethod
    def from_registry(cls, **overrides) -> "OptConfig":
        pre=_load_prereg()
        cfg=cls(
            gc_band=tuple(pre.get("gc_band", (0.40, 0.60))),
            cai_floor=float(pre.get("cai_floor", 0.80)),
            structure_target_mfe=float(pre.get("structure_target_mfe", -10.0)),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

def _make_context_guard(cfg: OptConfig):
    enzymes=cfg.forbidden_enzymes

    def guard(left: str, codon: str, right: str, abs_pos: int) -> bool:
        local=(left + codon + right).upper()
        if _restr.scan_sites(local, enzymes):
            return False
        for h in _splice.scan_cryptic_sites(local, cfg.donor_thr, cfg.acceptor_thr):
            return False
        return True

    return guard

def _no_new_site(cs: CircuitSeq, cfg: OptConfig) -> bool:
    if _restr.scan_sites(cs.seq, cfg.forbidden_enzymes):
        return False
    if _splice.scan_cryptic_sites(cs.seq, cfg.donor_thr, cfg.acceptor_thr):
        return False
    return True

def _proteins(cs: CircuitSeq) -> dict[str, str]:
    return {f.name: cs.translate_cds(f) for f in cs.cds_features()}

def _max_homopolymer(seq: str) -> int:
    best=run = 1
    for i in range(1, len(seq)):
        run=run + 1 if seq[i] == seq[i - 1] else 1
        best=max(best, run)
    return best if seq else 0

def _energy(cs: CircuitSeq, cfg: OptConfig, start_pos: int) -> float:
    lo, hi = cfg.gc_band
    mid=0.5 * (lo + hi)
    e=0.0
    for _, g in _gc.windowed_gc(cs.seq, cfg.gc_window, cfg.gc_step):
        if g < lo:
            e += 10.0 * (lo - g)
        elif g > hi:
            e += 10.0 * (g - hi)
        else:
            e += 0.2 * abs(g - mid)
    for f in cs.cds_features():
        c=_codon.cai(cs.feature_subseq(f), cfg.weights, frame=f.frame or 0)
        e += 3.0 * (1.0 - c)
        if c < cfg.cai_floor:
            e += 60.0 * (cfg.cai_floor - c)
    m=_struct.mfe_window(cs.seq, start_pos, cfg.structure_up, cfg.structure_down)
    if m < cfg.structure_target_mfe:
        e += 0.5 * (cfg.structure_target_mfe - m)
    hp=_max_homopolymer(cs.seq)
    if hp > cfg.max_homopolymer:
        e += 1.0 * (hp - cfg.max_homopolymer)
    return e

def optimize_circuit(cs: CircuitSeq, cfg: OptConfig | None = None) -> tuple[CircuitSeq, OptReport]:
    if cfg is None:
        cfg=OptConfig.from_registry()
    rng=np.random.default_rng(cfg.seed)
    intended=[tuple(s) for s in cfg.intended_sites]

    report=OptReport()
    start_pos=_start_codon_pos(cs)

    report.gc_before=_gc.gc_content(cs.seq)
    report.cai_before={f.name: _codon.cai(cs.feature_subseq(f), cfg.weights, frame=f.frame or 0)
                         for f in cs.cds_features()}
    report.splice_sites_before=len(_splice.scan_cryptic_sites(cs.seq, cfg.donor_thr, cfg.acceptor_thr))
    report.mfe_window_before=_struct.mfe_window(cs.seq, start_pos, cfg.structure_up, cfg.structure_down)
    target_proteins=_proteins(cs)

    removed=_restr.remove_sites(cs, cfg.forbidden_enzymes, intended_sites=intended,
                                  forbidden_for_check=cfg.forbidden_enzymes)
    report.removed_restriction_sites=removed

    def splice_accept(ci, feat, cand):
        base=feat.start + (feat.frame or 0) + 3 * ci
        trial=cs.seq[:base] + cand + cs.seq[base + 3:]
        local_new=trial[max(0, base - 8):base + 11]
        return not _restr.scan_sites(local_new, cfg.forbidden_enzymes)

    _splice.remove_cryptic_sites(cs, cfg.donor_thr, cfg.acceptor_thr, accept=splice_accept)

    guard=_make_context_guard(cfg)
    for feat in cs.cds_features():
        if feat.locked:
            continue
        off=feat.frame or 0
        sub=cs.feature_subseq(feat)
        flank_left=cs.seq[max(0, feat.start - 2):feat.start]
        flank_right=cs.seq[feat.end:feat.end + 2]
        new_sub=_codon.optimize_codons(
            sub, cfg.weights, frame=off,
            gc_target=0.5 * (cfg.gc_band[0] + cfg.gc_band[1]),
            context_ok=guard, flank_left=flank_left, flank_right=flank_right,
        )
        ncodons=(feat.length - off) // 3
        for ci in range(ncodons):
            b=off + 3 * ci
            oldc=sub[b:b + 3]
            newc=new_sub[b:b + 3]
            if oldc != newc:
                cs.apply_codon(feat, ci, newc, reason="codon_optimize", constraint="cai")

    _restr.remove_sites(cs, cfg.forbidden_enzymes, intended_sites=intended,
                        forbidden_for_check=cfg.forbidden_enzymes)
    _splice.remove_cryptic_sites(cs, cfg.donor_thr, cfg.acceptor_thr, accept=splice_accept)

    def gc_accept(ci, feat, cand):
        base=feat.start + (feat.frame or 0) + 3 * ci
        trial=cs.seq[:base] + cand + cs.seq[base + 3:]
        local=trial[max(0, base - 20):base + 23]
        if _restr.scan_sites(local, cfg.forbidden_enzymes):
            return False
        if _splice.scan_cryptic_sites(local, cfg.donor_thr, cfg.acceptor_thr):
            return False
        sub=trial[feat.start:feat.end]
        if _codon.cai(sub, cfg.weights, frame=feat.frame or 0) < cfg.cai_floor:
            return False
        return True

    _gc.normalize_gc(cs, cfg.gc_band, win=cfg.gc_window, step=cfg.gc_step, accept=gc_accept)
    _restr.remove_sites(cs, cfg.forbidden_enzymes, intended_sites=intended,
                        forbidden_for_check=cfg.forbidden_enzymes)
    _splice.remove_cryptic_sites(cs, cfg.donor_thr, cfg.acceptor_thr, accept=splice_accept)

    _anneal(cs, cfg, rng, start_pos)

    _restr.remove_sites(cs, cfg.forbidden_enzymes, intended_sites=intended,
                        forbidden_for_check=cfg.forbidden_enzymes)
    _splice.remove_cryptic_sites(cs, cfg.donor_thr, cfg.acceptor_thr, accept=splice_accept)

    report.gc_after=_gc.gc_content(cs.seq)
    report.cai_after={f.name: _codon.cai(cs.feature_subseq(f), cfg.weights, frame=f.frame or 0)
                        for f in cs.cds_features()}
    report.splice_sites_after=len(_splice.scan_cryptic_sites(cs.seq, cfg.donor_thr, cfg.acceptor_thr))
    report.mfe_window_after=_struct.mfe_window(cs.seq, start_pos, cfg.structure_up, cfg.structure_down)
    report.n_edits=len(cs.edit_log)

    now_proteins=_proteins(cs)
    report.protein_preserved=now_proteins == target_proteins
    if not report.protein_preserved:
        report.unsatisfied.append("protein_changed: optimizer altered a CDS translation")

    remaining=_restr.scan_sites(cs.seq, cfg.forbidden_enzymes)
    remaining={k: v for k, v in remaining.items()}
    if remaining:
        report.unsatisfied.append(
            f"restriction_sites_remain: {sorted(remaining)} could not be abolished synonymously"
        )
    if report.splice_sites_after > 0:
        report.unsatisfied.append(
            f"cryptic_splice_remain: {report.splice_sites_after} site(s) above threshold"
        )

    in_band=all(cfg.gc_band[0] <= g <= cfg.gc_band[1]
                  for _, g in _gc.windowed_gc(cs.seq, cfg.gc_window, cfg.gc_step))
    report.gc_windows_in_band=in_band
    if not in_band:
        worst=max(_gc.windowed_gc(cs.seq, cfg.gc_window, cfg.gc_step),
                    key=lambda t: max(cfg.gc_band[0] - t[1], t[1] - cfg.gc_band[1]))
        report.unsatisfied.append(
            f"gc_out_of_band: window@{worst[0]} gc={worst[1]:.3f} outside {cfg.gc_band}; "
            "feasibility limited by amino-acid composition (no synonymous swap reaches the band)"
        )

    return cs, report

def _anneal(cs: CircuitSeq, cfg: OptConfig, rng: np.random.Generator, start_pos: int) -> None:
    intended=[tuple(s) for s in cfg.intended_sites]
    target_proteins=_proteins(cs)
    cur_e=_energy(cs, cfg, start_pos)
    coding=cs.coding_mask(intended)
    cds_feats=[f for f in cs.cds_features() if not f.locked]
    if not cds_feats:
        return

    for it in range(cfg.sa_iters):
        T=cfg.sa_t0 * (cfg.sa_t1 / cfg.sa_t0) ** (it / max(1, cfg.sa_iters - 1))
        feat=cds_feats[int(rng.integers(len(cds_feats)))]
        off=feat.frame or 0
        ncodons=(feat.length - off) // 3
        if ncodons == 0:
            continue
        ci=int(rng.integers(ncodons))
        base=feat.start + off + 3 * ci
        if any(base + k < len(coding) and not coding[base + k] for k in range(3)):
            continue
        codon=cs.seq[base:base + 3]
        aa=_codon.CODON_TABLE.get(codon)
        if aa is None or aa == "*":
            continue
        syns=[c for c in _codon.SYNONYMOUS.get(aa, []) if c != codon]
        if not syns:
            continue
        cand=syns[int(rng.integers(len(syns)))]

        snapshot=cs.seq
        log_len=len(cs.edit_log)
        cs.apply_codon(feat, ci, cand, reason="anneal", constraint="soft")

        if not _no_new_site(cs, cfg) or _proteins(cs) != target_proteins:
            cs.seq=snapshot
            del cs.edit_log[log_len:]
            continue

        new_e=_energy(cs, cfg, start_pos)
        dE=new_e - cur_e
        if dE <= 0 or rng.random() < np.exp(-dE / max(T, 1e-6)):
            cur_e=new_e
        else:
            cs.seq=snapshot
            del cs.edit_log[log_len:]

def _start_codon_pos(cs: CircuitSeq) -> int:
    cds=cs.cds_features()
    if cds:
        return cds[0].start + (cds[0].frame or 0)
    return 0
