from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Mapping

@dataclass(frozen=True)
class SplitPolicy:

    validation_chromosomes: tuple[str, ...]=("chr6", "chr7")
    test_chromosomes: tuple[str, ...]=("chr8", "chr9")
    held_out_sample_groups: frozenset[str]=field(default_factory=frozenset)
    external_studies: frozenset[str]=field(default_factory=frozenset)
    temporal_cutoff: str="2026-06-20"

    def validate(self) -> None:
        val, test=set(self.validation_chromosomes), set(self.test_chromosomes)
        if val & test:
            raise ValueError(f"validation/test chromosome overlap: {sorted(val & test)}")
        date.fromisoformat(self.temporal_cutoff)

def assign_split(record: Mapping, policy: SplitPolicy) -> str:

    policy.validate()
    chrom=str(record["chrom"])
    sample_group=str(record.get("sample_group", ""))
    study=str(record.get("study", ""))
    released=str(record.get("released", ""))
    state_held_out=sample_group in policy.held_out_sample_groups
    locus_held_out=chrom in policy.test_chromosomes

    if study in policy.external_studies:
        return "external_study_test"
    if released and released > policy.temporal_cutoff:
        return "temporal_test"
    if state_held_out and locus_held_out:
        return "joint_locus_state_test"
    if state_held_out:
        return "state_test"
    if locus_held_out:
        return "locus_test"
    if chrom in policy.validation_chromosomes:
        return "validation"
    return "train"

def interval_key(record: Mapping) -> tuple[str, int, int]:
    return (str(record["chrom"]), int(record["start"]), int(record["end"]))

def audit_split_records(records: Iterable[Mapping], policy: SplitPolicy) -> dict:

    by_split: dict[str, list[Mapping]]={}
    for record in records:
        split=assign_split(record, policy)
        by_split.setdefault(split, []).append(record)

    failures: list[str]=[]
    train_groups={str(r.get("sample_group", "")) for r in by_split.get("train", [])}
    state_groups={
        str(r.get("sample_group", ""))
        for name in ("state_test", "joint_locus_state_test")
        for r in by_split.get(name, [])
    }
    overlap_groups=(train_groups & state_groups) - {""}
    if overlap_groups:
        failures.append(f"held-out sample groups leaked into train: {sorted(overlap_groups)}")

    train_intervals={interval_key(r) for r in by_split.get("train", [])}
    for name in ("locus_test", "joint_locus_state_test"):
        overlap=train_intervals & {interval_key(r) for r in by_split.get(name, [])}
        if overlap:
            failures.append(f"{name} shares {len(overlap)} exact windows with train")

    train_studies={str(r.get("study", "")) for r in by_split.get("train", [])}
    external={
        str(r.get("study", "")) for r in by_split.get("external_study_test", [])
    }
    overlap_studies=(train_studies & external) - {""}
    if overlap_studies:
        failures.append(f"external studies leaked into train: {sorted(overlap_studies)}")

    return {
        "ok": not failures,
        "failures": failures,
        "counts": {name: len(items) for name, items in sorted(by_split.items())},
        "sample_groups": {
            name: len({str(r.get("sample_group", "")) for r in items} - {""})
            for name, items in sorted(by_split.items())
        },
    }
