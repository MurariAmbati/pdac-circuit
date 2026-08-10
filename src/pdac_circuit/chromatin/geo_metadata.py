from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import unquote, urlparse

from .encode import assay_vector
from .geo import GEO_FTP, _registered_asset, geo_series_bucket
from .geo_archive import BIGWIG_RE
from .streaming import (
    TrackSpec,
    _assert_bigwig_native_genome,
    _open_bigwig,
    sha256_file,
)

SOFT_SCHEMA="pdac-circuit.geo-soft-metadata/1"
TRACK_INDEX_SCHEMA="pdac-circuit.geo-track-specs/1"

def geo_soft_url(accession: str) -> str:
    accession=accession.upper()
    return (
        f"{GEO_FTP}/{geo_series_bucket(accession)}/{accession}/soft/"
        f"{accession}_family.soft.gz"
    )

def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session=requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=5,
                connect=5,
                read=5,
                backoff_factor=0.75,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
            )
        ),
    )
    return session

def _append(record: dict[str, list[str]], key: str, value: str) -> None:
    record.setdefault(key, []).append(value.strip())

def parse_geo_soft(text: str) -> dict[str, dict]:

    raw_samples: dict[str, dict[str, list[str]]]={}
    current: dict[str, list[str]] | None=None
    accession: str | None=None
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            accession=line.split("=", 1)[1].strip()
            if not re.fullmatch(r"GSM\d+", accession):
                raise ValueError(f"invalid GEO sample accession {accession!r}")
            current=raw_samples.setdefault(accession, {})
            continue
        if line.startswith("^"):
            current=None
            accession=None
            continue
        if current is None or not line.startswith("!Sample_") or " = " not in line:
            continue
        key, value=line[1:].split(" = ", 1)
        _append(current, key.removeprefix("Sample_"), value)

    samples: dict[str, dict]={}
    for gsm, fields in raw_samples.items():
        characteristics: dict[str, list[str]]={}
        unkeyed_characteristics: list[str]=[]
        for value in fields.get("characteristics_ch1", []):
            if ":" in value:
                key, item=value.split(":", 1)
                characteristics.setdefault(key.strip().lower(), []).append(item.strip())
            else:
                unkeyed_characteristics.append(value)

        def first(name: str, default: str = "", fields=fields) -> str:
            values=fields.get(name, [])
            return values[0] if values else default

        samples[gsm]={
            "gsm": gsm,
            "title": first("title"),
            "source_name": first("source_name_ch1"),
            "organism": first("organism_ch1"),
            "status": first("status"),
            "release_date": _public_date(first("status")),
            "characteristics": characteristics,
            "unkeyed_characteristics": unkeyed_characteristics,
            "relations": fields.get("relation", []),
            "data_processing": fields.get("data_processing", []),
            "supplementary_files": [
                value
                for key, values in fields.items()
                if key.startswith("supplementary_file")
                for value in values
                if value and value.upper() != "NONE"
            ],
        }
    if not samples:
        raise ValueError("GEO SOFT document contains no sample blocks")
    return samples

def _public_date(status: str) -> str:
    match=re.search(r"Public on (\w{3} \d{1,2} \d{4})", status)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1), "%b %d %Y").date().isoformat()
    except ValueError:
        return ""

def _one_characteristic(sample: dict, *keys: str) -> str:
    values: list[str]=[]
    for key in keys:
        values.extend(sample["characteristics"].get(key.lower(), []))
    distinct=list(dict.fromkeys(value.strip() for value in values if value.strip()))
    if len(distinct) > 1:
        raise ValueError(
            f"{sample['gsm']} has conflicting {keys!r} characteristics: {distinct}"
        )
    return distinct[0] if distinct else ""

def canonical_state(sample: dict) -> str:

    cell_type=_one_characteristic(sample, "cell type")
    normalized=" ".join(cell_type.lower().split())
    if "normal pancreatic organoid" in normalized:
        return "healthy_pancreas"
    if "panin organoid" in normalized:
        return "PanIN"
    if "primary tumor organoid" in normalized:
        return "primary_PDAC"
    if "liver metastasis" in normalized or "peritoneum metastasis" in normalized:
        return "metastatic_PDAC"
    raise ValueError(
        f"{sample['gsm']} has unmapped depositor cell type {cell_type!r}; explicit review required"
    )

def canonical_organism(sample: dict) -> tuple[str, str]:
    organism=" ".join(str(sample.get("organism", "")).split())
    mapping={"Homo sapiens": "human", "Mus musculus": "mouse"}
    if organism not in mapping:
        raise ValueError(f"{sample['gsm']} has unsupported organism {organism!r}")
    return organism, mapping[organism]

def canonical_genome(sample: dict, species: str) -> str:

    processing=" ".join(sample.get("data_processing", [])).lower()
    allowed={"human": ("hg19", "hg38"), "mouse": ("mm9", "mm10")}[species]
    matches=[genome for genome in allowed if re.search(rf"\b{genome}\b", processing)]
    if len(matches) != 1:
        raise ValueError(
            f"{sample['gsm']} has ambiguous depositor genome build for {species}: {matches}"
        )
    return matches[0]

def perturbation_label(sample: dict) -> str:
    value=_one_characteristic(sample, "perturbation", "peturbation")
    return value.strip() or "none"

def state_vector(state: str, species: str, sample: dict) -> tuple[float, ...]:

    vector=[0.0] * 18
    state_index={
        "healthy_pancreas": 0,
        "PanIN": 1,
        "primary_PDAC": 2,
        "metastatic_PDAC": 3,
    }
    if state in state_index:
        vector[state_index[state]]=1.0
    elif state != "engineered_precursor":
        raise ValueError(f"unsupported canonical state {state!r}")
    cell_type=_one_characteristic(sample, "cell type").lower()
    if "organoid" in cell_type:
        vector[8]=1.0
    if "pdx" in cell_type:
        vector[9]=1.0
    if "primary tumor" in cell_type:
        vector[10]=1.0
    vector[15]=1.0
    vector[16 if species == "human" else 17]=1.0
    return tuple(vector)

def driver_perturbation_vector(genotype: str, treatment: str = "") -> tuple[float, ...]:

    genotype=genotype.strip().upper()
    if genotype not in {"GFP (CONTROL)", "WT", "K", "KC", "KP", "KCP", "KCPS"}:
        raise ValueError(f"unmapped engineered genotype {genotype!r}")
    vector=[0.0] * 22
    if genotype not in {"GFP (CONTROL)", "WT"}:
        if "C" in genotype:
            vector[15]=-1.0
        if "P" in genotype:
            vector[14]=-1.0
        if "S" in genotype:
            vector[6]=-1.0
    treatment_key=" ".join(treatment.lower().split())
    if treatment_key in {"", "dmso"}:
        pass
    elif treatment_key == "erki":
        vector[0]=-1.0
        vector[18]=1.0
    else:
        raise ValueError(f"unmapped engineered treatment {treatment!r}")
    if any(vector):
        vector[21]=1.0
    return tuple(vector)

def _title_replicate(sample: dict) -> str:
    title=sample.get("title", "").strip()
    match=re.search(r"(?:\bRep|_)(\d+)$", title, flags=re.IGNORECASE)
    return f"rep{match.group(1)}" if match else "unspecified"

def _profile_resolution(assay: str, *, eligible: bool = True, reason: str = "") -> dict:
    return {
        "canonical_assay": assay,
        "profile_eligible": eligible,
        "profile_exclusion_reason": reason,
    }

def _cell_line_assay(sample: dict) -> dict:
    title=sample.get("title", "").lower()
    if "atac-seq" in title:
        return _profile_resolution("ATAC")
    for mark in ("h3k4me3", "h3k27ac", "h3k27me3", "h3k9me3"):
        if mark in title:
            return _profile_resolution(mark)
    if "ctcf" in title:
        return _profile_resolution("CTCF_or_TF")
    if "h3k36me3" in title:
        return _profile_resolution(
            "H3K36me3",
            eligible=False,
            reason="H3K36me3 is not a registered output channel",
        )
    return _profile_resolution(
        "unsupported",
        eligible=False,
        reason="Hi-C, RNA, and input controls are outside profile supervision",
    )

def _engineered_state(genotype: str) -> str:
    genotype=genotype.strip().upper()
    if genotype in {"GFP (CONTROL)", "WT"}:
        return "healthy_pancreas"
    if genotype in {"K", "KC", "KP"}:
        return "engineered_precursor"
    if genotype in {"KCP", "KCPS"}:
        return "primary_PDAC"
    raise ValueError(f"unmapped engineered genotype {genotype!r}")

def _engineered_state_vector(state: str, sample: dict, genotype: str) -> tuple[float, ...]:
    vector=list(state_vector(state, "human", sample))
    if genotype.strip().upper() not in {"GFP (CONTROL)", "WT"}:
        vector[13]=1.0
    if state == "engineered_precursor":
        vector[15]=0.75
    return tuple(vector)

def _strict_title_group(sample: dict, pattern: str, label: str) -> str:
    match=re.match(pattern, sample.get("title", "").strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"{sample['gsm']} has unmapped depositor {label} title")
    return match.group(1)

def _external_klf5_time_hours(sample: dict) -> int | None:

    title=sample.get("title", "").strip()
    characteristic=" ".join(
        value
        for key in ("treatment", "time", "time point", "timepoint")
        for value in sample.get("characteristics", {}).get(key, [])
    )
    combined=f"{title} {characteristic}"
    matches={
        int(value)
        for value in re.findall(
            r"(?:dTAG(?:v1)?[^0-9]{0,12}|\b)(0|1|4|24)\s*h(?:ours?)?(?=\W|_|$)",
            combined,
            flags=re.IGNORECASE,
        )
    }
    if len(matches) > 1:
        raise ValueError(f"{sample['gsm']} has conflicting KLF5 dTAG times {sorted(matches)}")
    if matches:
        return next(iter(matches))
    lowered=combined.lower()
    if any(token in lowered for token in ("dmso", "vehicle", "untreated")):
        return 0
    if "dtag" in lowered:
        raise ValueError(f"{sample['gsm']} has dTAG treatment without a registered time")
    return None

def _external_klf5_perturbation(hours: int | None) -> tuple[float, ...]:
    vector=[0.0] * 22
    if hours is None or hours == 0:
        return tuple(vector)
    if hours not in {1, 4, 24}:
        raise ValueError(f"unsupported KLF5 dTAG time {hours}")
    vector[16]=-1.0
    vector[18]=1.0
    vector[19]=hours / 24.0
    vector[20]=1.0
    vector[21]=1.0
    return tuple(vector)

def _external_chip_assay(sample: dict) -> str:
    title=sample.get("title", "").lower()
    if "atac" in title:
        return "ATAC"
    if "h3k27ac" in title:
        return "H3K27ac"
    if "h3k4me3" in title:
        return "H3K4me3"
    if any(token in title for token in ("klf5", "ctcf", "ruvbl", "ino80", "input")):
        return "CTCF_or_TF"
    raise ValueError(f"{sample['gsm']} has an unmapped external chromatin assay title")

def resolve_sample_metadata(sample: dict, asset: dict) -> dict:

    organism, species=canonical_organism(sample)
    genome=canonical_genome(sample, species)
    decoder=asset.get("metadata_decoder")
    if decoder == "bojq_progression_v1":
        state=canonical_state(sample)
        perturbation=perturbation_label(sample)
        profile=_profile_resolution("from_archive_filename")
        exclusion=asset.get("sample_profile_exclusions", {}).get(sample["gsm"])
        if exclusion:
            profile=_profile_resolution(
                "from_archive_filename", eligible=False, reason=str(exclusion)
            )
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": state,
            "canonical_genome": genome,
            "sample_group": _sample_group(sample),
            "perturbation_label": perturbation,
            "perturbation_control_family": perturbation_control_family(perturbation),
            "state_features": state_vector(state, species, sample),
            "perturbation_features": signed_perturbation_vector(perturbation),
            "biological_replicate": _title_replicate(sample),
            "pair_relation": "intervention"
            if any(signed_perturbation_vector(perturbation))
            else "control",
            **profile,
        }
    if species != "human":
        raise ValueError(f"{decoder} requires human samples")

    if decoder == "pdac_cell_line_state_v1":
        cell_type=" ".join(_one_characteristic(sample, "cell type").lower().split())
        state_map={
            "normal pancreatic cells": "healthy_pancreas",
            "primary pancreatic cancer cells": "primary_PDAC",
            "pancreatic cancer cells derived from liver metastasis": "metastatic_PDAC",
        }
        if cell_type not in state_map:
            raise ValueError(f"unmapped PDAC cell-line state {cell_type!r}")
        group=_strict_title_group(sample, r"(HPNE|PANC-?1|Capan-?1)\b", "cell-line")
        state=state_map[cell_type]
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": state,
            "canonical_genome": genome,
            "sample_group": group.upper().replace("-", ""),
            "perturbation_label": "none",
            "perturbation_control_family": "unperturbed",
            "state_features": state_vector(state, species, sample),
            "perturbation_features": (0.0,) * 22,
            "biological_replicate": _title_replicate(sample),
            "pair_relation": "unpaired",
            **_cell_line_assay(sample),
        }

    if decoder == "human_progenitor_organoid_progression_v1":
        genotype=_one_characteristic(sample, "genotype")
        treatment=_one_characteristic(sample, "treatment")
        state=_engineered_state(genotype)
        replicate=_title_replicate(sample)
        title=sample.get("title", "").lower()
        profile=(
            _profile_resolution("ATAC")
            if "atac" in title
            else _profile_resolution("WGBS")
            if "methyl" in title
            else _profile_resolution(
                "RNA", eligible=False, reason="RNA auxiliary transform is not frozen"
            )
        )
        treatment_group=treatment.strip() or "untreated"
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": state,
            "canonical_genome": genome,
            "sample_group": f"HUES8:{genotype}:{treatment_group}:{replicate}",
            "perturbation_label": f"genotype={genotype};treatment={treatment_group}",
            "perturbation_control_family": "unperturbed",
            "state_features": _engineered_state_vector(state, sample, genotype),
            "perturbation_features": driver_perturbation_vector(genotype, treatment),
            "biological_replicate": replicate,
            "pair_relation": "unpaired",
            **profile,
        }

    if decoder == "human_progenitor_organoid_rna_v1":
        raw_genotype=_one_characteristic(sample, "genotype")
        genotype_upper=raw_genotype.upper()
        genotype=(
            "GFP (CONTROL)"
            if genotype_upper == "GFP CONTROL"
            else "KCP"
            if genotype_upper.startswith("KCP WITH")
            else raw_genotype
        )
        treatment=_one_characteristic(sample, "treatment")
        treatment_lower=treatment.lower()
        mechanism_treatment="ERKi" if "treated with erki" in treatment_lower else ""
        perturbation=list(driver_perturbation_vector(genotype, mechanism_treatment))
        if "sgrna" in treatment_lower and "control sgrna" not in treatment_lower:
            if not any(factor in treatment_lower for factor in ("fos", "junb", "jund")):
                raise ValueError(f"unmapped AP-1 perturbation {treatment!r}")
            perturbation[4]=-1.0
            perturbation[21]=1.0
        state=_engineered_state(genotype)
        replicate=_title_replicate(sample)
        condition=re.sub(r"_RNA-seq_\d+$", "", sample.get("title", ""))
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": state,
            "canonical_genome": genome,
            "sample_group": f"HUES8:{condition}:{replicate}",
            "perturbation_label": treatment,
            "perturbation_control_family": "unperturbed",
            "state_features": _engineered_state_vector(state, sample, genotype),
            "perturbation_features": tuple(perturbation),
            "biological_replicate": replicate,
            "pair_relation": "unpaired",
            **_profile_resolution(
                "RNA", eligible=False, reason="RNA auxiliary transform is not frozen"
            ),
        }

    if decoder == "human_progenitor_organoid_occupancy_v1":
        genotype=_one_characteristic(sample, "genotype")
        treatment=_one_characteristic(sample, "treatment")
        state=_engineered_state(genotype)
        title=sample.get("title", "")
        target=_strict_title_group(
            sample,
            r"KCP_(FOSL2|FOS|JUNB|H3K27Ac|DMSO_H3K27Ac|LY_H3K27Ac)",
            "CUT&RUN target",
        )
        assay="H3K27ac" if "h3k27ac" in title.lower() else "CTCF_or_TF"
        replicate=_title_replicate(sample)
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": state,
            "canonical_genome": genome,
            "sample_group": f"HUES8:KCP:{target}:{replicate}",
            "perturbation_label": f"genotype={genotype};treatment={treatment or 'untreated'}",
            "perturbation_control_family": "unperturbed",
            "state_features": _engineered_state_vector(state, sample, genotype),
            "perturbation_features": driver_perturbation_vector(genotype, treatment),
            "biological_replicate": replicate,
            "pair_relation": "unpaired",
            **_profile_resolution(assay),
        }

    if decoder == "patient_pdac_validation_v1":
        tissue=" ".join(_one_characteristic(sample, "tissue type").lower().split())
        if tissue != "pancreatic ductal adenocarcinoma":
            raise ValueError(f"unmapped patient tissue type {tissue!r}")
        group=_strict_title_group(sample, r"(MSKB\d+)\b", "patient")
        state="primary_PDAC"
        state_features=list(state_vector(state, species, sample))
        if _one_characteristic(sample, "kras allele").upper() != "ND":
            state_features[13]=1.0
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": state,
            "canonical_genome": genome,
            "sample_group": group.upper(),
            "perturbation_label": "none",
            "perturbation_control_family": "unperturbed",
            "state_features": tuple(state_features),
            "perturbation_features": (0.0,) * 22,
            "biological_replicate": "patient_profile",
            "pair_relation": "unpaired",
            **_profile_resolution("ATAC"),
        }

    if decoder == "human_pancreatic_progenitor_occupancy_v1":
        cell_type=" ".join(_one_characteristic(sample, "cell type").lower().split())
        if cell_type != "early pancreatic progenitor":
            raise ValueError(f"unmapped progenitor cell type {cell_type!r}")
        title=sample.get("title", "")
        replicate=_title_replicate(sample)
        eligible="input" not in title.lower()
        target_match=re.search(r"(TET1|ONECUT1)", title, flags=re.IGNORECASE)
        target=target_match.group(1).upper() if target_match else "INPUT"
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": "engineered_precursor",
            "canonical_genome": genome,
            "sample_group": f"H1:{target}:{replicate}",
            "perturbation_label": "none",
            "perturbation_control_family": "unperturbed",
            "state_features": state_vector("engineered_precursor", species, sample),
            "perturbation_features": (0.0,) * 22,
            "biological_replicate": replicate,
            "pair_relation": "unpaired",
            **_profile_resolution(
                "CTCF_or_TF",
                eligible=eligible,
                reason="input control is not a prediction target" if not eligible else "",
            ),
        }

    if decoder == "pdac_differentiation_chip_v1":
        group=_strict_title_group(
            sample,
            r"(?:H3\w+\.|INPUT\.|)(CAPAN1|CAPAN2|CFPAC1|HPAF2|MiaPaca2|PANC1|PT45P1)",
            "PDAC cell-line",
        )
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": "primary_PDAC",
            "canonical_genome": genome,
            "sample_group": group.upper(),
            "perturbation_label": "none",
            "perturbation_control_family": "unperturbed",
            "state_features": state_vector("primary_PDAC", species, sample),
            "perturbation_features": (0.0,) * 22,
            "biological_replicate": _title_replicate(sample),
            "pair_relation": "unpaired",
            **_profile_resolution(
                "BED", eligible=False, reason="BED peak rasterizer is not frozen"
            ),
        }
    if decoder == "external_klf5_dtag_timecourse_v1":
        title=sample.get("title", "")
        line_match=re.search(
            r"(L36pl(?:Clone2)?|AsPC-?1|BxPC-?3|HPAF-?II)",
            title,
            flags=re.IGNORECASE,
        )
        if not line_match:
            raise ValueError(f"{sample['gsm']} has an unmapped external PDAC cell line")
        normalized_line=re.sub(r"[^a-z0-9]", "", line_match.group(1).lower())
        registered_l36=normalized_line in {"l36pl", "l36plclone2"}
        hours=_external_klf5_time_hours(sample)
        perturbation=_external_klf5_perturbation(hours)
        assay=_external_chip_assay(sample)
        replicate=_title_replicate(sample)
        context="ekstrom_johnsen::L36pl_clone2"
        eligible=registered_l36 and "input" not in title.lower()
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": "primary_PDAC",
            "canonical_genome": genome,
            "sample_group": context,
            "perturbation_label": (
                "none" if hours in {None, 0} else f"KLF5_dTAG_{hours}h"
            ),
            "perturbation_control_family": "KLF5_dTAG_timecourse",
            "state_features": state_vector("primary_PDAC", species, sample),
            "perturbation_features": perturbation,
            "biological_replicate": replicate,
            "pair_relation": (
                "unpaired"
                if hours is None
                else "control"
                if hours == 0
                else "intervention"
            ),
            **_profile_resolution(
                assay,
                eligible=eligible,
                reason=(
                    "cell line is not a registered external independence context"
                    if not registered_l36
                    else "input control is not a prediction target"
                    if "input" in title.lower()
                    else ""
                ),
            ),
        }
    if decoder == "external_klf5_lineage_dtag_v1":
        title=sample.get("title", "")
        line_match=re.search(r"(AsPC-?1|T3M4)", title, flags=re.IGNORECASE)
        if not line_match:
            return {
                "canonical_organism": organism,
                "species": species,
                "canonical_state": "primary_PDAC",
                "canonical_genome": genome,
                "sample_group": "excluded_external_context",
                "perturbation_label": "none",
                "perturbation_control_family": "KLF5_dTAG_4h",
                "state_features": state_vector("primary_PDAC", species, sample),
                "perturbation_features": (0.0,) * 22,
                "biological_replicate": _title_replicate(sample),
                "pair_relation": "unpaired",
                **_profile_resolution(
                    "CTCF_or_TF",
                    eligible=False,
                    reason="cell line is not a registered external independence context",
                ),
            }
        compact_line=re.sub(r"[^A-Za-z0-9]", "", line_match.group(1))
        line="AsPC1" if compact_line.lower() == "aspc1" else "T3M4"
        hours=_external_klf5_time_hours(sample)
        perturbation=_external_klf5_perturbation(hours)
        assay=_external_chip_assay(sample)
        replicate=_title_replicate(sample)
        eligible="input" not in title.lower()
        return {
            "canonical_organism": organism,
            "species": species,
            "canonical_state": "primary_PDAC",
            "canonical_genome": genome,
            "sample_group": f"cunniff_vakoc::{line}",
            "perturbation_label": (
                "none" if hours in {None, 0} else f"KLF5_dTAG_{hours}h"
            ),
            "perturbation_control_family": "KLF5_dTAG_4h",
            "state_features": state_vector("primary_PDAC", species, sample),
            "perturbation_features": perturbation,
            "biological_replicate": replicate,
            "pair_relation": (
                "unpaired"
                if hours is None
                else "control"
                if hours == 0
                else "intervention"
            ),
            **_profile_resolution(
                assay,
                eligible=eligible,
                reason="input control is not a prediction target" if not eligible else "",
            ),
        }
    raise ValueError(f"unsupported metadata decoder {decoder!r}")

def signed_perturbation_vector(label: str) -> tuple[float, ...]:

    vector=[0.0] * 22
    compact=re.sub(r"[^a-z0-9]+", "", label.lower())
    if compact in {"", "none"} or any(
        control in compact for control in ("mscvemp", "shren", "sgrosa")
    ):
        return tuple(vector)

    if "foxa1" in compact and "gata5" in compact:
        vector[12]=vector[13]=1.0
    elif "shfoxa1" in compact or "sgfoxa1" in compact:
        vector[12]=-1.0
    elif "sgp53" in compact:
        vector[14]=-1.0
    elif "foxa1" in compact:
        vector[12]=1.0
    elif "gata5" in compact:
        vector[13]=1.0
    else:
        raise ValueError(f"unmapped perturbation label {label!r}; explicit review required")
    vector[21]=1.0
    return tuple(vector)

def perturbation_control_family(label: str) -> str:

    compact=re.sub(r"[^a-z0-9]+", "", label.lower())
    if compact in {"", "none"}:
        return "unperturbed"
    if "mscvemp" in compact or (
        "mscv" in compact and any(axis in compact for axis in ("foxa1", "gata5"))
    ):
        return "mscv_overexpression"
    if "shren" in compact or "shfoxa1" in compact:
        return "mire_shrna"
    if "sgrosa" in compact or "sgp53" in compact or "sgfoxa1" in compact:
        return "lentiviral_crispr"
    raise ValueError(
        f"unmapped perturbation control family {label!r}; explicit review required"
    )

def geo_assay_vector(assay: str) -> tuple[float, ...]:
    assay_key=assay.lower()
    if assay_key == "atac":
        metadata={"assay": "ATAC-seq", "target": "", "output_type": "signal"}
    elif assay_key in {"h3k27ac", "h3k4me1", "h3k4me3", "h3k27me3", "h3k9me3"}:
        metadata={
            "assay": "Histone ChIP-seq",
            "target": assay,
            "output_type": "signal",
        }
    elif assay_key in {"foxa1", "ctcf_or_tf"}:
        target="FOXA1" if assay_key == "foxa1" else "TF"
        metadata={"assay": "TF ChIP-seq", "target": target, "output_type": "signal"}
    elif assay_key == "wgbs":
        metadata={"assay": "WGBS", "target": "", "output_type": "methylation"}
    else:
        raise ValueError(f"unsupported GEO profile assay {assay!r}")
    metadata.update({"status": "released", "audit_errors": 0})
    return assay_vector(metadata)

def _sample_group(sample: dict) -> str:
    source=sample.get("source_name", "").strip()
    title=sample.get("title", "").strip()
    candidate=source or title
    if not candidate:
        raise ValueError(f"{sample['gsm']} lacks source name and title")
    return re.split(r"[/_\s]+", candidate, maxsplit=1)[0]

def fetch_geo_soft_metadata(
    project_root: str | Path,
    accession: str,
    *,
    refresh: bool = False,
    allow_protected_metadata: bool = False,
    protected_release_path: str | Path | None = None,
) -> dict:
    root=Path(project_root)
    accession=accession.upper()
    asset=_registered_asset(root, accession)
    split=str(asset.get("split", "")).lower()
    role=str(asset.get("role", "")).lower()
    protected="test" in split or "holdout" in role
    if protected and not allow_protected_metadata:
        raise PermissionError(
            f"{accession} is a protected test/holdout; metadata labels stay sealed until freeze"
        )
    if protected:
        from .protected import validate_protected_metadata_release

        if protected_release_path is None:
            raise PermissionError(
                f"{accession} metadata requires a post-training protected release manifest"
            )
        release=validate_protected_metadata_release(
            root, protected_release_path, accession=accession
        )
        if not release["ok"]:
            raise PermissionError(
                f"{accession} protected metadata release failed: {release['failures']}"
            )
    cache_dir=root / "data" / "metadata" / "geo" / accession
    cache_dir.mkdir(parents=True, exist_ok=True)
    soft_path=cache_dir / f"{accession}_family.soft.gz"
    if refresh or not soft_path.exists():
        url=geo_soft_url(accession)
        response=_session().get(
            url,
            headers={"User-Agent": "pdac-circuit/0.2 (research-use-only)"},
            timeout=(30, 180),
        )
        response.raise_for_status()
        temporary=soft_path.with_suffix(soft_path.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(soft_path)
    compressed=soft_path.read_bytes()
    text=gzip.decompress(compressed).decode("utf-8", errors="replace")
    samples=parse_geo_soft(text)
    records={}
    errors=[]
    forbidden_characteristics={
        str(key).lower()
        for key in asset.get("forbidden_outcome_characteristics", [])
    }
    for gsm, sample in samples.items():
        try:
            resolved=resolve_sample_metadata(sample, asset)
            sanitized={
                **sample,
                "characteristics": {
                    key: value
                    for key, value in sample["characteristics"].items()
                    if key not in forbidden_characteristics
                },
            }
            records[gsm]={**sanitized, **resolved}
        except ValueError as exc:
            errors.append(
                {
                    "gsm": gsm,
                    "error": str(exc),
                    "sample": {
                        **sample,
                        "characteristics": {
                            key: value
                            for key, value in sample["characteristics"].items()
                            if key not in forbidden_characteristics
                        },
                    },
                }
            )
    report={
        "schema": SOFT_SCHEMA,
        "accession": accession,
        "source_url": geo_soft_url(accession),
        "source_path": str(soft_path.relative_to(root)),
        "source_sha256": hashlib.sha256(compressed).hexdigest(),
        "protected": protected,
        "samples_total": len(samples),
        "samples_resolved": len(records),
        "samples": records,
        "errors": errors,
        "state_policy": "Depositor cell type only; filenames/titles never determine state.",
        "perturbation_policy": "Depositor perturbation/peturbation only; signed mechanism axes.",
        "metadata_decoder": asset.get("metadata_decoder"),
        "forbidden_outcome_characteristics_redacted": sorted(forbidden_characteristics),
        "outcome_policy": (
            "Registered outcome characteristics are removed from derived metadata and never "
            "enter TrackSpecs, checkpoint selection, or model inputs."
            if forbidden_characteristics
            else "No study-specific outcome characteristics are registered."
        ),
    }
    registry_path=cache_dir / "metadata.json"
    registry_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["registry_path"]=str(registry_path.relative_to(root))
    return report

def _iter_bigwigs(extracted_dir: Path) -> Iterable[Path]:
    for path in sorted(extracted_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".bw", ".bigwig"}:
            yield path

def _declared_profile_files(samples: dict[str, dict]) -> dict[str, str]:
    declared: dict[str, str]={}
    for gsm, sample in samples.items():
        for url in sample.get("supplementary_files", []):
            name=Path(unquote(urlparse(url).path)).name
            if not name:
                continue
            prior=declared.setdefault(name.lower(), gsm)
            if prior != gsm:
                raise ValueError(f"supplementary profile {name!r} maps to multiple samples")
    return declared

def build_geo_track_specs(
    project_root: str | Path,
    accession: str,
    extracted_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    evaluation_only: bool = False,
    protected_release_path: str | Path | None = None,
) -> dict:

    root=Path(project_root)
    accession=accession.upper()
    asset=_registered_asset(root, accession)
    allowed_genomes=set(asset.get("reference_genomes") or [])
    if asset.get("reference_genome"):
        allowed_genomes.add(str(asset["reference_genome"]))
    if not allowed_genomes or not allowed_genomes <= {"hg38", "hg19", "mm10", "mm9"}:
        raise ValueError(
            f"{accession} has no supported reference genome set in pdac_chromatin_assets.json"
        )
    metadata_path=Path(metadata_path) if metadata_path else (
        root / "data" / "metadata" / "geo" / accession / "metadata.json"
    )
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SOFT_SCHEMA or metadata.get("accession") != accession:
        raise ValueError("metadata registry does not match requested GEO accession")
    protected=bool(metadata.get("protected"))
    protected_release_sha256=None
    if protected:
        if not evaluation_only:
            raise PermissionError(
                "protected metadata can create evaluation-only TrackSpecs, never training TrackSpecs"
            )
        if protected_release_path is None:
            raise PermissionError("protected evaluation TrackSpecs require a release manifest")
        from .protected import validate_protected_metadata_release

        release=validate_protected_metadata_release(
            root, protected_release_path, accession=accession
        )
        if not release["ok"]:
            raise PermissionError(
                f"protected evaluation release failed: {release['failures']}"
            )
        release_file=Path(protected_release_path)
        if not release_file.is_absolute():
            release_file=root / release_file
        protected_release_sha256=sha256_file(release_file)
    elif evaluation_only:
        raise ValueError("evaluation_only is reserved for protected test studies")
    if metadata.get("errors"):
        raise RuntimeError(
            f"{accession} has {len(metadata['errors'])} unresolved metadata records; review first"
        )
    samples=metadata["samples"]
    extracted_dir=Path(extracted_dir)
    extraction_manifest=extracted_dir / "manifest.json"
    hashes={}
    if extraction_manifest.exists():
        extraction=json.loads(extraction_manifest.read_text(encoding="utf-8"))
        hashes={row["name"]: row["sha256"] for row in extraction.get("files", [])}
    holdout_groups=set(asset.get("held_out_sample_groups") or [])
    output_dir=root / "data" / (
        "evaluation_track_specs" if evaluation_only else "track_specs"
    ) / accession
    output_dir.mkdir(parents=True, exist_ok=True)
    declared_profiles=_declared_profile_files(samples)
    written, failures, excluded=[], [], []
    for path in _iter_bigwigs(extracted_dir):
        relative=path.relative_to(extracted_dir).as_posix()
        match=BIGWIG_RE.fullmatch(path.name)
        gsm=declared_profiles.get(path.name.lower())
        if gsm is None and match:
            gsm=match.group("gsm")
        if gsm is None:
            failures.append(
                {"path": str(path), "error": "profile is not declared by GEO metadata"}
            )
            continue
        try:
            sample=samples[gsm]
            if not sample.get("profile_eligible", True):
                excluded.append(
                    {
                        "path": relative,
                        "gsm": gsm,
                        "reason": sample.get("profile_exclusion_reason"),
                    }
                )
                continue
            state=sample["canonical_state"]
            group=sample["sample_group"]
            organism=sample["canonical_organism"]
            expected_genome=sample["canonical_genome"]
            if expected_genome not in allowed_genomes:
                raise ValueError(
                    f"asset genomes {sorted(allowed_genomes)} conflict with "
                    f"{gsm} organism {organism}"
                )
            genome=expected_genome
            track_id=path.stem
            replicate=sample.get("biological_replicate") or "unspecified"
            if replicate == "unspecified" and match and match.group("replicate"):
                replicate=f"rep{match.group('replicate')}"
            perturbation_features=tuple(sample["perturbation_features"])
            pair_relation=sample.get("pair_relation", "unpaired")
            control_family=str(sample["perturbation_control_family"])
            assay=sample.get("canonical_assay")
            if assay == "from_archive_filename":
                if not match:
                    raise ValueError("archive filename does not declare the profile assay")
                assay=match.group("assay")
            pair_group=""
            if pair_relation in {"control", "intervention"}:
                pair_group=(
                    f"{accession}:{group}:{str(assay).lower()}:{replicate}:"
                    f"{control_family}"
                )
            scoped_group=(
                group
                if evaluation_only or asset.get("group_scope") == "global_cell_line"
                else f"{accession}:{group}"
            )
            validation_only=asset.get("split") == "validation_study_by_patient"
            spec=TrackSpec(
                accession=track_id,
                path=str(path.resolve()),
                assay_features=geo_assay_vector(assay),
                state_features=tuple(sample["state_features"]),
                perturbation_features=perturbation_features,
                sample_group=scoped_group,
                study=accession,
                released=sample.get("release_date", ""),
                disease=state in {"primary_PDAC", "metastatic_PDAC"},
                source_sha256=hashes.get(relative) or sha256_file(path),
                split_role=(
                    "external_study"
                    if evaluation_only
                    else "validation_study"
                    if validation_only
                    else "held_out_state"
                    if group in holdout_groups
                    else "train_state"
                ),
                genome=genome,
                organism=organism,
                sample_accession=gsm,
                biological_state=state,
                perturbation_label=sample["perturbation_label"],
                metadata_sha256=metadata["source_sha256"],
                biological_replicate=replicate,
                pair_group=pair_group,
                pair_relation=pair_relation,
                pair_control_family=control_family,
            )
            spec.validate()
            with _open_bigwig(spec.path) as (reader, _):
                _assert_bigwig_native_genome(reader, spec)
            spec_path=output_dir / f"{track_id}.json"
            spec_path.write_text(json.dumps(asdict(spec), indent=2, sort_keys=True), encoding="utf-8")
            written.append(
                {
                    "track": track_id,
                    "gsm": gsm,
                    "sample_group": spec.sample_group,
                    "state": state,
                    "assay": assay,
                    "perturbation": spec.perturbation_label,
                    "pair_control_family": spec.pair_control_family,
                    "genome": genome,
                    "split_role": spec.split_role,
                    "native_genome_validated": True,
                    "spec": str(spec_path.relative_to(root)),
                }
            )
        except (KeyError, ValueError) as exc:
            failures.append({"path": str(path), "gsm": gsm, "error": str(exc)})
    report={
        "schema": TRACK_INDEX_SCHEMA,
        "accession": accession,
        "genomes": sorted(allowed_genomes),
        "metadata_sha256": metadata["source_sha256"],
        "extraction_manifest_sha256": (
            sha256_file(extraction_manifest) if extraction_manifest.exists() else None
        ),
        "written": written,
        "failures": failures,
        "excluded_profiles": excluded,
        "evaluation_only": bool(evaluation_only),
        "protected_release_sha256": protected_release_sha256,
        "state_policy": "Authoritative GEO metadata only.",
    }
    (output_dir / "index.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
