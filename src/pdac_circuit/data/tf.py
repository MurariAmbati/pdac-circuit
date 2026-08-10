from __future__ import annotations

import csv
import io
import zipfile
from functools import lru_cache

from ..core.paths import RAW

LAMBERT_CSV = RAW / "lambert-tf" / "DatabaseExtract_v_1.01.csv"
INTOGEN_ZIP = RAW / "intogen-pdac" / "IntOGen-Drivers.zip"

MOFFITT_BASAL = [
    "VGLL1","UCA1","S100A2","LY6D","SPRR3","SPRR1B","LEMD1","KRT15","CTSV",
    "DHRS9","AREG","CST6","SERPINB3","KRT6C","KRT6A","SERPINB4","FAM83A",
    "SCEL","FGFBP1","KRT7","KRT17","GPR87","TNS4","SLC2A1","ANXA8L2",
]
MOFFITT_CLASSICAL = [
    "BTNL8","FAM3D","AGR3","CTSE","LYZ","TFF2","TFF1","ANXA10","LGALS4",
    "PLA2G10","CEACAM6","VSIG2","TSPAN8","ST6GALNAC1","AGR2","TFF3","CYP3A7",
    "MYO1A","CLRN3","KRT20","CDH17","SPINK4","REG4","GATA6","FOXA2",
]
SUBTYPE_SIGNATURES = {"basal": MOFFITT_BASAL,"classical": MOFFITT_CLASSICAL}

PDAC_TF_CONTROLS = ["GATA6","KLF5","HNF1A","ZEB1","MYC","FOXA2","TP63"]

@lru_cache(maxsize=1)
def load_tf_list() -> dict[str,dict]:
    if not LAMBERT_CSV.exists():
        raise FileNotFoundError(f"Lambert TF catalog not found at {LAMBERT_CSV}; run fetch-data lambert-tf")
    out: dict[str,dict] = {}
    with open(LAMBERT_CSV,newline="",encoding="utf-8",errors="replace") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        sym_c = cols.get("hgnc symbol") or cols.get("symbol") or list(cols.values())[1]
        ens_c = cols.get("ensembl id") or cols.get("ensembl") or list(cols.values())[0]
        istf_c = next((cols[k] for k in cols if "is tf" in k),None)
        fam_c = next((cols[k] for k in cols if "family" in k or "dbd" in k),None)
        for row in reader:
            is_tf = (row.get(istf_c,"") or "").strip().lower() == "yes" if istf_c else True
            if not is_tf:
                continue
            sym = (row.get(sym_c,"") or "").strip()
            if not sym:
                continue
            out[sym] = {
                "ensembl": (row.get(ens_c,"") or "").strip(),
                "family": (row.get(fam_c,"") or "").strip() if fam_c else "",
                "is_tf": True,
            }
    return out

@lru_cache(maxsize=1)
def load_intogen_drivers(cancer_types: tuple[str,...] = ("PAAD","PACA","PDAC")) -> dict[str,dict]:
    if not INTOGEN_ZIP.exists():
        return {}
    drivers: dict[str,dict] = {}
    with zipfile.ZipFile(INTOGEN_ZIP) as z:
        member = next((n for n in z.namelist() if n.lower().endswith(".tsv") and "cancer_genes" in n.lower()),None)
        if member is None:
            member = next((n for n in z.namelist() if n.lower().endswith(".tsv")),None)
        if member is None:
            return {}
        with z.open(member) as fh:
            text = io.TextIOWrapper(fh,encoding="utf-8",errors="replace")
            reader = csv.DictReader(text,delimiter="\t")
            cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
            gene_c = cols.get("symbol") or cols.get("gene") or cols.get("hugo_symbol")
            ct_c = next((cols[k] for k in cols if "cancer" in k and "type" in k),None) or cols.get("cohort")
            q_c = next((cols[k] for k in cols if k in ("qvalue","qvalue_combination","q_value")),None)
            for row in reader:
                ct = (row.get(ct_c,"") or "").upper() if ct_c else ""
                if cancer_types and not any(c in ct for c in cancer_types):
                    continue
                gene = (row.get(gene_c,"") or "").strip()
                if not gene:
                    continue
                try:
                    q = float(row.get(q_c,"nan")) if q_c else float("nan")
                except (TypeError,ValueError):
                    q = float("nan")
                drivers[gene] = {"qvalue": q,"cancer_type": ct}
    return drivers

def subtype_signature_genes() -> dict[str,list[str]]:
    return {k: list(v) for k,v in SUBTYPE_SIGNATURES.items()}
