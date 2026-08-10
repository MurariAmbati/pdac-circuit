from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from figstyle import PALETTE, apply_publication_style, finalize_figure, ygrid

_HERE=Path(__file__).resolve().parents[1]
SRC=_HERE if (_HERE/"results").is_dir() else Path("C:/Users/murar/pdac-circuit")
OUT=_HERE / "images"

INK=PALETTE["ink"]
BLUE=PALETTE["blue_main"]
BLUE2=PALETTE["blue_secondary"]
RED=PALETTE["red_strong"]
NEUT=PALETTE["neutral"]
GREY=PALETTE["grey_mid"]
TEAL=PALETTE["teal"]


def load():
    s=json.loads((SRC/"results"/"circuit_design_campaign.json").read_text(encoding="utf-8"))
    g=SRC/"results"/s["all_circuits_file"]
    rows=[json.loads(ln) for ln in gzip.open(g,"rt",encoding="utf-8")]
    return s,rows


def main():
    apply_publication_style()
    s,rows=load()
    comp=np.array([r["composite"] for r in rows])
    eff=np.array([r["efficacy"] for r in rows])
    saf=np.array([r["safety"] for r in rows])
    ps=np.array([r["promoter_strength"] for r in rows])
    kd=np.array([r["knockdown"] for r in rows])
    front=np.array([r["pareto_rank"]==0 for r in rows])
    single=np.array([r["partner"] is None for r in rows])

    fig,axes=plt.subplots(2,2,figsize=(13.8,10.6))
    fig.subplots_adjust(left=0.072,right=0.985,top=0.955,bottom=0.105,wspace=0.30,hspace=0.34)

    ax=axes[0][0]
    ax.hist(comp[single],bins=44,color=BLUE2,edgecolor="white",linewidth=0.5,label="single target")
    if (~single).any():
        ax.hist(comp[~single],bins=44,color=RED,alpha=0.75,edgecolor="white",
                linewidth=0.5,label="paired target")
    ax.axvline(float(np.median(comp)),color=INK,ls="--",lw=1.6)
    ax.text(float(np.median(comp)),ax.get_ylim()[1]*0.94,f"  median {np.median(comp):.3f}",
            color=INK,fontsize=12,va="top")
    ax.set_xlabel("composite score")
    ax.set_ylabel("circuits")
    ax.set_title(f"A  Composite score, {len(rows):,} circuits",loc="left")
    ax.legend(frameon=False,fontsize=12)
    ygrid(ax)

    ax=axes[0][1]
    ax.scatter(eff[~front],saf[~front],s=7,c=NEUT,alpha=0.55,linewidths=0,label="dominated")
    ax.scatter(eff[front],saf[front],s=16,c=RED,linewidths=0,label=f"Pareto front ({front.sum()})")
    fl=s.get("floors",{})
    if fl:
        ax.axhline(fl["safety_floor"],color=INK,ls="--",lw=1.6)
        ax.axvline(fl["efficacy_floor"],color=GREY,ls=":",lw=1.4)
        ax.text(0.02,fl["safety_floor"],f" safety floor {fl['safety_floor']}  (nothing reaches it)",
                color=INK,fontsize=11.5,va="bottom",ha="left")
        ax.text(fl["efficacy_floor"],0.02," efficacy floor",color=GREY,fontsize=11.5,
                va="bottom",ha="left",rotation=90)
        ax.set_ylim(0,max(1.0,fl["safety_floor"]*1.12))
    ax.set_xlabel("efficacy")
    ax.set_ylabel("safety")
    ax.set_title("B  Efficacy against safety",loc="left")
    ax.legend(frameon=True,framealpha=0.92,edgecolor="none",fontsize=12,loc="lower left")
    ygrid(ax)

    ax=axes[1][0]
    lv=sorted({round(v,4) for v in ps})
    med=[np.median(kd[np.isclose(ps,v)]) for v in lv]
    q1=[np.percentile(kd[np.isclose(ps,v)],25) for v in lv]
    q3=[np.percentile(kd[np.isclose(ps,v)],75) for v in lv]
    ax.fill_between(lv,q1,q3,color=BLUE2,alpha=0.25,linewidth=0)
    ax.plot(lv,med,color=BLUE,lw=2.2,marker="o",ms=5)
    ax.set_xlabel("predicted promoter strength")
    ax.set_ylabel("simulated TF knockdown")
    ax.set_title("C  Knockdown against promoter strength",loc="left")
    ygrid(ax)

    ax=axes[1][1]
    per={}
    for r in rows:
        if r["partner"] is None:
            per.setdefault(r["target"],[]).append(r["composite"])
    top=sorted(per,key=lambda t:-max(per[t]))[:18]
    vals=[max(per[t]) for t in top]
    y=np.arange(len(top))[::-1]
    ax.barh(y,vals,color=[TEAL if v>=np.median(vals) else GREY for v in vals],height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(top,fontsize=11)
    ax.set_xlabel("best composite score for that target")
    ax.set_xlim(min(vals)-0.03,max(vals)+0.012)
    ax.set_title(f"D  Best circuit per target, {len(top)} of {len(per)}",loc="left")
    ax.grid(axis="x",color=PALETTE["neutral"],lw=0.7,alpha=0.6)
    ax.set_axisbelow(True)

    fig.text(0.072,0.028,
             f"{s['design_space']['targets_designed']} targets x "
             f"{s['design_space']['promoters_per_target']} promoters x "
             f"{s['design_space']['enhancers_per_target']} enhancers x "
             f"{s['design_space']['guides_per_target']} guides, plus "
             f"{s['n_multi_target']:,} paired-target circuits over "
             f"{'/'.join(s['design_space']['pair_logics'])} logic. "
             f"Every guide carries a genome-wide off-target scan of hg38 at up to four mismatches.",
             fontsize=10.5,color=GREY)
    finalize_figure(fig,OUT,"fig10_design_campaign",pad=None)
    print(f"  fig10_design_campaign  ({len(rows):,} circuits, {front.sum()} on front 0)")


if __name__=="__main__":
    main()
