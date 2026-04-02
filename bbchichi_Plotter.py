"""Make stacked plots from bbχχ histogram ROOT outputs.

Purpose:
- Read histogram ROOT files produced by the analyzer.
- Stack MC backgrounds, overlay data, and save publication-style plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

import argparse
import uproot
from hist import Hist, Stack
import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
from cycler import cycler

from bbchichi_variables import vardict, regions as REGION_LIST, variables_common


def lumi_label():
    return ""


hep.style.use("CMS")
plt.rcParams["axes.prop_cycle"] = cycler(
    color=[
        "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
        "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd",
    ]
)


def ensure_weight_hist(h_in: Hist) -> Hist:
    axes = []
    for ax in h_in.axes:
        edges = getattr(ax, "edges", None)
        if edges is not None:
            axes.append(bh.axis.Variable(edges))
        else:
            axes.append(bh.axis.Regular(ax.size, ax.edges[0], ax.edges[-1]))
    h_out = Hist(*axes, storage=bh.storage.Weight())
    vals = np.asarray(h_in.values(), dtype="f8")
    vars_ = h_in.variances()
    vars_ = np.zeros_like(vals, dtype="f8") if vars_ is None else np.asarray(vars_, dtype="f8")
    view = h_out.view()
    view.value[...] = vals
    view.variance[...] = vars_
    return h_out


def add_hist_safe(a: Hist | None, b: Hist) -> Hist:
    b = ensure_weight_hist(b)
    if a is None:
        return b
    a = ensure_weight_hist(a)
    a.view().value[...] += b.view().value
    a.view().variance[...] += b.view().variance
    return a


def sum_hist_list(hlist):
    out = None
    for h in hlist:
        out = add_hist_safe(out, h)
    return out


def get_ratio(hist_a, hist_b):
    edges_a = hist_a.axes.edges[0]
    edges_b = hist_b.axes.edges[0]
    if not np.array_equal(edges_a, edges_b):
        raise ValueError("Histograms have different binning")
    a = np.where(hist_a.values() < 0, 0.0, hist_a.values())
    b = np.where(hist_b.values() < 0, 0.0, hist_b.values())
    ea = np.sqrt(a)
    eb = np.sqrt(b)
    ratio = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    ratio_hist = Hist(hist_a.axes[0], storage=bh.storage.Double())
    ratio_hist[...] = ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = np.divide(ea, a, out=np.zeros_like(ea, dtype=float), where=a != 0)
        rb = np.divide(eb, b, out=np.zeros_like(eb, dtype=float), where=b != 0)
    ratio_err = ratio * np.sqrt(ra**2 + rb**2)
    return ratio_hist, ratio_err


def list_top_dirs(upfile):
    out = []
    for k in upfile.keys():
        # keys look like 'Sample;1'
        name = k.split(";")[0]
        if name and name not in out:
            out.append(name)
    return out


def stack1d_histograms(up, output_dir, data_dir, mc_dirs):
    jobs = [(reg, var) for reg in REGION_LIST for var in variables_common[reg]]

    for region, var in jobs:
        hname = vardict[var]

        data_path = f"{data_dir}/{region}/{hname}"
        if data_path not in up:
            continue
        data_hist = ensure_weight_hist(Hist(up[data_path]))

        mc_hists = []
        for m in mc_dirs:
            p = f"{m}/{region}/{hname}"
            if p in up:
                h = ensure_weight_hist(Hist(up[p]))
                h.name = m
                mc_hists.append(h)

        if not mc_hists:
            continue

        dyn_w = max(11, int(1.5 * len(mc_hists)))
        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, figsize=(dyn_w, 12), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        fig.subplots_adjust(hspace=0.05)

        data_hist.plot(ax=ax, stack=False, histtype="errorbar", yerr=True, xerr=True, color="black", label="Data", flow="sum")
        Stack(*mc_hists).plot(ax=ax, stack=True, histtype="fill", flow="sum", sort="yield")

        mc_sum = sum_hist_list(mc_hists)
        ratio, rerr = get_ratio(data_hist, mc_sum)
        ratio.plot(ax=ax_ratio, histtype="errorbar", yerr=rerr, xerr=True, color="black", flow="sum")
        ax_ratio.axhline(1, linestyle="--", color="gray")
        ax_ratio.set_ylim(0, 3)
        ax_ratio.set_ylabel("Data / MC")
        ax_ratio.set_xlabel(var)

        ax.set_yscale("log")
        ax.set_ylim(0.1, 1e8)
        ax.set_ylabel("Events")
        hep.cms.label("", ax=ax, lumi=lumi_label(), loc=0, llabel="Work in progress", com=13.6)
        ax.legend(ncol=2, loc="upper right", fontsize=16)
        ax.set_xlabel("")

        outdir = os.path.join(output_dir, region)
        os.makedirs(outdir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{var}.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(outdir, f"{var}.png"), bbox_inches="tight")
        plt.close()
        print(f"[OK] {region}/{var}")


def main():
    ap = argparse.ArgumentParser(description="Plotter for pp->bbchichi histograms")
    ap.add_argument("--root", required=True, help="Path to ppbbchichi-histograms.root")
    ap.add_argument("--out", default="stack_plots_bbchichi", help="Output directory")
    ap.add_argument("--data", required=True, help="Top-level directory name in ROOT file for data (e.g. 'Data')")
    ap.add_argument("--mc", action="append", required=True, help="Top-level directory name(s) for MC (repeatable)")
    args = ap.parse_args()

    up = uproot.open(args.root)
    os.makedirs(args.out, exist_ok=True)
    stack1d_histograms(up, args.out, data_dir=args.data, mc_dirs=args.mc)


if __name__ == "__main__":
    main()
