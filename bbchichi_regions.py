"""Region selections (masks) for the bbχχ analysis.

Purpose:
- Provide boolean masks for preselection, SR, and control regions.
- Centralize region definitions used by the analyzer.
"""

import numpy as np


def _delta_phi(phi1, phi2):
    d = np.abs(phi1 - phi2)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def get_mask_preselection(ev):
    # Minimal bb + MET preselection (tunable in code if needed)
    return (
        (ev.lead_bjet_pt > 30.0)
        & (ev.sublead_bjet_pt > 30.0)
        & (np.abs(ev.lead_bjet_eta) < 2.4)
        & (np.abs(ev.sublead_bjet_eta) < 2.4)
        & (ev.puppiMET_pt > 100.0)
    )


def get_mask_srbbchichi(ev, met_thr=200.0, btag_thr=0.2605, dphi_thr=0.5):
    return (
        (ev.puppiMET_pt > met_thr)
        & (ev.lead_bjet_PNetB > btag_thr)
        & (ev.sublead_bjet_PNetB > btag_thr)
        & (ev.DeltaPhi_j1MET > dphi_thr)
        & (ev.DeltaPhi_j2MET > dphi_thr)
    )


def get_mask_cr_antibtag(ev, met_thr=200.0, btag_thr=0.2605, dphi_thr=0.5):
    return (
        (ev.puppiMET_pt > met_thr)
        & (
            (ev.lead_bjet_PNetB < btag_thr)
            | (ev.sublead_bjet_PNetB < btag_thr)
        )
        & (ev.DeltaPhi_j1MET > dphi_thr)
        & (ev.DeltaPhi_j2MET > dphi_thr)
    )


def get_mask_cr_lowmet(ev, met_lo=100.0, met_hi=200.0, btag_thr=0.2605):
    return (
        (ev.puppiMET_pt >= met_lo)
        & (ev.puppiMET_pt < met_hi)
        & (ev.lead_bjet_PNetB > btag_thr)
        & (ev.sublead_bjet_PNetB > btag_thr)
    )


def get_mask_cr_lepton(ev, met_thr=200.0, lep_pt_thr=10.0):
    # If lepton1_pt is missing upstream, it should be filled with -9999.
    return (ev.puppiMET_pt > met_thr) & (ev.lepton1_pt > lep_pt_thr)
