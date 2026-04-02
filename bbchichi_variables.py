"""Common variable and region definitions for the bbχχ analysis.

Purpose:
- Define region names.
- Map analysis variable keys to on-disk branch/histogram names.
- Provide per-region variable lists used by the analyzer and plotter.
"""

import copy

# Regions for pp -> bb chi chi (bb + MET)
regions = [
    "preselection",
    "srbbchichi",
    "cr_antibtag",
    "cr_lowmet",
    "cr_lepton",
]

# On-disk histogram names (kept flat and stable)
vardict = {
    # dijet
    "dijet_mass": "dijet_mass",
    "dijet_pt": "dijet_pt",
    "dijet_eta": "dijet_eta",
    "dijet_phi": "dijet_phi",

    # b-jets
    "lead_bjet_pt": "lead_bjet_pt",
    "lead_bjet_eta": "lead_bjet_eta",
    "lead_bjet_phi": "lead_bjet_phi",
    "sublead_bjet_pt": "sublead_bjet_pt",
    "sublead_bjet_eta": "sublead_bjet_eta",
    "sublead_bjet_phi": "sublead_bjet_phi",
    "lead_bjet_PNetB": "lead_bjet_PNetB",
    "sublead_bjet_PNetB": "sublead_bjet_PNetB",

    # MET
    "puppiMET_pt": "puppiMET_pt",
    "puppiMET_phi": "puppiMET_phi",
    "DeltaPhi_j1MET": "DeltaPhi_j1MET",
    "DeltaPhi_j2MET": "DeltaPhi_j2MET",

    # event-level
    "n_jets": "n_jets",
    "lepton1_pt": "lepton1_pt",
}

_base_vars = [
    "dijet_mass",
    "dijet_pt",
    "dijet_eta",
    "dijet_phi",
    "lead_bjet_pt",
    "sublead_bjet_pt",
    "lead_bjet_eta",
    "sublead_bjet_eta",
    "lead_bjet_phi",
    "sublead_bjet_phi",
    "lead_bjet_PNetB",
    "sublead_bjet_PNetB",
    "puppiMET_pt",
    "puppiMET_phi",
    "DeltaPhi_j1MET",
    "DeltaPhi_j2MET",
    "n_jets",
    "lepton1_pt",
]

variables_common = {"preselection": copy.deepcopy(_base_vars)}
for r in regions:
    variables_common[r] = copy.deepcopy(_base_vars)
