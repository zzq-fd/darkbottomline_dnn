"""Histogram binning definitions for the bbχχ analysis.

Purpose:
- Define per-variable binning used when filling ROOT histograms.
- Keep bin edges stable across regions for consistent plots/yields.
"""

import copy

# Binning for pp -> bb chi chi analysis
binning = {}

binning["preselection"] = {
    "dijet_mass": [36, 0, 360],
    "dijet_pt": [30, 0, 900],
    "dijet_eta": [24, -3.0, 3.0],
    "dijet_phi": [24, -3.14159, 3.14159],

    "lead_bjet_pt": [30, 0, 600],
    "sublead_bjet_pt": [30, 0, 600],
    "lead_bjet_eta": [24, -3.0, 3.0],
    "sublead_bjet_eta": [24, -3.0, 3.0],
    "lead_bjet_phi": [24, -3.14159, 3.14159],
    "sublead_bjet_phi": [24, -3.14159, 3.14159],

    "lead_bjet_PNetB": [20, 0.0, 1.0],
    "sublead_bjet_PNetB": [20, 0.0, 1.0],

    "puppiMET_pt": [40, 0, 800],
    "puppiMET_phi": [24, -3.14159, 3.14159],
    "DeltaPhi_j1MET": [32, 0.0, 3.14159],
    "DeltaPhi_j2MET": [32, 0.0, 3.14159],

    "n_jets": [15, 0, 15],
    "lepton1_pt": [30, 0, 300],
}

for r in ["srbbchichi", "cr_antibtag", "cr_lowmet", "cr_lepton"]:
    binning[r] = copy.deepcopy(binning["preselection"])
