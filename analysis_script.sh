#!/bin/bash

echo "Starting the Analyzer script"
echo "============================="

# pp -> bb chi chi analyzer (bb + MET)
# NOTE: Replace the -i paths with your local inputs.
python ppbbchichi_analyzer_par.py --year 2023 --era All \
  -i data/inputs/2023/preBPix/ \
  -i data/inputs/2023/postBPix/ \
  --tag bbchichi_CombinedAll

echo "Analyzer script completed."

echo "Outputs (example):"
echo "  outputfiles/merged/bbchichi_CombinedAll/ppbbchichi-trees.root"
echo "  outputfiles/merged/bbchichi_CombinedAll/ppbbchichi-histograms.root"

echo "If you want plots, run bbchichi_Plotter.py, e.g.:"
echo "  python bbchichi_Plotter.py --root <.../ppbbchichi-histograms.root> --out stack_plots_bbchichi --data <DataSampleName> --mc <MCSampleName>"
