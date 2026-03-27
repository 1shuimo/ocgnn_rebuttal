#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

datasets=(
  "Amazon"
  "tolokers"
  "tf_finace"
  "YelpChi-all"
)

for ds in "${datasets[@]}"; do
  python rebuttal/fp_fn_ratio_sweep.py \
    --dataset "${ds}" \
    --logits_path "../rebuttal_log/ggad_labeledNormal_normreg_compare/${ds}_with_normreg_best_logits.npy" \
    --labels_path "../rebuttal_log/ggad_labeledNormal_normreg_compare/${ds}_with_normreg_test_labels.npy" \
    --base_ratio 0.2
done
