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
  "reddit"
  "photo"
)

for ds in "${datasets[@]}"; do
  python rebuttal/ggad_teacher_distance.py \
    --dataset "${ds}" \
    --teacher_path "../ggad_new_best_pth/${ds}_ggad_teacher_final.pth"

  python rebuttal/fp_fn_ratio_sweep.py \
    --dataset "${ds}_teacher" \
    --logits_path "../rebuttal_log/ggad_teacher_distance/${ds}_teacher_best_logits.npy" \
    --labels_path "../rebuttal_log/ggad_teacher_distance/${ds}_teacher_test_labels.npy" \
    --base_ratio 0.2 \
    --log_subdir "fp_fn_ratio_sweep_teacher"
done
