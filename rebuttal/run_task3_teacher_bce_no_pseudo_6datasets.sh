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
  python rebuttal/ggad_teacher_bce_no_pseudo.py \
    --dataset "${ds}" \
    --teacher_path "../ggad_new_best_pth/${ds}_ggad_teacher_final.pth" \
    --hard_label_ratio 0.2
done
