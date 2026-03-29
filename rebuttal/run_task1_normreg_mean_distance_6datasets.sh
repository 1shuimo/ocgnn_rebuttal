#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for ds in Amazon tolokers tf_finace YelpChi-all reddit photo; do
  python rebuttal/ggad_labeledNormal_our_rebuttal_full.py \
    --dataset "$ds" \
    --teacher_path "../ggad_new_best_pth/${ds}_ggad_teacher_final.pth" \
    --use_normreg 1 \
    --base_ratio 0.2

  python rebuttal/ggad_labeledNormal_our_rebuttal_full.py \
    --dataset "$ds" \
    --teacher_path "../ggad_new_best_pth/${ds}_ggad_teacher_final.pth" \
    --use_normreg 0 \
    --base_ratio 0.2
done
