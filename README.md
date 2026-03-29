# OCGNN Rebuttal

这个仓库目前整理成了几类目录：

- `main/`: 主要实验入口
- `teachers/`: teacher 训练脚本
- `students/`: student / distillation / ablation 脚本
- `models/`: 模型定义
- `utils/`: 数据和日志工具
- `analysis/`: 画图和离线分析
- `archive/`: 旧脚本和临时脚本

更详细的脚本说明见 [REPO_EXPERIMENT_GUIDE.md](/Users/shuimo/Desktop/ocgnn_rebuttal/REPO_EXPERIMENT_GUIDE.md)。

## 目录约定

推荐服务器目录结构：

```text
parent_dir/
├── dataset/
│   └── Amazon.mat
├── ggad_new_best_pth/
│   └── Amazon_ggad_teacher_final.pth
├── rebuttal_log/
└── ocgnn_rebuttal/
```

当前代码约定：

- 数据集默认读取仓库同级目录：
  - `../dataset/<dataset>.mat`
- 实验日志默认写到仓库同级目录：
  - `../rebuttal_log/<log_subdir>/`
- teacher 权重路径通过 `--teacher_path` 显式传入

## 主实验

### GGAD with noise

入口脚本：

- `main/ggad_labeledNormal_our.py`

示例命令：

```bash
python main/ggad_labeledNormal_our.py \
  --dataset Amazon \
  --teacher_path ../ggad_new_best_pth/Amazon_ggad_teacher_final.pth
```

如果要自定义日志子目录：

```bash
python main/ggad_labeledNormal_our.py \
  --dataset Amazon \
  --teacher_path ../ggad_new_best_pth/Amazon_ggad_teacher_final.pth \
  --log_subdir amazon_seed0
```

### GGAD no noise

入口脚本：

- `main/ggad_labeledNormal_no_noise.py`

示例命令：

```bash
python main/ggad_labeledNormal_no_noise.py \
  --dataset Amazon \
  --teacher_path ../no_noise_pth/Amazon_best_model_0305.pth
```

## 其他主线

### OCGNN

- 主入口：`main/ocgnn_melt.py`
- teacher 训练：`teachers/tea_train_ocgnn_latest.py`

### DOMINANT

- 主入口：`main/dominant.py`
- teacher 训练：`teachers/tea_train_dominant.py`

### RHO

- RHO teacher：`teachers/tea_train_rho.py`
- RHO teacher（不读取官方 npz）：`teachers/tea_train_rho_no_npz.py`
- RHO student / distill：`rebuttal/rho_labeledNormal_our.py`

当前 `tea_train_rho.py` 的运行口径：

1. 默认参数按官方 `RHO` 仓库的 `run.sh` 分数据集设置
2. 默认使用官方风格 split
   - 先随机切 `train/val/test`
   - 再从 `train` 里的正常节点中取前 `50%` 作为 labeled normal
3. 默认优先读取仓库内 vendored 的官方预存 Lap
   - 位置：`rho_official/datasets/`
   - 当前现成有：`reddit / photo / elliptic / questions`
4. 如果某个数据集没有官方预存 Lap
   - 会自动回退到根据当前 `adj` 现场构造 Lap

RHO teacher 权重建议保存到仓库同级：

```text
../rho_teacher_best_pth/
```

先建目录：

```bash
mkdir -p ../rho_teacher_best_pth
```

示例命令：

```bash
python teachers/tea_train_rho.py \
  --dataset Amazon \
  --weight_save_path ../rho_teacher_best_pth/Amazon_rho_teacher_best.pth
```

```bash
python teachers/tea_train_rho.py \
  --dataset reddit \
  --weight_save_path ../rho_teacher_best_pth/reddit_rho_teacher_best.pth
```

```bash
python teachers/tea_train_rho.py \
  --dataset tolokers \
  --weight_save_path ../rho_teacher_best_pth/tolokers_rho_teacher_best.pth
```

```bash
python teachers/tea_train_rho.py \
  --dataset photo \
  --weight_save_path ../rho_teacher_best_pth/photo_rho_teacher_best.pth
```

如果想统一跑 `1000 epoch`：

```bash
for ds in Amazon reddit tolokers photo; do
  python teachers/tea_train_rho.py \
    --dataset "$ds" \
    --epochs 1000 \
    --weight_save_path "../rho_teacher_best_pth/${ds}_rho_teacher_best.pth"
done
```

日志位置：

```text
../rebuttal_log/tea_train_rho/
```

训练过程日志：

```text
../rebuttal_log/tea_train_rho/<dataset>_rho_teacher.txt
```

最佳结果摘要：

```text
../rebuttal_log/tea_train_rho/<dataset>_rho_teacher_best.txt
```

最佳摘要里会记录：
- `Best epoch`
- `Best AUROC`
- `Best AUPRC`
- `Weight path`

### RHO teacher（不使用官方 npz）

脚本：

- `teachers/tea_train_rho_no_npz.py`

运行口径：

1. 默认训练参数仍按官方 `run.sh` 分数据集
2. 默认仍然使用官方风格 split
3. 不读取任何官方预存 `Lap_matrix_*.npz`
4. 一律根据当前 `adj` 现场构造 Lap

默认权重保存到新的同级目录：

```text
../rho_teacher_best_pth_no_npz/
```

例如：

```bash
python teachers/tea_train_rho_no_npz.py \
  --dataset Amazon
```

默认会保存到：

```text
../rho_teacher_best_pth_no_npz/Amazon_rho_teacher_no_npz_best.pth
```

如果想统一跑 `1000 epoch`：

```bash
for ds in Amazon reddit tolokers photo; do
  python teachers/tea_train_rho_no_npz.py \
    --dataset "$ds" \
    --epochs 1000 \
    --weight_save_path "../rho_teacher_best_pth_no_npz/${ds}_rho_teacher_no_npz_best.pth"
done
```

日志位置：

```text
../rebuttal_log/tea_train_rho_no_npz/
```

### RHO student / distill

脚本：

- `rebuttal/rho_labeledNormal_our.py`

这条线的输入是已经训练好的 RHO teacher 权重，然后用 RHO teacher score 去蒸馏 OCGNN student。

如果 teacher 来自 `tea_train_rho.py`：

```bash
python rebuttal/rho_labeledNormal_our.py \
  --dataset Amazon \
  --teacher_path ../rho_teacher_best_pth/Amazon_rho_teacher_best.pth \
  --use_normreg 1
```

如果 teacher 来自 `tea_train_rho_no_npz.py`：

```bash
python rebuttal/rho_labeledNormal_our.py \
  --dataset Amazon \
  --teacher_path ../rho_teacher_best_pth_no_npz/Amazon_rho_teacher_no_npz_best.pth \
  --use_normreg 1
```

如果你训练 `no_npz teacher` 时手动把权重保存到了旧目录 `../rho_teacher_best_pth/`，那 student 也要按真实保存位置去读。例如你实际跑过的是：

```bash
for ds in Amazon reddit tolokers photo; do
  python teachers/tea_train_rho_no_npz.py \
    --dataset "$ds" \
    --epochs 1000 \
    --weight_save_path "../rho_teacher_best_pth/${ds}_rho_teacher_no_npz_best.pth"
done
```

那 student 对应应该写成：

```bash
for ds in Amazon reddit tolokers photo; do
  python rebuttal/rho_labeledNormal_our.py \
    --dataset "$ds" \
    --teacher_path "../rho_teacher_best_pth/${ds}_rho_teacher_no_npz_best.pth" \
    --use_normreg 1 \
    --num_epoch 1500
done
```

先做 smoke test 时，建议少跑几轮：

```bash
python rebuttal/rho_labeledNormal_our.py \
  --dataset Amazon \
  --teacher_path ../rho_teacher_best_pth_no_npz/Amazon_rho_teacher_no_npz_best.pth \
  --use_normreg 1 \
  --num_epoch 5
```

如果想跑 4 个数据集：

```bash
for ds in Amazon reddit tolokers photo; do
  python rebuttal/rho_labeledNormal_our.py \
    --dataset "$ds" \
    --teacher_path "../rho_teacher_best_pth_no_npz/${ds}_rho_teacher_no_npz_best.pth" \
    --use_normreg 1
done
```

日志位置：

```text
../rebuttal_log/rho_labeledNormal_our/
```

## Rebuttal 脚本

`rebuttal/` 目录下放的是这次 rebuttal 额外加的实验脚本。

### 1. NormReg 对比

脚本：

- `rebuttal/ggad_labeledNormal_our_normreg_compare.py`

作用：

1. 基于 `our` 主实验
2. 通过 `--use_normreg 1/0` 对比有无 NormReg
3. 自动输出：
   - `真实 abnormal center -> labeled normal center`
   - `真实 abnormal center -> all normal center`
4. 自动保存最佳 logits 和测试集 labels，供后续 FP/FN 分析使用

有 NormReg：

```bash
python rebuttal/ggad_labeledNormal_our_normreg_compare.py \
  --dataset Amazon \
  --teacher_path ../ggad_new_best_pth/Amazon_ggad_teacher_final.pth \
  --use_normreg 1
```

无 NormReg：

```bash
python rebuttal/ggad_labeledNormal_our_normreg_compare.py \
  --dataset Amazon \
  --teacher_path ../ggad_new_best_pth/Amazon_ggad_teacher_final.pth \
  --use_normreg 0
```

默认日志目录：

```text
../rebuttal_log/ggad_labeledNormal_normreg_compare/
```

### 2. FP / FN 比例阈值 sweep

脚本：

- `rebuttal/fp_fn_ratio_sweep.py`

作用：

1. 对主实验 GraphNC 的 logits 做离线分析
2. 默认统一比例阈值 `0.2`
3. 自动计算：
   - `0.1`
   - `0.2`
   - `0.3`
4. 输出：
   - TP / FP / TN / FN
   - Precision / Recall / F1
   - FP / FN 节点索引

示例命令：

```bash
python rebuttal/fp_fn_ratio_sweep.py \
  --dataset Amazon \
  --logits_path ../rebuttal_log/ggad_labeledNormal_normreg_compare/Amazon_with_normreg_best_logits.npy \
  --labels_path ../rebuttal_log/ggad_labeledNormal_normreg_compare/Amazon_with_normreg_test_labels.npy \
  --base_ratio 0.2
```

默认日志目录：

```text
../rebuttal_log/fp_fn_ratio_sweep/
```

## 日志示例

### 1. NormReg 对比日志

`rebuttal/ggad_labeledNormal_our_normreg_compare.py` 的日志会写到：

```text
../rebuttal_log/ggad_labeledNormal_normreg_compare/
```

例如 `Amazon_with_normreg.txt` 大致会长这样：

```text
Teacher baseline: Testing_last_ggad_ Amazon AUC: 0.8123
Teacher real_abnormal_center -> labeled_normal_center: 1.8421
Teacher real_abnormal_center -> all_normal_center: 1.6354

NormReg enabled: True
Epoch 0: Total Loss = 0.1432
MSE Loss = 0.1315
Reg2 MSE Loss = 1.1734
student_score: 0.4218
ggad_score: 0.3987
student_score_non_normalize: -0.2154
ggad_score_non_normalize: -0.1876
Testing Amazon AUC_student_mlp_s: 0.7312
Testing Amazon AUC_student_mlp_s_non_normalize: 0.7285
Testing Amazon AUC_student_ggad: 0.8044
Testing Amazon AUC_student_ggad_non_normalize: 0.8019
Testing Amazon AUC_student_minus_ggad: 0.6761
Testing Amazon AUC_student_minus_ggad_non_normalize: 0.6708
Student real_abnormal_center -> labeled_normal_center: 1.1248
Student real_abnormal_center -> all_normal_center: 0.9736
Testing AP_student_mlp_s: 0.4521
Testing AP_student_mlp_s_non_normalize: 0.4478
Testing AP_student_ggad: 0.5884
Testing AP_student_ggad_non_normalize: 0.5829
Testing AP_student_minus_ggad: 0.4016
Testing AP_student_minus_ggad_non_normalize: 0.3962
Total time is: 0.00
```

同时还会额外保存：

```text
../rebuttal_log/ggad_labeledNormal_normreg_compare/Amazon_with_normreg_best_logits.npy
../rebuttal_log/ggad_labeledNormal_normreg_compare/Amazon_with_normreg_test_labels.npy
../rebuttal_log/ggad_labeledNormal_normreg_compare/Amazon_with_normreg_best_metrics.txt
```

`best_metrics.txt` 大致如下：

```text
Best AUC_student_mlp_s: 0.7814
NormReg enabled: True
Teacher real_abnormal_center -> labeled_normal_center: 1.8421
Teacher real_abnormal_center -> all_normal_center: 1.6354
Best Student real_abnormal_center -> labeled_normal_center: 1.3928
Best Student real_abnormal_center -> all_normal_center: 1.2147
```

### 2. FP / FN sweep 日志

`rebuttal/fp_fn_ratio_sweep.py` 的输出日志大致如下：

```text
Dataset: Amazon
Logits path: ../rebuttal_log/ggad_labeledNormal_normreg_compare/Amazon_with_normreg_best_logits.npy
Labels path: ../rebuttal_log/ggad_labeledNormal_normreg_compare/Amazon_with_normreg_test_labels.npy
Base ratio: 0.2000

Ratio: 0.1000
Predicted abnormal count: 330
TP: 145
FP: 185
TN: 2140
FN: 260
Precision: 0.4394
Recall: 0.3580
F1: 0.3945
FP indices: [3, 8, 12, ...]
FN indices: [1, 5, 9, ...]
Top-ranked abnormal indices: [77, 103, 5, ...]

Ratio: 0.2000
Predicted abnormal count: 660
TP: 238
FP: 422
TN: 1903
FN: 167
Precision: 0.3606
Recall: 0.5877
F1: 0.4470
FP indices: [...]
FN indices: [...]
Top-ranked abnormal indices: [...]

Ratio: 0.3000
Predicted abnormal count: 990
TP: 301
FP: 689
TN: 1636
FN: 104
Precision: 0.3040
Recall: 0.7432
F1: 0.4315
FP indices: [...]
FN indices: [...]
Top-ranked abnormal indices: [...]
```

### 3. 四个优先数据集

当前优先跑：

1. `Amazon`
2. `tolokers`
3. `tf_finace`
4. `YelpChi-all`

例如 `tolokers`：

```bash
python rebuttal/ggad_labeledNormal_our_normreg_compare.py \
  --dataset tolokers \
  --teacher_path ../ggad_new_best_pth/tolokers_ggad_teacher_final.pth \
  --use_normreg 1
```

例如 `tf_finace`：

```bash
python rebuttal/ggad_labeledNormal_our_normreg_compare.py \
  --dataset tf_finace \
  --teacher_path ../ggad_new_best_pth/tf_finace_ggad_teacher_final.pth \
  --use_normreg 1
```

## 说明

- `main/ggad_labeledNormal_our.py` 使用：
  - `models/model.py`
  - `models/model_ocgnn.py`
  - `utils/utils_old.py`
- `main/ggad_labeledNormal_no_noise.py` 使用：
  - `models/model3.py`
  - `models/model_ocgnn.py`
  - `utils/utils.py`

`with noise` 和 `no noise` 的 teacher 权重不要混用。
