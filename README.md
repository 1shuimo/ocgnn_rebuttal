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
