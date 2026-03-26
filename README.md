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
