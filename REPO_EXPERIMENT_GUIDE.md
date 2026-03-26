# Experiment Guide

这个仓库里的脚本很多，但目前能看出主线其实是 3 组：

1. `GGAD`
2. `OCGNN`
3. `DOMINANT`

建议你优先只看下面这些文件。

## 1. 你现在的主实验

### With noise 版本

- 入口脚本: `main/ggad_labeledNormal_our.py`
- 你现在常用的命令:

```bash
python main/ggad_labeledNormal_our.py --dataset Amazon --teacher_path ggad_new_best_pth/Amazon_ggad_teacher_final.pth
```

- 依赖:
  - `models/model.py`
  - `models/model_ocgnn.py`
  - `utils/utils_old.py`

### No noise 版本

- 入口脚本: `main/ggad_labeledNormal_no_noise.py`
- 依赖:
  - `models/model3.py`
  - `models/model_ocgnn.py`
  - `utils/utils.py`

示例命令:

```bash
python main/ggad_labeledNormal_no_noise.py --dataset Amazon --teacher_path <no_noise_teacher_path>
```

## 2. `with noise` 和 `no noise` 的核心区别

不是只有主脚本名字不同，实际上换了整套 teacher 实现。

### `with noise`

- `main/ggad_labeledNormal_our.py` 导入 `from models.model import Model_ggad`
- `models/model.py` 里会在 `forward(...)` 中对 `emb_abnormal` 加噪声:

```python
noise = torch.randn(emb_abnormal.size()) * args.var + args.mean
emb_abnormal = emb_abnormal + noise
```

### `no noise`

- `main/ggad_labeledNormal_no_noise.py` 导入 `from models.model3 import Model_ggad`
- `models/model3.py` 里对应噪声逻辑是注释掉的，不会加到 `emb_abnormal`

## 3. Teacher 权重不要混用

这点最重要。

- `main/ggad_labeledNormal_our.py` 需要加载 `models/model.py` 定义出来的 GGAD teacher 权重
- `main/ggad_labeledNormal_no_noise.py` 需要加载 `models/model3.py` 定义出来的 GGAD teacher 权重

也就是说：

- `with noise` 的 teacher 和 `no noise` 的 teacher 不建议混用
- 因为 `forward` 签名和内部模块并不完全一样

## 4. GGAD 相关脚本怎么分

### 建议保留并优先看

- `main/ggad_labeledNormal_our.py`
- `main/ggad_labeledNormal_no_noise.py`
- `models/model.py`
- `models/model3.py`
- `models/model_ocgnn.py`
- `utils/utils_old.py`
- `utils/utils.py`

### GGAD teacher 训练相关

- `teachers/tea_train_ggad_latest.py`
  - GGAD teacher 训练入口
  - 依赖 `model.py` + `utils_old.py`
- `teachers/with_noise_teacher.py`
  - 更像单独整理出的 with-noise teacher 训练版本
  - 保存到 `./with_noise_pth/`
- `teachers/no_noise_teacher.py`
  - no-noise teacher 训练版本
  - 保存到 `./no_noise_pth/`
- `teachers/ggad.py`
  - 早期 GGAD teacher 训练脚本
  - 会直接保存成 `{dataset}_ggad_teacher_final.pth`

### GGAD student / distill / ablation 分支

这些大多像实验分支，命名对应不同损失或蒸馏策略：

- `students/ggad_melt.py`
- `students/ggad_melt_abnormal.py`
- `students/ggad_melt_mse_top.py`
- `students/ggad_one_reg.py`
- `students/ggad_only_distill.py`
- `students/ggad_main_only_distill.py`
- `students/ggad_unify.py`
- `students/ggad_find_np.py`
- `students/ggad_find_np_only_distill.py`
- `students/ggad_0.5reg_dis.py`
- `students/2-step-ggad-emb-norm.py`
- `students/only_distill_ggad_figure_1.py`
- `students/save_student_ggad_main.py`
- `students/save_student_ggad_melt.py`

还有一批 `stu_train_ggad_*`，本质上也是 student 训练的不同 loss 版本：

- `students/stu_train_ggad_reg.py`
- `students/stu_train_ggad_reg_kl.py`
- `students/stu_train_ggad_hard_label.py`
- `students/stu_train_ggad_2_reg.py`
- `students/stu_train_ggad_2_reg_mse.py`
- `students/stu_train_ggad_2_reg_data_enhance.py`
- `students/stu_train_ggad_2_reg_data_enhance_mse.py`
- `students/stu_train_ggad_score_distillation.py`
- `students/stu_train_ggad_score_emb.py`
- `students/stu_train_ggad_emb_norm.py`
- `students/stu_train_ggad_abnormal_new.py`
- `students/stu_train_ggad_0.01reg.py`

如果你现在只是复现实验，不建议先看这一大批。

## 5. OCGNN 主线

### 建议先看

- `teachers/tea_train_ocgnn_latest.py`
  - OCGNN teacher 训练
- `main/ocgnn_melt.py`
  - OCGNN student / distill 主脚本
- `models/model_ocgnn.py`
- `utils/utils.py`

### OCGNN 其他 student 变体

- `students/stu_train_ocgnn_norm_emb.py`
- `students/stu_train_ocgnn_reg2_mse.py`
- `students/stu_train_ocgnn_score_emb.py`

## 6. DOMINANT 主线

### 建议先看

- `teachers/tea_train_dominant.py`
  - DOMINANT teacher 训练
  - 用的是 `model_dominant_official.py`
- `main/dominant.py`
  - 比较像早期 baseline / 单独跑 DOMINANT 的脚本
  - 用的是 `model_dominant.py`
- `students/stu_train_dominant_data_enhance.py`
  - DOMINANT student / distill 分支

### DOMINANT 相关模型

- `models/model_dominant.py`
- `models/model_dominant_official.py`
- `models/model_dominant_edge.py`

### 其他变体

- `teachers/dominant_official.py`
- `archive/dominant_origin.py`
- `students/dominant_unify.py`
- `teachers/tea_train_dominant_edge_no_dgl.py`
- `students/add_dominant_loss.py`

## 7. 画图和分析脚本

这些一般不是训练主线：

- `analysis/draw_ggad.py`
- `analysis/draw_ocgnn.py`
- `analysis/draw_dominant.py`
- `analysis/draw_pdf_ggad.py`
- `analysis/draw_tsne.py`
- `analysis/tea_tnse.py`
- `analysis/draw_auc.py`
- `archive/draw_auc-Copy1.py`
- `analysis/final_draw.py`
- `analysis/teacher_find_fp.py`
- `analysis/score_forth.py`
- `analysis/single_mlp_forth.py`
- `analysis/double_mlp_forth.py`
- `analysis/emb_add_mlp_forth.py`

## 8. 明显可以先忽略的文件

### 很像临时文件

- `archive/untitled.py`
- `archive/untitled1.py`

### 很像历史拷贝 / 老脚本

- `archive/draw_auc-Copy1.py`
- `archive/run3.py`
- `archive/test.py`

## 9. `utils.py` 和 `utils_old.py` 也不是一回事

这里也要小心。

- `main/ggad_labeledNormal_our.py` 用的是 `utils/utils_old.py`
- `main/ggad_labeledNormal_no_noise.py` 用的是 `utils/utils.py`

`utils.py` 和 `utils_old.py` 在这些地方存在差异：

- `normalize_adj` 实现不同
- `load_mat(...)` 里 labeled normal 的采样比例不同
- `utils.py` 里还有 `load_mat_10` / `load_mat_100`
- 部分画图输出格式也不同

所以不要随手把 `utils_old.py` 替换成 `utils.py`。

## 10. 如果你现在只想最小集合复现

### GGAD with noise

看这 4 个文件就够：

- `main/ggad_labeledNormal_our.py`
- `models/model.py`
- `models/model_ocgnn.py`
- `utils/utils_old.py`

### GGAD no noise

看这 4 个文件就够：

- `main/ggad_labeledNormal_no_noise.py`
- `models/model3.py`
- `models/model_ocgnn.py`
- `utils/utils.py`

### OCGNN

- `teachers/tea_train_ocgnn_latest.py`
- `main/ocgnn_melt.py`
- `models/model_ocgnn.py`
- `utils/utils.py`

### DOMINANT

- `teachers/tea_train_dominant.py`
- `main/dominant.py`
- `models/model_dominant*.py`
- `utils/utils.py`

## 11. 我对当前仓库的建议

现在目录已经按下面的结构整理好了：

- `main/`
  - 放 3 个主入口
- `teachers/`
  - 放 `tea_train_*` 和 `*_teacher.py`
- `students/`
  - 放 `stu_train_*`、`*_melt.py`、`*_distill.py`
- `models/`
  - 放 `model*.py`
- `analysis/`
  - 放 `draw_*`、`tsne`、`find_fp`
- `archive/`
  - 放 `untitled.py`、`run3.py`、`draw_auc-Copy1.py`

如果你要，我下一步可以继续做两件事里的一个：

1. 直接帮你把这些文件真正分目录整理，并顺手修正 import
2. 只保留你现在会用到的主实验，给你生成一份更短的 `README`
