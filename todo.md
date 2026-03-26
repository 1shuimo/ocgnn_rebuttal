# TODO

## 1. 计算四个数据集“异常中心到正常中心”的距离

### 我目前的想法

这件事我倾向于先做成“分析输出”，不要先揉进训练逻辑。

第一版建议这样做：

1. 先固定在 GGAD 这条线做
2. 先算 teacher embedding 上的距离
3. 同时统计两种 normal center：
   - `labeled normal center`
   - `all normal center`
4. 同时统计两种 abnormal 集合：
   - `teacher_hard == 1`
   - `ground-truth anomaly`

这样最后每个实验版本至少会有 4 组距离：

1. abnormal center -> labeled normal center
2. abnormal center -> all normal center
3. mean(abnormal_i -> labeled normal center)
4. mean(abnormal_i -> all normal center)

然后再按你的要求做两层比较：

1. `有 NormReg` vs `无 NormReg`
2. 如果差异不明显，再比较：
   - `ScoreDA + NormReg`
   - `teacher 本身`

### 现有代码里我已经确认的点

- [main/ggad_labeledNormal_our.py](/Users/shuimo/Desktop/ocgnn_rebuttal/main/ggad_labeledNormal_our.py)
  - 能构造 `teacher_hard`
  - 能构造 `all_normal_idx`
  - 有 `normal_label_idx`
- [students/ggad_find_np.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/ggad_find_np.py)
  - 虽然也算了 `all_normal_idx`
  - 但真正用于特征增强和约束的是 `normal_label_idx`
- [students/ggad_0.5reg_dis.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/ggad_0.5reg_dis.py)
  - 已经有“normal 到 normal center”的统计
  - 但还不是你要的 abnormal-center 分析

### 我现在的疑惑

1. 你说的“四个数据集”具体是哪四个？
   - Amazon / reddit / tf_finace / elliptic？
   - 还是你有别的四个固定数据集
2. “有无 NormReg”你希望对比的具体脚本是哪两个？
   - `main/ggad_labeledNormal_our.py` vs `main/ggad_labeledNormal_no_noise.py`
   - 还是你指的是某两条 student 分支
3. “同时有 ScoreDA+NormReg 和同时没有这两个”你对应的具体实验脚本是哪些？
   - 我现在能理解成：
     - 有：当前主实验线
     - 没有：teacher 本身
   - 但这点还需要你确认
4. 距离最终你想看哪一种作为主指标？
   - center-to-center distance
   - 还是每个异常点到 normal center 的平均距离

### 你需要补充/确认什么

1. 四个数据集名字
2. “有 NormReg / 无 NormReg”分别对应哪个脚本
3. “有 ScoreDA+NormReg / 都没有”分别对应哪个脚本
4. 最终论文里你更想报哪一个距离指标

### 我建议的实施步骤

1. 你先把上面 4 个点确认掉
2. 我先把距离统计函数加到：
   - [main/ggad_labeledNormal_our.py](/Users/shuimo/Desktop/ocgnn_rebuttal/main/ggad_labeledNormal_our.py)
   - [main/ggad_labeledNormal_no_noise.py](/Users/shuimo/Desktop/ocgnn_rebuttal/main/ggad_labeledNormal_no_noise.py)
3. 如果 teacher 也要对比，再补到 teacher 脚本
4. 最后统一输出到 `rebuttal_log`

## 2. false positive / false negative 的阈值，以及在这个基础上 ±0.1

### 我目前的想法

这件事我不建议直接继续改训练脚本，而是建议做一个独立离线分析脚本。

因为我现在确认下来，`find_np` 当前不是固定数值阈值，而是：

1. 把测试集 logits 排序
2. 固定取 top 20% 判成异常

也就是当前真正起作用的是：

- `abnormal_ratio = 0.2`

所以“+-0.1”这里有两种可能解释：

1. 改比例阈值：
   - `0.1 / 0.2 / 0.3`
2. 改 score cutoff：
   - 先找到 top-20% 对应的 cutoff score
   - 再算 `cutoff-0.1 / cutoff / cutoff+0.1`

我目前更倾向于两种都做，因为它们回答的问题不同。

### 现有代码里我已经确认的点

- [students/ggad_find_np.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/ggad_find_np.py)
  - 当前 FP/FN 逻辑是固定 `abnormal_ratio = 0.2`
  - 不是固定 score threshold
- [students/ggad_find_np_only_distill.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/ggad_find_np_only_distill.py)
  - 同一类逻辑
- [students/stu_train_ggad_hard_label.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/stu_train_ggad_hard_label.py)
  - 有 `threshold = 0.8`
  - 但这是 teacher hard label 的监督阈值
  - 不是 `find_np` 用来做 FP/FN 的阈值

### 我现在的疑惑

1. 你说的“这个阈值”到底是指哪一个？
   - `find_np` 当前的 `abnormal_ratio = 0.2`
   - 还是你脑子里想的是一个 score threshold
2. 你要的最终结果是：
   - 只看 FP / FN 数量
   - 还是还要节点索引、Precision、Recall、F1
3. 你要分析的是：
   - teacher
   - student
   - only distill
   - 还是三者都做

### 你需要补充/确认什么

1. `+-0.1` 你想加减的是：
   - 比例阈值
   - 还是 score threshold
2. 你要分析哪些模型输出：
   - teacher
   - student
   - only distill
3. 最终你需要报什么：
   - 只报 FP/FN count
   - 还是也报 Precision / Recall / F1

### 我建议的实施步骤

1. 你先确认“+-0.1”对应的阈值定义
2. 我新建一个离线分析脚本：
   - `analysis/fp_fn_threshold_sweep.py`
3. 读取现有 `best_logits.npy` 和 `test_labels.npy`
4. 输出：
   - FP / FN count
   - 可选的节点索引
   - Precision / Recall / F1

## 3. 先用 GGAD 打 normal / abnormal 标签，再对 student 做 BCE；去掉伪异常

### 我目前的想法

这件事我觉得最容易先落成一个干净 baseline。

因为现有代码里已经有一版非常接近你要的东西：

- [students/stu_train_ggad_hard_label.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/stu_train_ggad_hard_label.py)

它当前就是：

1. teacher 出分数
2. teacher 分数转 hard label
3. student 只在原始节点上输出 score
4. BCE 只在原始节点上算

也就是说，这一版本身没有把伪异常拼进 BCE。

所以我现在的想法不是去复杂脚本里删逻辑，而是：

1. 以 `stu_train_ggad_hard_label.py` 为底
2. 整理成一个更明确的新脚本
3. 第一版先保持最干净：
   - teacher hard label
   - student BCE
   - no pseudo abnormal
   - no extra reg

### 现有代码里我已经确认的点

- [students/stu_train_ggad_hard_label.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/stu_train_ggad_hard_label.py)
  - 当前 BCE 不含伪异常
- 下面这些脚本才是把伪异常拼进去一起做 BCE / label concat 的：
  - [students/stu_train_ggad_2_reg_mse.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/stu_train_ggad_2_reg_mse.py)
  - [students/stu_train_ggad_0.01reg.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/stu_train_ggad_0.01reg.py)
  - [students/stu_train_ggad_2_reg_data_enhance.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/stu_train_ggad_2_reg_data_enhance.py)

### 我现在的疑惑

1. hard label 你想继续用固定 `threshold=0.8`，还是改成可调参数？
2. label 是不是只针对原始图所有节点？
   - 还是只针对测试集 / 某个子集
3. 第一版你要不要保留任何额外项？
   - `reg2_mse`
   - augmentation
   - 还是严格 `BCE only`

### 你需要补充/确认什么

1. 第一版是否就做 `BCE only`
2. hard label 阈值是否先用 `0.8`
3. 是否只监督原始节点的全部节点

### 我建议的实施步骤

1. 新建：
   - `students/stu_train_ggad_teacher_bce_no_pseudo.py`
2. 从 `stu_train_ggad_hard_label.py` 拷一份
3. 把 threshold 变成命令行参数
4. 把日志路径、输出命名整理干净
5. 第一版先跑 `BCE only`
6. 后面你如果需要，再加：
   - `reg2_mse`
   - top-percent hard label

## 我建议的总体执行顺序

1. 先做任务 3
   - 最容易快速形成一个干净 baseline
2. 再做任务 2
   - 属于离线分析，独立性高
3. 最后做任务 1
   - 因为它依赖你先明确实验对照组

## 我现在最需要你补充的 8 个点

1. 四个数据集具体是哪四个
2. 任务 1 里“有 NormReg / 无 NormReg”具体对应哪两个脚本
3. 任务 1 里“有 ScoreDA+NormReg / 都没有”具体对应哪两个脚本
4. 任务 1 你最终更想报 center-to-center 还是 abnormal-to-center mean
5. 任务 2 里的“+-0.1”是针对比例阈值还是 score threshold
6. 任务 2 需要分析 teacher / student / only-distill 中的哪些
7. 任务 3 第一版是否就做 `BCE only`
8. 任务 3 hard label 阈值是否先固定 `0.8`
