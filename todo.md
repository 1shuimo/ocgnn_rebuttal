# TODO

## 已确认的总体设置

### 数据集优先级

优先跑 4 个：

1. Amazon
2. Tolokers
3. T-Finance
4. YelpChi-All

如果时间允许，再补：

5. Reddit
6. Photo

也就是说，能做的话最好直接把 6 个数据集都跑掉。

### 任务 1 的两个对比口径

#### 情况 1

只考虑 `有 NormReg` 和 `无 NormReg`。

这里的“有无 NormReg”不是：

- [main/ggad_labeledNormal_no_noise.py](/Users/shuimo/Desktop/ocgnn_rebuttal/main/ggad_labeledNormal_no_noise.py)

而是指主实验 loss 里是否带正则项。

目前你的定义是：

- `无 NormReg`:
  - `Loss = Loss1`
- `有 NormReg`:
  - `Loss = Loss1 + Loss2`
  - 其中 `Loss2` 是 NormReg

#### 情况 2

当情况 1 里发现：

- `有 NormReg` 的距离没有比 `无 NormReg` 更大
- 或者几乎不变

则进一步比较：

1. `ScoreDA + NormReg`
2. `都没有`

其中：

- `ScoreDA + NormReg` = 当前主实验
- `都没有` = teacher 模型本身

### 任务 1 最终要报什么

你现在希望两种距离都算，但更偏向：

- `异常中心 -> 正常中心`

最终至少要固定输出：

1. `真实 abnormal center -> labeled normal center`
2. `真实 abnormal center -> all normal center`

这个要求在两种情况里都要做：

1. `有 / 无 NormReg`
2. `ScoreDA+NormReg / teacher`

### 任务 2 的阈值定义

你确认这里沿用以前 `find_np` 的设置，也就是：

- 比例阈值

所以 `+-0.1` 也是对比例阈值做：

1. 原阈值
2. 原阈值 - 0.1
3. 原阈值 + 0.1

你还提到一个重要点：

- 之前比例阈值那套好像主要只看了 Amazon 和 Tolokers
- 这次剩下 4 个数据集也都要补上

因此任务 2 需要扩成：

1. 原阈值的 4 个数据集
2. `-0.1` 的 6 个数据集
3. `+0.1` 的 6 个数据集

总之，这次需要把 6 个数据集都覆盖进去。

### 任务 2 分析哪个模型

只分析主实验，也就是：

- GraphNC

并且：

1. FP 要算
2. FN 也要算

### 任务 3 的目标版本

任务 3 不是单纯问 “BCE only 好不好”，而是要做下面这个版本：

1. 先训练一个 teacher
2. 用 teacher 分数 + 一个阈值，给节点打 hard label
3. 用这些 hard label 对 student 做监督学习 BCE
4. 明确不要再把 GGAD 生成的伪异常放进去

你也确认了：

- hard label 阈值先固定 `0.8` 就可以
- 效果不需要特别强，最好比我们主方法差一点

## 1. 任务 1：异常中心到正常中心距离

### 我目前的想法

这部分先做成分析输出，不先改 loss。

第一版建议直接挂到主实验和对应对照实验里，固定统计：

1. `真实 abnormal center -> labeled normal center`
2. `真实 abnormal center -> all normal center`

如果需要补充，再额外输出：

1. 每个真实 abnormal 到 labeled normal center 的平均距离
2. 每个真实 abnormal 到 all normal center 的平均距离

但主表述还是以 center-to-center 为主。

### 现有代码里已经确认的点

- [main/ggad_labeledNormal_our.py](/Users/shuimo/Desktop/ocgnn_rebuttal/main/ggad_labeledNormal_our.py)
  - 可以取到 `normal_label_idx`
  - 可以构造 `teacher_hard`
  - 可以构造 `all_normal_idx`
- [students/ggad_find_np.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/ggad_find_np.py)
  - 当前很多约束最终实际用的是 `normal_label_idx`
  - 不是 `all_normal_idx`
- [students/ggad_0.5reg_dis.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/ggad_0.5reg_dis.py)
  - 有现成距离统计
  - 但目前统计的是 `normal -> normal center`

### 现在剩下的疑惑

1. “无 NormReg”具体基于哪一个脚本来跑最合适？
   - 是直接从主实验拷一个 `no_normreg` 版本
   - 还是仓库里已经有对应分支
2. 你说的“teacher 本身”这里：
   - 是直接用 teacher embedding 去算
   - 还是用 teacher score 先分出 abnormal / normal 再基于 embedding 算
   - 我现在理解成后者，但这点最好执行前再确认一次

### 我建议的实施步骤

1. 先从主实验复制一个 `no_normreg` 版本
2. 在两边都加统一距离分析函数
3. 先跑 4 个优先数据集
4. 如果结果不够，再补 teacher 对比和另外 2 个数据集

## 2. 任务 2：FP / FN 比例阈值 sweep

### 我目前的想法

这部分不要继续改训练脚本，直接做离线分析脚本。

输入：

1. GraphNC 的 logits
2. labels
3. 原比例阈值

输出：

1. FP count
2. FN count
3. 可选的 FP/FN 节点索引

然后统一做：

1. 原阈值
2. 原阈值 - 0.1
3. 原阈值 + 0.1

### 现有代码里已经确认的点

- [students/ggad_find_np.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/ggad_find_np.py)
  - 当前是按比例阈值分
  - 不是固定 score threshold

### 现在剩下的疑惑

1. 之前各数据集对应的“原比例阈值”是不是统一一个值？
   - 还是 Amazon / Tolokers 用过特殊值
2. 你说“原阈值的 4 个 + 6 + 6”
   - 我现在理解成：
     - 原设定只补 4 个
     - `-0.1` 跑 6 个
     - `+0.1` 跑 6 个
   - 但这个统计口径后面最好再明确成表格

### 我建议的实施步骤

1. 先把各数据集原比例阈值整理出来
2. 新建离线脚本：
   - `analysis/fp_fn_ratio_sweep.py`
3. 先只支持 GraphNC
4. 跑：
   - 原阈值
   - 原阈值 - 0.1
   - 原阈值 + 0.1

## 3. 任务 3：teacher hard label -> student BCE，且不要伪异常

### 我目前的想法

这部分最适合新开一个干净脚本。

最接近的现成基线是：

- [students/stu_train_ggad_hard_label.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/stu_train_ggad_hard_label.py)

因为它已经是：

1. teacher 出分数
2. teacher 分数转 hard label
3. student 在原始节点上做 BCE
4. 没有把伪异常拼进 BCE

所以最稳的做法是：

1. 从它复制一份
2. 命名更明确
3. 把阈值、日志、输出清理一下

### 现有代码里已经确认的点

- [students/stu_train_ggad_hard_label.py](/Users/shuimo/Desktop/ocgnn_rebuttal/students/stu_train_ggad_hard_label.py)
  - 当前 hard label 阈值就是 `0.8`
  - BCE 不含伪异常
- 一些 `2_reg` / `data_enhance` 系列才是把伪异常拼进去了

### 现在剩下的疑惑

1. 你说“最好比我们的差”
   - 我理解成这个实验只需要作为 baseline，不用再额外调很多超参数
2. hard label 是不是就直接基于 teacher score 的 sigmoid 后结果做 `threshold=0.8`
   - 目前 `stu_train_ggad_hard_label.py` 就是这样

### 我建议的实施步骤

1. 新建：
   - `students/stu_train_ggad_teacher_bce_no_pseudo.py`
2. 从 `stu_train_ggad_hard_label.py` 复制
3. 保留：
   - teacher score
   - threshold=0.8
   - student BCE
4. 去掉或避免所有 pseudo abnormal 相关逻辑
5. 先跑 4 个优先数据集

## 4. 新增实验：RHO

### 目标

在 RHO 上跑 2 个数据集，尽量挑两个能体现提升的数据集。

### 我目前查到的信息

RHO 仓库主页是：

- [mala-lab/RHO](https://github.com/mala-lab/RHO)

README 里写的是：

1. 这是一个半监督图异常检测方法
2. 官方运行方式包括：
   - `sh run.sh`
   - `python reproduction.py --dataset name`
3. 仓库里有：
   - `datasets/`
   - `train.py`
   - `reproduction.py`
   - `checkpoint/`

这是我根据它的 GitHub README 做的归纳。[来源](https://github.com/mala-lab/RHO)

### 我目前的想法

先不要一上来就做全量迁移，先做一个最小验证：

1. 选 2 个数据集
2. 先确认 RHO 仓库是否原生支持
3. 如果支持，先复现它自己的结果
4. 再考虑把我们的设置或方法接进去

### 现在剩下的疑惑

1. 你想放到 RHO 上的是：
   - 我们的 student 训练策略
   - 我们的正则项
   - 还是整个 GraphNC 框架思路
2. “最好能放两个提升的”
   - 这个我可以先根据数据集特性和现有结果猜
   - 但最终还是得跑一下 baseline 才能确认

### 你需要补充/确认什么

1. 你想优先试哪两个数据集
   - Amazon / Tolokers 是我当前最优先的猜测
2. 你希望迁移到 RHO 的具体成分是什么

### 我建议的实施步骤

1. 先 clone / 检查 RHO
2. 确认它支持的数据集
3. 先选两个最可能有提升的数据集
4. 先复现 RHO baseline
5. 再决定把我们的哪一部分接进去

## 我下一步建议直接做的事情

1. 新建 `students/stu_train_ggad_teacher_bce_no_pseudo.py`
2. 新建 `analysis/fp_fn_ratio_sweep.py`
3. 给主实验复制一个 `no_normreg` 版本，用于任务 1 对比
4. 最后再看 RHO 接入
