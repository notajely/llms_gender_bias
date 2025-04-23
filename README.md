# 职业性别偏见分析项目 (Occupation Gender Bias Analysis)

本项目用于分析语言模型中的职业性别偏见，特别是研究GPT-2等语言模型在处理职业词汇时表现出的性别刻板印象。

## 项目概述

该项目分析了职业词嵌入与性别词嵌入之间的关系，使用以下方法：

1. 从BLS（美国劳工统计局）数据中提取职业的性别比例信息
2. 使用GPT-2模型计算职业词和性别词的嵌入表示
3. 计算职业词与性别词之间的余弦相似度
4. 分析职业词在性别轴上的投影
5. 可视化结果并计算与实际性别比例的相关性

## 项目结构

```
.
├── data/                  # 数据文件
│   ├── occupation_gender_data.csv  # 职业性别数据
│   ├── bls_cps_data.xlsx           # BLS CPS数据
│   └── ...
│
├── notebooks/             # Jupyter笔记本
│
├── results/               # 结果和可视化
│
├── src/                   # 源代码
│   ├── analysis/          # 分析相关脚本
│   ├── data_processing/   # 数据处理相关脚本
│   ├── experiments/       # 实验脚本
│   ├── models/            # 模型相关脚本
│   ├── utils/             # 工具函数
│   └── visualization/     # 可视化相关脚本
│
└── docs/                  # 文档
```

详细的源代码结构请参见 [src/README.md](src/README.md)。

## 功能特点

- 加载职业数据并计算嵌入表示
- 计算职业词与性别词之间的余弦相似度
- 创建多种可视化：
  - 职业词与性别词之间相似度的热图
  - 职业词性别投影分数的条形图
  - BLS性别比例与模型偏见之间相关性的散点图
- 支持完整的职业列表分析
- 提供多种实验设置和分析方法

## 环境要求

- Python 3.6+
- PyTorch
- Transformers
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- tqdm

可以使用以下命令安装所需的包：

```bash
pip install torch transformers pandas numpy matplotlib seaborn scikit-learn tqdm
```

## 使用方法

### 生成职业列表

```bash
python -m src.data_processing.generate_full_occupation_lists
```

### 运行职业性别分析

```bash
python -m src.analysis.occupation_gender_analysis
```

### 运行实验

```bash
python -m src.experiments.exp_1
```

## 数据流程

1. 从BLS CPS数据中提取职业性别比例信息
2. 使用`occupation_data.py`处理原始数据，生成带有性别标签的职业数据
3. 使用`generate_full_occupation_lists.py`生成分类的职业列表
4. 使用`occupation_gender_analysis.py`分析职业词与性别词的关系
5. 使用可视化脚本生成结果图表

## 输出结果

该项目在`results`目录中生成以下输出：

1. `occupation_gender_bias_results.csv`: 包含每个职业的相似度分数和偏见值的CSV文件
2. `occupation_gender_similarity_heatmap.png`: 职业词与性别词之间相似度的热图可视化
3. `occupation_gender_projection_barplot.png`: 职业词性别投影分数的条形图
4. `occupation_gender_correlation.png`: BLS性别比例与模型偏见之间相关性的散点图

## 方法论

该项目使用以下方法论：

1. **嵌入计算**:
   - 使用GPT-2为每个职业和性别词计算嵌入表示
   - 对于多词术语，使用所有标记嵌入的平均值（或最后一个标记嵌入，取决于策略）

2. **相似度计算**:
   - 计算每个职业嵌入与每个性别词嵌入之间的余弦相似度
   - 计算男性相似度为与'he'和'man'的相似度的平均值
   - 计算女性相似度为与'she'和'woman'的相似度的平均值
   - 计算性别偏见为男性相似度减去女性相似度

3. **可视化**:
   - 创建职业词与性别词之间相似度的热图
   - 创建职业词性别投影分数（偏见值）的条形图
   - 创建BLS性别比例与模型偏见之间相关性的散点图

## 注意事项

- 该脚本在可用时使用GPU，否则回退到CPU
- 处理所有职业可能需要一些时间，特别是在CPU上
- 可以通过修改可视化函数中的`num_occupations`参数来调整可视化中显示的职业数量
