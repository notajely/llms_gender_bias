# 职业性别偏见分析项目

本项目用于分析语言模型中的职业性别偏见，特别是研究GPT-2等语言模型在处理职业词汇时表现出的性别刻板印象。

## 项目结构

```
src/
├── analysis/              # 分析相关脚本
│   ├── correlation_visualization.py    # 相关性可视化
│   ├── occupation_gender_analysis.py   # 职业性别分析主脚本
│   ├── raincloud_plot.py               # 雨云图绘制
│   ├── representational_distortion_analysis.py # 表征扭曲分析
│   └── stereotype_projection_analysis.py # 刻板印象投影分析
│
├── data_processing/       # 数据处理相关脚本
│   ├── bls_stereotype_processor.py     # BLS数据处理器
│   ├── generate_full_occupation_lists.py  # 生成完整职业列表
│   ├── occupation_data.py              # 职业数据处理
│   ├── occupation_lists_full.py        # 完整职业列表定义
│   └── update_occupation_lists.py      # 更新职业列表
│
├── experiments/           # 实验脚本
│   └── exp1.py                         # 实验1主脚本
│
├── models/                # 模型相关脚本
│   └── multi_word_embedding.py         # 多词嵌入模型
│
└── utils/                 # 工具函数
    ├── __init__.py
    └── logger.py                       # 日志工具
```

## 主要模块说明

### 分析模块 (analysis/)

- `occupation_gender_analysis.py`: 主要分析脚本，用于计算职业词与性别词之间的余弦相似度，并生成可视化结果。
- `correlation_visualization.py`: 用于可视化BLS性别比例与模型性别偏见之间的相关性。
- `stereotype_projection_analysis.py`: 分析职业词在性别轴上的投影，研究刻板印象。
- `representational_distortion_analysis.py`: 识别和分析模型中表征扭曲显著的职业词汇，计算扭曲分数并生成可视化结果。

### 数据处理模块 (data_processing/)

- `bls_stereotype_processor.py`: 处理BLS数据并生成职业性别刻板印象标签的类。
- `occupation_data.py`: 处理职业数据，将O*NET职业与BLS性别统计数据匹配。
- `generate_full_occupation_lists.py`: 从BLS数据生成完整的职业列表，按性别刻板印象分类。
- `update_occupation_lists.py`: 更新职业列表，生成简化版本用于notebook。
- `occupation_lists_full.py`: 包含完整的职业列表定义和访问函数。

### 实验模块 (experiments/)

- `exp1.py`: 实验1的主要脚本，包含WEAT测试和职业词嵌入分析。

### 模型模块 (models/)

- `multi_word_embedding.py`: 处理多词嵌入的模型，用于获取职业词和性别词的嵌入表示。

### 工具模块 (utils/)

- `logger.py`: 日志工具，用于记录实验过程和结果。

## 数据流程

1. 使用`bls_stereotype_processor.py`或`occupation_data.py`从BLS CPS数据中提取职业性别比例信息
2. 使用`generate_full_occupation_lists.py`生成分类的职业列表
3. 使用`occupation_gender_analysis.py`分析职业词与性别词的关系
4. 使用可视化脚本生成结果图表

## 使用方法

### 处理BLS数据并生成职业性别标签

```bash
python -m src.data_processing.occupation_data
```

或使用面向对象的方法：

```bash
python -m src.data_processing.bls_stereotype_processor
```

### 生成职业列表

```bash
python -m src.data_processing.generate_full_occupation_lists
```

### 运行职业性别分析

```bash
python -m src.analysis.occupation_gender_analysis
```

### 运行实验1

```bash
python -m src.experiments.exp1
```

### 运行表征扭曲分析

```bash
python -m src.analysis.representational_distortion_analysis
```

## 结果输出

分析结果将保存在`results/`目录中，包括：

1. `occupation_gender_bias_results.csv`: 包含每个职业的性别偏见分数
2. `distortion_analysis/significant_distortion_occupations.csv`: 包含表征扭曲显著的职业列表
3. 各种可视化图表，如相似度热图、投影条形图、相关性散点图和扭曲分析图

## 注意事项

- 确保在运行脚本前已安装所有必要的依赖
- 数据文件应放在`data/`目录下
- 结果将保存在`results/`目录中
- 对于大型数据集，某些分析可能需要较长时间
