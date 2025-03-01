import os
import shutil

def create_directory_structure(base_path):
    # 定义要创建的目录结构
    directories = [
        'data/raw',
        'data/processed',
        'experiments/gpt2',
        'experiments/other_models',
        'notebooks',
        'results/figures',
        'results/tables',
        'src/data_processing',
        'src/models',
        'src/evaluation',
        'tests'
    ]

    # 创建目录
    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f'Created directory: {full_path}')

    # 移动现有文件到新的位置
    file_moves = [
        ('bios_train.pkl', 'data/raw/bios_train.pkl'),
        ('bios_test.pkl', 'data/raw/bios_test.pkl'),
        ('bios_dev.pkl', 'data/raw/bios_dev.pkl'),
        ('gpt2_exp_1.py', 'experiments/gpt2/exp_1.py'),
        ('exp_1.ipynb', 'notebooks/exp_1.ipynb')
    ]

    for src, dst in file_moves:
        src_path = os.path.join(base_path, src)
        dst_path = os.path.join(base_path, dst)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f'Moved {src} to {dst}')

    # 创建基本文件
    files_to_create = {
        '.gitignore': '''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environment
venv/
env/
llm_env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
''',
        'requirements.txt': '''
torch
transformers
numpy
pandas
scikit-learn
matplotlib
tqdm
''',
        'README.md': '''
# Master Thesis: LLMs Bias Analysis

## Project Structure

```
master_thesis_llms_bias/
├── data/                       # 存放所有数据文件
│   ├── raw/                    # 原始数据
│   │   └── bios_*.pkl          # BIB数据集文件
│   └── processed/              # 处理后的数据
├── experiments/                # 实验代码
│   ├── gpt2/                   # GPT-2相关实验
│   │   ├── exp_1.py            # 实验1代码
│   │   └── ...                 # 其他实验
│   └── other_models/           # 其他模型实验
├── notebooks/                  # Jupyter笔记本
│   └── exp_1.ipynb             # 实验1笔记本
├── results/                    # 实验结果
│   ├── figures/                # 图表
│   └── tables/                 # 表格数据
├── src/                        # 源代码
│   ├── data_processing/        # 数据处理代码
│   ├── models/                 # 模型相关代码
│   └── evaluation/             # 评估指标代码
├── tests/                      # 测试代码
├── .gitignore                  # Git忽略文件
├── requirements.txt            # 项目依赖
└── README.md                   # 项目说明
```

## 项目说明

这个项目主要研究大型语言模型（LLMs）中的偏见问题，特别关注性别偏见。项目使用BIB数据集进行实验，通过多个实验来分析和量化模型中的偏见。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

项目使用BIB（Bias in Bios）数据集，该数据集包含职业传记文本，用于分析性别偏见。

## 实验

- exp_1：使用GPT-2模型分析职业词嵌入中的性别偏见
'''
    }

    for filename, content in files_to_create.items():
        file_path = os.path.join(base_path, filename)
        with open(file_path, 'w') as f:
            f.write(content.lstrip())
        print(f'Created file: {filename}')

if __name__ == '__main__':
    # 获取当前脚本所在目录的绝对路径
    base_path = os.path.dirname(os.path.abspath(__file__))
    create_directory_structure(base_path)
    print('Project structure setup completed!')