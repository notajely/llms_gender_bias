import pandas as pd
import re
import os

def clean_occupation_name(name):
    """
    清理职业名称：
    1. 转换为小写
    2. 删除引号
    3. 将空格替换为下划线
    4. 删除特殊字符
    """
    # 转换为小写
    name = name.lower()
    # 删除引号
    name = name.replace('"', '').replace("'", "")
    # 将空格、逗号、句号、连字符等替换为下划线
    name = re.sub(r'[\s,.\-/&]', '_', name)
    # 删除括号及其内容
    name = re.sub(r'\(.*?\)', '', name)
    # 删除其他特殊字符
    name = re.sub(r'[^\w_]', '', name)
    # 替换多个连续的下划线为单个下划线
    name = re.sub(r'_+', '_', name)
    # 删除开头和结尾的下划线
    name = name.strip('_')
    return name

# 加载处理后的职业数据
# 使用相对路径，确保可以从不同目录运行
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_path = os.path.join(project_root, 'data', 'occupation_gender_data.csv')
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# 排除 unknown 类别
df = df[~df['bls_label'].str.contains('unknown')]

# 提取各类别的职业
male_stereotyped = df[df['bls_label'] == 'male-stereotyped']['occupation'].tolist()
female_stereotyped = df[df['bls_label'] == 'female-stereotyped']['occupation'].tolist()
neutral = df[df['bls_label'] == 'neutral']['occupation'].tolist()

# 清理职业名称
male_stereotyped_clean = [clean_occupation_name(name) for name in male_stereotyped]
female_stereotyped_clean = [clean_occupation_name(name) for name in female_stereotyped]
neutral_clean = [clean_occupation_name(name) for name in neutral]

# 打印结果
print(f"Male-stereotyped occupations: {len(male_stereotyped_clean)}")
print(f"Female-stereotyped occupations: {len(female_stereotyped_clean)}")
print(f"Neutral occupations: {len(neutral_clean)}")

# 生成 Python 代码
code = f"""# 根据 BLS 数据生成的职业列表
male_stereotype = {male_stereotyped_clean[:10]}
female_stereotype = {female_stereotyped_clean[:10]}
neutral = {neutral_clean[:10]}

# 合并所有职业
professions = male_stereotype + female_stereotype + neutral
gender_terms = ['he', 'she', 'man', 'woman']
"""

print("\n可以直接复制到 notebook 的代码片段:")
print(code)

# 保存到文件
with open('occupation_lists.py', 'w') as f:
    f.write(code)

print("\n代码已保存到 occupation_lists.py")
