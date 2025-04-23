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
code = f"""# 根据 BLS 数据生成的职业列表 - 完整版
# 男性刻板印象职业 ({len(male_stereotyped_clean)}个)
male_stereotype_full = {male_stereotyped_clean}

# 女性刻板印象职业 ({len(female_stereotyped_clean)}个)
female_stereotype_full = {female_stereotyped_clean}

# 中性职业 ({len(neutral_clean)}个)
neutral_full = {neutral_clean}

# 为了便于使用，选择每个类别的前10个职业
male_stereotype = {male_stereotyped_clean[:10]}
female_stereotype = {female_stereotyped_clean[:10]}
neutral = {neutral_clean[:10]}

# 合并所有职业
professions = male_stereotype + female_stereotype + neutral
gender_terms = ['he', 'she', 'man', 'woman']
"""

# 保存到文件
with open('occupation_lists_full.py', 'w') as f:
    f.write(code)

print("\n完整的职业列表已保存到 occupation_lists_full.py")

# 打印每个类别的前10个和后10个职业，以便查看
print("\n男性刻板印象职业 (前10个):")
for i, occ in enumerate(male_stereotyped_clean[:10]):
    print(f"{i+1}. {occ}")

print("\n男性刻板印象职业 (后10个):")
for i, occ in enumerate(male_stereotyped_clean[-10:]):
    print(f"{len(male_stereotyped_clean)-9+i}. {occ}")

print("\n女性刻板印象职业 (前10个):")
for i, occ in enumerate(female_stereotyped_clean[:10]):
    print(f"{i+1}. {occ}")

print("\n女性刻板印象职业 (后10个):")
for i, occ in enumerate(female_stereotyped_clean[-10:]):
    print(f"{len(female_stereotyped_clean)-9+i}. {occ}")

print("\n中性职业 (前10个):")
for i, occ in enumerate(neutral_clean[:10]):
    print(f"{i+1}. {occ}")

print("\n中性职业 (后10个):")
for i, occ in enumerate(neutral_clean[-10:]):
    print(f"{len(neutral_clean)-9+i}. {occ}")
