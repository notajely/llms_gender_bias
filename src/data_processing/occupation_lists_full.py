# 根据 BLS 数据生成的职业列表 - 完整版
# 此文件由 generate_full_occupation_lists.py 自动生成
# 包含所有职业的完整列表，按照性别刻板印象分类

# 导入必要的库
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 从生成的文件中导入职业列表
try:
    from occupation_lists_full import male_stereotype_full, female_stereotype_full, neutral_full
except ImportError:
    # 如果文件不存在，提供空列表
    male_stereotype_full = []
    female_stereotype_full = []
    neutral_full = []

# 为了便于使用，选择每个类别的前10个职业
male_stereotype = male_stereotype_full[:10] if male_stereotype_full else []
female_stereotype = female_stereotype_full[:10] if female_stereotype_full else []
neutral = neutral_full[:10] if neutral_full else []

# 合并所有职业
professions = male_stereotype + female_stereotype + neutral
gender_terms = ['he', 'she', 'man', 'woman']

# 提供获取职业列表的函数
def get_male_stereotyped_occupations(full=False):
    """获取男性刻板印象职业列表"""
    return male_stereotype_full if full else male_stereotype

def get_female_stereotyped_occupations(full=False):
    """获取女性刻板印象职业列表"""
    return female_stereotype_full if full else female_stereotype

def get_neutral_occupations(full=False):
    """获取中性职业列表"""
    return neutral_full if full else neutral

def get_all_occupations(full=False):
    """获取所有职业列表"""
    if full:
        return male_stereotype_full + female_stereotype_full + neutral_full
    else:
        return professions
