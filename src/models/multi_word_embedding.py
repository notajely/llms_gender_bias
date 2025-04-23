import torch
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载 GPT-2 模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

def get_embedding(word_or_phrase, method='mean'):
    """
    获取单词或短语的 embedding
    
    参数：
    - word_or_phrase: 要获取 embedding 的单词或短语
    - method: 处理多词短语的方法
      - 'mean': 使用整个短语的 token 序列，然后取平均
      - 'last': 使用整个短语的最后一个 token 的 embedding
    
    返回：
    - embedding: numpy 数组，表示单词或短语的 embedding
    """
    # 添加特殊 token
    inputs = tokenizer(word_or_phrase, return_tensors="pt")
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state
    
    if method == 'mean':
        # 取所有 token 的平均值
        embedding = last_hidden_states.mean(dim=1).squeeze().numpy()
    elif method == 'last':
        # 取最后一个 token 的 embedding
        embedding = last_hidden_states[0, -1, :].numpy()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return embedding

def get_word_similarity(word1, word2, method='mean'):
    """计算两个单词或短语之间的余弦相似度"""
    emb1 = get_embedding(word1, method)
    emb2 = get_embedding(word2, method)
    
    # 计算余弦相似度
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
    return sim[0][0]

def get_occupation_gender_bias(occupation, gender_terms=['he', 'she', 'man', 'woman']):
    """
    计算职业与性别词之间的偏差
    
    返回：
    - male_sim: 与男性词的平均相似度
    - female_sim: 与女性词的平均相似度
    - bias: 男性相似度 - 女性相似度（正值表示男性偏向，负值表示女性偏向）
    """
    # 男性词和女性词
    male_terms = [gender_terms[0], gender_terms[2]]  # he, man
    female_terms = [gender_terms[1], gender_terms[3]]  # she, woman
    
    # 计算与男性词的相似度
    male_similarities = [get_word_similarity(occupation, term) for term in male_terms]
    male_sim = np.mean(male_similarities)
    
    # 计算与女性词的相似度
    female_similarities = [get_word_similarity(occupation, term) for term in female_terms]
    female_sim = np.mean(female_similarities)
    
    # 计算偏差（正值表示男性偏向，负值表示女性偏向）
    bias = male_sim - female_sim
    
    return male_sim, female_sim, bias

def main():
    # 示例：计算一些职业的性别偏差
    occupations = [
        "engineer",
        "software_engineer",
        "nurse",
        "teacher",
        "doctor",
        "programmer"
    ]
    
    print("职业\t男性相似度\t女性相似度\t偏差")
    print("-" * 50)
    
    for occupation in occupations:
        male_sim, female_sim, bias = get_occupation_gender_bias(occupation)
        print(f"{occupation}\t{male_sim:.4f}\t{female_sim:.4f}\t{bias:.4f}")
    
    # 示例：处理多词职业
    multi_word_occupations = [
        "software engineer",
        "software_engineer",
        "medical doctor",
        "medical_doctor"
    ]
    
    print("\n多词职业处理示例:")
    print("职业\t男性相似度\t女性相似度\t偏差")
    print("-" * 50)
    
    for occupation in multi_word_occupations:
        male_sim, female_sim, bias = get_occupation_gender_bias(occupation)
        print(f"{occupation}\t{male_sim:.4f}\t{female_sim:.4f}\t{bias:.4f}")

if __name__ == "__main__":
    main()
