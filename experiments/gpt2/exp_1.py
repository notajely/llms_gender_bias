import pickle
import torch
import os
import json
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import random
from tqdm import tqdm
from src.utils import setup_logger

# 创建实验特定的日志记录器
logger = setup_logger('gpt2_bias_analysis')

# Get root directory
def get_project_root():
    """Get absolute path to project root"""
    current_path = os.path.abspath(__file__)
    while not os.path.exists(os.path.join(os.path.dirname(current_path), '.git')):
        current_path = os.path.dirname(current_path)
    return os.path.dirname(current_path)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
logger.info(f"Random seed set to {RANDOM_SEED}")

# Experiment configuration
CONFIG = {
    "dataset_path": os.path.join(get_project_root(), "data/raw/bios_train.pkl"),
    "results_dir": os.path.join(get_project_root(), "results"),
    "figures_dir": os.path.join(get_project_root(), "results/figures"),
    "model_name": "gpt2",
    "n_permutations": 10000,
}

# 创建实验ID（使用时间戳确保唯一性）
import datetime
experiment_id = f"exp1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
experiment_dir = os.path.join(CONFIG["results_dir"], experiment_id)
experiment_figures_dir = os.path.join(experiment_dir, "figures")

# 更新配置
CONFIG["experiment_id"] = experiment_id
CONFIG["experiment_dir"] = experiment_dir
CONFIG["experiment_figures_dir"] = experiment_figures_dir

# Create result directories
os.makedirs(CONFIG["results_dir"], exist_ok=True)
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(experiment_figures_dir, exist_ok=True)

# Define device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Function to load dataset
def load_dataset(path):
    logger.info(f"开始加载数据集: {path}")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            data = pd.DataFrame(data)
        logger.info(f"数据集加载成功，形状: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"数据集文件未找到: {path}")
        exit(1)
    except Exception as e:
        logger.error(f"加载数据集时出错: {str(e)}", exc_info=True)
        exit(1)

# Function to get word embedding
def get_word_embedding(word, tokenizer, model, device):
    logger.debug(f"提取词嵌入: {word}")
    inputs = tokenizer(word, return_tensors="pt").to(device)
    outputs = model(**inputs)
    word_embedding = outputs.last_hidden_state[:, 0, :].mean(dim=0)
    return word_embedding.detach().cpu().numpy()

# Function to calculate similarity scores
def calculate_similarity_scores(profession_embeddings, gender_embeddings):
    logger.info("开始计算职业词和性别词之间的余弦相似度")
    similarity_scores = {}
    for profession, profession_emb in profession_embeddings.items():
        similarity_scores[profession] = {}
        logger.debug(f"处理职业词: {profession}")
        for gender_term, gender_emb in gender_embeddings.items():
            similarity = cosine_similarity(
                np.array([profession_emb]), np.array([gender_emb]))[0][0]
            similarity_scores[profession][gender_term] = similarity
            logger.debug(f"  '{profession}' 与 '{gender_term}' 的相似度: {similarity:.4f}")
    logger.info("相似度计算完成")
    return similarity_scores

# Function to visualize embeddings with PCA
def visualize_embeddings_pca(profession_embeddings, save_path=None):
    profession_names = list(profession_embeddings.keys())
    embeddings_for_pca = np.array([profession_embeddings[name]
                                  for name in profession_names])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_for_pca)

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1],
                c=range(len(profession_names)), cmap='viridis')
    for i, profession in enumerate(profession_names):
        plt.annotate(profession, xy=(
            pca_result[i, 0], pca_result[i, 1]), fontsize=8)
    plt.title('PCA Projection of Profession Word Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Profession Index')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"PCA visualization saved to {save_path}")
    
    plt.show()

# Function to calculate association score
def calculate_s(word, A, B, embeddings):
    emb_w = embeddings[word]
    mean_a = np.mean([embeddings[a] for a in A], axis=0)
    mean_b = np.mean([embeddings[b] for b in B], axis=0)
    return cosine_similarity(np.array([emb_w]), np.array([mean_a]))[0][0] - cosine_similarity(np.array([emb_w]), np.array([mean_b]))[0][0]

# Function to calculate WEAT score
def weat_score(target_words_1, target_words_2, attribute_words_1, attribute_words_2, embeddings):
    logger.info("开始计算WEAT分数")
    logger.debug(f"目标词集1大小: {len(target_words_1)}, 目标词集2大小: {len(target_words_2)}")
    logger.debug(f"属性词集1大小: {len(attribute_words_1)}, 属性词集2大小: {len(attribute_words_2)}")
    
    S_diffs = [calculate_s(word, attribute_words_1, attribute_words_2, embeddings)
               for word in target_words_1 + target_words_2]
    mean_diff_targets = np.mean([calculate_s(word, attribute_words_1, attribute_words_2, embeddings)
                                 for word in target_words_1])
    std_dev_all_targets = np.std(S_diffs)
    
    if std_dev_all_targets == 0:
        logger.warning("标准差为0，返回WEAT分数0.0")
        return 0.0
    else:
        weat_score = (mean_diff_targets - np.mean([calculate_s(word, attribute_words_1, attribute_words_2, embeddings)
                                             for word in target_words_2])) / std_dev_all_targets
        logger.info(f"WEAT分数计算完成: {weat_score:.4f}")
        return weat_score

# Function to run permutation test
def permutation_test(target_words_1, target_words_2, attribute_words_1, attribute_words_2, embeddings, n_permutations=10000):
    logger.info(f"开始排列检验，执行{n_permutations}次排列")
    observed_effect_size = weat_score(
        target_words_1, target_words_2, attribute_words_1, attribute_words_2, embeddings)
    logger.info(f"观察到的效应量: {observed_effect_size:.4f}")
    
    combined_words = target_words_1 + target_words_2
    num_greater = 0
    for i in tqdm(range(n_permutations), desc="Permutation Test"):
        if i % 1000 == 0:
            logger.debug(f"完成{i}次排列")
        shuffled = combined_words.copy()
        random.shuffle(shuffled)
        permuted_target_1 = shuffled[:len(target_words_1)]
        permuted_target_2 = shuffled[len(target_words_1):]
        permuted_effect_size = weat_score(
            permuted_target_1, permuted_target_2, attribute_words_1, attribute_words_2, embeddings)
        if permuted_effect_size >= observed_effect_size:
            num_greater += 1
    
    p_value = num_greater / n_permutations
    logger.info(f"排列检验完成，p值: {p_value:.4f}")
    return p_value, observed_effect_size

# Function to save results
def save_results(similarity_scores, effect_size, p_value, word_scores, config, 
                profession_list, gender_terms, target_male, target_female, 
                attribute_male, attribute_female, weat_results=None):
    logger.info("开始保存实验结果")
    # 保存实验参数和结果
    results = {
        "similarity_scores": similarity_scores,
        "weat_effect_size": effect_size,
        "p_value": p_value,
        "word_association_scores": word_scores,
        "config": {k: v for k, v in config.items() if k != "experiment_figures_dir" and k != "experiment_dir"},  # 移除路径对象
        "parameters": {
            "profession_list": profession_list,
            "gender_terms": gender_terms,
            "target_professions_male": target_male,
            "target_professions_female": target_female,
            "attribute_male_terms": attribute_male,
            "attribute_female_terms": attribute_female
        }
    }
    
    # 如果有额外的WEAT结果，也保存
    if weat_results:
        # 确保weat_results中的所有值都是可序列化的
        weat_results_json = {}
        for k, v in weat_results.items():
            if isinstance(v, (np.integer, np.floating)):
                weat_results_json[k] = float(v)
            elif isinstance(v, (list, np.ndarray)):
                weat_results_json[k] = [str(x) if not isinstance(x, (str, int, float, bool, type(None))) else float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v]
            else:
                weat_results_json[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None), dict, list)) else v
        results["weat_detailed_results"] = weat_results_json
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for k, v in results.items():
        if k == "similarity_scores":
            results_json[k] = {prof: {gender: float(sim) for gender, sim in scores.items()} 
                              for prof, scores in v.items()}
        elif k == "word_association_scores":
            results_json[k] = {word: float(score) for word, score in v.items()}
        elif k == "parameters":
            # 确保参数中的所有列表都是可序列化的
            params_json = {}
            for param_key, param_value in v.items():
                if isinstance(param_value, list):
                    params_json[param_key] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item for item in param_value]
                else:
                    params_json[param_key] = param_value
            results_json[k] = params_json
        elif k == "config":
            # 确保配置中的所有值都是可序列化的
            config_json = {}
            for config_key, config_value in v.items():
                if isinstance(config_value, (np.integer, np.floating)):
                    config_json[config_key] = float(config_value)
                elif isinstance(config_value, (list, np.ndarray)):
                    config_json[config_key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in config_value]
                else:
                    config_json[config_key] = str(config_value) if not isinstance(config_value, (str, int, float, bool, type(None))) else config_value
            results_json[k] = config_json
        elif k == "weat_detailed_results":
            # 这部分已经在上面处理过了
            results_json[k] = v
        elif isinstance(v, np.ndarray):
            results_json[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            results_json[k] = float(v)
        else:
            # 处理其他可能的非JSON可序列化类型
            results_json[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None), dict, list)) else v
    
    # 保存到实验特定目录
    results_file = os.path.join(config["experiment_dir"], "results.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"结果已保存到 {results_file}")
    except TypeError as e:
        logger.error(f"JSON序列化错误: {str(e)}", exc_info=True)
        # 尝试更严格的类型转换 - 不要重新导入np
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
        
        # 完全转换
        fully_converted = convert_to_serializable(results_json)
        with open(results_file, 'w') as f:
            json.dump(fully_converted, f, indent=2)
        logger.info(f"结果已保存到 {results_file} （完全转换后）")

def main():
    # Load dataset
    try:
        bios_df = load_dataset(CONFIG["dataset_path"])
        print(f"Loaded {len(bios_df)} records from dataset")
    except Exception as e:
        # Fallback to local path if relative path fails
        print(f"Failed to load from relative path: {e}")
        print("Trying to load from current directory...")
        bios_df = load_dataset("bios_train.pkl")
        print(f"Loaded {len(bios_df)} records from dataset")

    # Define profession and gender terms
    profession_list_from_readme = [
        "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
        "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
        "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
        "photographer", "physician", "poet", "professor", "psychologist", "rapper",
        "software_engineer", "surgeon", "teacher", "yoga_teacher"
    ]
    gender_terms = ["he", "she", "man", "woman", "person", "individual", "male", "female"]

    # Load GPT-2 tokenizer and model
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model = model.to(device)

    # Get word embeddings
    print("Generating word embeddings...")
    profession_embeddings = {
        profession: get_word_embedding(profession, tokenizer, model, device)
        for profession in profession_list_from_readme
    }
    gender_embeddings = {
        gender: get_word_embedding(gender, tokenizer, model, device)
        for gender in gender_terms
    }

    # Calculate similarity scores
    similarity_scores = calculate_similarity_scores(profession_embeddings, gender_embeddings)

    # 修改可视化调用
    pca_save_path = os.path.join(CONFIG["experiment_figures_dir"], "profession_embeddings_pca.png")
    visualize_embeddings_pca(profession_embeddings, pca_save_path)

    # Define stereotype profession lists
    target_professions_male_stereotype = [
        "architect", "attorney", "dentist", "physician", "software_engineer",
        "surgeon", "composer", "filmmaker", "photographer", "rapper"
    ]
    target_professions_female_stereotype = [
        "nurse", "teacher", "dietitian", "yoga_teacher", "interior_designer",
        "model", "paralegal", "personal_trainer"
    ]
    attribute_male_terms = ["he", "man", "male"]
    attribute_female_terms = ["she", "woman", "female"]

    # Validate profession lists
    valid_target_professions_male = [
        p for p in target_professions_male_stereotype if p in profession_embeddings]
    valid_target_professions_female = [
        p for p in target_professions_female_stereotype if p in profession_embeddings]

    # Prepare embeddings for WEAT test
    weat_embeddings = {}
    for word in valid_target_professions_male + valid_target_professions_female + attribute_male_terms + attribute_female_terms:
        if word in profession_embeddings:
            weat_embeddings[word] = profession_embeddings[word]
        elif word in gender_embeddings:
            weat_embeddings[word] = gender_embeddings[word]
        else:
            print(f"Warning: {word} not found in any embedding dictionary.")

    # Run WEAT test
    if (len(valid_target_professions_male) > 0 and len(valid_target_professions_female) > 0 and
            len(attribute_male_terms) > 0 and len(attribute_female_terms) > 0):
        
        print("\nRunning WEAT test...")
        p_value, effect_size = permutation_test(
            valid_target_professions_male, 
            valid_target_professions_female,
            attribute_male_terms, 
            attribute_female_terms, 
            weat_embeddings,
            CONFIG["n_permutations"]
        )
        
        print(f"\nWEAT Test Effect Size (Male vs. Female Stereotype Professions): {effect_size:.4f}")
        print(f"P-value: {p_value:.4f}")

        # Calculate individual word association scores
        print("\nIndividual word association scores:")
        word_scores = {}
        for word in valid_target_professions_male + valid_target_professions_female:
            s = calculate_s(word, attribute_male_terms, attribute_female_terms, weat_embeddings)
            word_scores[word] = s
            print(f"  {word}: {s:.4f}")
        
        # 收集WEAT详细结果
        weat_detailed = {
            "effect_size": effect_size,
            "p_value": p_value,
            "valid_target_professions_male": valid_target_professions_male,
            "valid_target_professions_female": valid_target_professions_female
        }
        
        # 保存结果，包括所有参数
        save_results(
            similarity_scores, 
            effect_size, 
            p_value, 
            word_scores, 
            CONFIG,
            profession_list_from_readme,
            gender_terms,
            target_professions_male_stereotype,
            target_professions_female_stereotype,
            attribute_male_terms,
            attribute_female_terms,
            weat_detailed
        )
    else:
        print("\nWarning: Incomplete word sets for WEAT test. Please check that target professions and attribute words are included in the embedding dictionary.")
        print("Valid male stereotype professions:", valid_target_professions_male)
        print("Valid female stereotype professions:", valid_target_professions_female)

    print("\nExperiment 1 code execution completed.")

if __name__ == "__main__":
    main()
