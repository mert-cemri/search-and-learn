# Load PRM model
# prm_name = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
# prm_tokenizer = AutoTokenizer.from_pretrained(prm_name)
# prm_model = AutoModelForSequenceClassification.from_pretrained(
#     prm_name,
#     torch_dtype=torch.float16,
#     device_map=device
# )
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification
import torch
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from sal.utils.parser import H4ArgumentParser
from sal.config import Config
from sal.models.reward_models import load_prm
# parser = H4ArgumentParser(Config)
# config = parser.parse()
# scores = []
# prm = load_prm(config)      

import pandas as pd
import matplotlib.pyplot as plt
# Flatten any nested lists in the scores data to ensure all elements are numeric
def flatten_scores(scores):
    flat_scores = []
    for score in scores:
        if isinstance(score, list):
            flat_scores.extend(flatten_scores(score))
        else:
            flat_scores.append(score)
    return flat_scores

# Load scores from JSON files
file_paths = [
    "/home/mert/spec/search-and-learn/prm_double_head_sanity_check_scores_answers.json",
    "/home/mert/spec/search-and-learn/prm_mistral_sanity_check_scores_answers.json",
    "/home/mert/spec/search-and-learn/prm_sanity_check_scores_answers.json"
]

scores_data = []
for file_path in file_paths:
    with open(file_path, "r") as file:
        scores = json.load(file)
        scores_data.append(scores)
# print(f"scores_data: {len(scores_data)}")
# print(f"scores_data[0]: {len(scores_data[0])}")
# print(f"scores_data[1]: {len(scores_data[1])}")
# print(f"scores_data[2]: {len(scores_data[2])}")
# print(f"scores_data[0][1]: {scores_data[0][1]}")
# assert False
# Apply flattening to each scores list
for i in range(len(scores_data)):
    scores_data[i] = flatten_scores(scores_data[i])
# Create a DataFrame

# Normalize the scores individually for each dataset
def normalize_scores(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    return [(score - mean) / std for score in scores]

# Apply normalization to each scores list individually
for i in range(len(scores_data)):
    scores_data[i] = normalize_scores(scores_data[i])

print(f"Scores data 0 mean: {np.mean(scores_data[0])}")
print(f"Scores data 0 std: {np.std(scores_data[0])}")
print(f"Scores data 1 mean: {np.mean(scores_data[1])}")
print(f"Scores data 1 std: {np.std(scores_data[1])}")
print(f"Scores data 2 mean: {np.mean(scores_data[2])}")
print(f"Scores data 2 std: {np.std(scores_data[2])}")

df = pd.DataFrame({
    "double_head_scores": scores_data[0],
    "mistral_scores": scores_data[1],
    "sanity_check_scores": scores_data[2]
})


# Calculate correlation matrix
double_head_vs_mistral = df[['double_head_scores', 'mistral_scores']].corr()
double_head_vs_sanity_check = df[['double_head_scores', 'sanity_check_scores']].corr()
mistral_vs_sanity_check = df[['mistral_scores', 'sanity_check_scores']].corr()

print("Double Head vs Mistral Correlation matrix:")
print(double_head_vs_mistral)
print("Double Head vs Sanity Check Correlation matrix:")
print(double_head_vs_sanity_check)
print("Mistral vs Sanity Check Correlation matrix:")
print(mistral_vs_sanity_check)

# # Plot the correlation matrix
# plt.figure(figsize=(8, 6))
# plt.matshow(correlation_matrix, fignum=1)
# plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
# plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
# plt.colorbar()
# plt.title("Correlation Matrix of PRM Scores", pad=20)
# plt.show()

"/home/mert/spec/search-and-learn/prm_double_head_sanity_check_scores_answers.json"
"/home/mert/spec/search-and-learn/prm_mistral_sanity_check_scores_answers.json"
"/home/mert/spec/search-and-learn/prm_sanity_check_scores_answers.json"
# for i in tqdm(range(0, len(data))):
#     problem = data[i]['problem']
#     answer = data[i]['answer']
#     message = [
#     {'role': 'user', f'content': "problem: {problem}"},
#     {'role': 'assistant', 'content': f"answer: {answer}"}
#     ]
#     reward_tensor, reward = evaluate_reward(message, tokenizer, reward_model, device)
#     scores.append(reward)
#     with open("prm_double_head_sanity_check_scores_answers.json", "w") as file:
#         json.dump(scores, file)

# print(f"len(scores): {len(scores)}")
# scores = np.array(scores)

# print(f"Average PRM score: {scores.mean():.4f}")
# print(f"Std of PRM scores: {scores.std():.4f}")
# print(f"Min PRM score: {scores.min():.4f}")
# print(f"Max PRM score: {scores.max():.4f}")

"""
mistral answers
Average PRM score: 0.4014
Std of PRM scores: 0.1823
Min PRM score: 0.0411
Max PRM score: 0.9683
"""
"""
mistral solutions
Average PRM score: 0.6341
Std of PRM scores: 0.2134
Min PRM score: 0.1009
Max PRM score: 0.9927
"""
"""
answers
Average PRM score: 0.6477
Std of PRM scores: 0.3085
Min PRM score: 0.0293
Max PRM score: 1.0000
"""

"""
solutions
Average PRM score: 0.4813
Std of PRM scores: 0.3476
Min PRM score: 0.0028
Max PRM score: 1.0000
"""

"""
custom solutions
Average PRM score: -3.0555
Std of PRM scores: 1.9144
Min PRM score: -7.8320
Max PRM score: 3.8633
"""
"""
custom answers
Average PRM score: -7.1693
Std of PRM scores: 0.6661
Min PRM score: -8.9688
Max PRM score: -4.1719
"""