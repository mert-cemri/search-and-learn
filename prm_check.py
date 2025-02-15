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
device = 'cuda:5'
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from sal.utils.parser import H4ArgumentParser
from sal.config import Config
from sal.models.reward_models import load_prm
parser = H4ArgumentParser(Config)
config = parser.parse()
scores = []
prm = load_prm(config)      

file_path = "/home/mert/spec/search-and-learn/data_iclr/beam_search/AMead10Llama-3.2-3B-Instruct-AWQ-_n4_b0.1_q500_period0_model_len10000_1.jsonl"
# Read the JSON file
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]

for i in tqdm(range(0, len(data))):
    problem = data[i]['problem']
    solution = data[i]['solution']
    score = prm.score([problem], [[solution]])[0][0]
    if isinstance(score, (np.ndarray, list)):
            score = float(score[0]) 
    # print(f"score: {score}")
    # assert False
    scores.append(score)
    with open("prm_mistral_sanity_check_scores.json", "w") as file:
        json.dump(scores, file)

print(f"len(scores): {len(scores)}")
scores = np.array(scores)

print(f"Average PRM score: {scores.mean():.4f}")
print(f"Std of PRM scores: {scores.std():.4f}")
print(f"Min PRM score: {scores.min():.4f}")
print(f"Max PRM score: {scores.max():.4f}")

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
