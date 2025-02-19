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
parser = H4ArgumentParser(Config)
config = parser.parse()
scores = []
prm = load_prm(config)      

file_path = "/home/mert/spec/search-and-learn/data_iclr/beam_search/AMead10Llama-3.2-3B-Instruct-AWQ-_n4_b0.1_q500_period0_model_len10000_1.jsonl"
# Read the JSON file
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]

# for i in tqdm(range(0, len(data))):
#     problem = data[i]['problem']
#     solution = data[i]['solution']
#     score = prm.score([problem], [[solution]])[0][0]
#     if isinstance(score, (np.ndarray, list)):
#             score = float(score[0]) 
#     # print(f"score: {score}")
#     # assert False
#     scores.append(score)
#     with open("prm_mistral_sanity_check_scores.json", "w") as file:
#         json.dump(scores, file)
def evaluate_reward(message, tokenizer, reward_model, device='cpu'):
    """
    Evaluate reward for a given message using the reward model.
    
    Args:
        message (list): List of message dictionaries with 'role' and 'content' keys
        tokenizer: The tokenizer to use
        reward_model: The reward model to use
        device (str): Device to run inference on
        
    Returns:
        tuple: (reward_tensor, reward) containing the raw tensor and scalar reward value
    """
    message_template = tokenizer.apply_chat_template(message, tokenize=False)
    
    kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
    tokens = tokenizer.encode_plus(message_template, **kwargs)

    with torch.no_grad():
        reward_tensor = reward_model(
            tokens["input_ids"][0].view(1,-1).to(device), 
            attention_mask=tokens["attention_mask"][0].view(1,-1).to(device)
        )[0]
        reward = reward_tensor.cpu().detach().item()
    
    return reward_tensor, reward

device = 'cuda:0'


model_name_or_path = "/home/mert/spec/mergekit/GRM-Llama3--crime"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
reward_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16, 
                device_map=device,
                )
for i in tqdm(range(0, len(data))):
    problem = data[i]['problem']
    answer = data[i]['answer']
    message = [
    {'role': 'user', f'content': "problem: {problem}"},
    {'role': 'assistant', 'content': f"answer: {answer}"}
    ]
    reward_tensor, reward = evaluate_reward(message, tokenizer, reward_model, device)
    scores.append(reward)
    with open("prm_double_head_sanity_check_scores_answers.json", "w") as file:
        json.dump(scores, file)

print(f"len(scores): {len(scores)}")
scores = np.array(scores)

print(f"Average PRM score: {scores.mean():.4f}")
print(f"Std of PRM scores: {scores.std():.4f}")
print(f"Min PRM score: {scores.min():.4f}")
print(f"Max PRM score: {scores.max():.4f}")


"""
qwen (merged) math answers
0.6660591802299023
0.14266165240892736
"""
"""
qwen (merged) math solutions
0.8356026131510734
0.11564540849264689
"""
"""
mistral answers
Average PRM score: 0.4014
Std of PRM scores: 0.1823
"""
"""
mistral solutions
Average PRM score: 0.6341
Std of PRM scores: 0.2134
"""
"""
deepseek answers
Average PRM score: 0.6477
Std of PRM scores: 0.3085
"""

"""
deepseek solutions
Average PRM score: 0.4813
Std of PRM scores: 0.3476
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