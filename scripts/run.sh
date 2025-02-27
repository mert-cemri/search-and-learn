bash scripts_from_rsd/serve_draft_model.sh 
bash scripts_from_rsd/serve_target_model.sh 
bash scripts_from_rsd/serve_prm.sh 
python scripts/servered_test_time_compute.py recipes/Qwen-1B-Qwen-7B/online_speculative_beam_search.yaml --model_path=Qwen/Qwen2.5-Math-1.5B-Instruct --n=4 --gpu_memory_utilization=0.15 --max_model_len=10000 --dataset_end=20 --num_iterations=20 --rm_regularizer=0
# bash scripts_from_rsd/serve_custom_prm.sh