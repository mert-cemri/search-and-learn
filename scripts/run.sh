CUDA_DEVICES=("4" "5" "6")
SCRIPTS=("serve_draft_model.sh" "serve_prm.sh" "serve_target_model.sh")

echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
TOTAL_MEMORY=$(nvidia-smi -i ${CUDA_DEVICES[0]} --query-gpu=memory.total --format=csv,noheader,nounits)
echo "Total memory of GPU ${CUDA_DEVICES[0]}: $TOTAL_MEMORY MiB"

HALF_TOTAL_MEMORY=$((TOTAL_MEMORY / 2))

for i in "${!CUDA_DEVICES[@]}"; do
    echo "Starting script ${SCRIPTS[$i]} on GPU ${CUDA_DEVICES[$i]}"
    CUDA_DEVICE=${CUDA_DEVICES[$i]}
    bash scripts/${SCRIPTS[$i]} $CUDA_DEVICE &
    while [ "$(nvidia-smi -i $CUDA_DEVICE --query-gpu=memory.used --format=csv,noheader,nounits)" -lt "$HALF_TOTAL_MEMORY" ]; do
        sleep 1
    done
done

wait
# bash scripts_from_rsd/serve_custom_prm.sh