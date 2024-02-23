set -ex

# model="7b"
# data_path="predictor_data/Mixed-ReluLLaMA-7B-c4-self-data"
# model_name="Mixed-ReluLLaMA-7B-c4-self"

# model="7b"
# data_path="predictor_data/Mixed-ReluLLaMA-7B-c4-505074-16500-data"
# model_name="Mixed-ReluLLaMA-7B-c4-505074-16500"

# model="13b"
# data_path="predictor_data/Mixed-ReluLLaMA-13B-c4-self-data"
# model_name="Mixed-ReluLLaMA-13B-c4-self"

# model="13b"
# data_path="predictor_data/sparse_llama_13b_576835_16000-data"
# model_name="sparse_llama_13b_576835_16000"

# model="7b"
# data_path="predictor_data/vanilla_relu_7b_563709_17000-data"
# model_name="vanilla_relu_7b_563709_17000"

# model="7b"
# data_path="predictor_data/fixed_lambda_7b_563716_17000-data"
# model_name="fixed_lambda_7b_563716_17000"

# model="7b"
# data_path="predictor_data/shift300_7b_567496_17000-data"
# model_name="shift300_7b_567496_17000"

# model="7b"
# data_path="predictor_data/sparse_llama_7b_505074_16500-fatrelu-0.01-data"
# model_name="sparse_llama_7b_505074_16500_fatrelu_0.01"

model="13b"
data_path="predictor_data/sparse_llama_13b_576835_16000-fatrelu-0.01-data"
model_name="sparse_llama_13b_576835_16000_fatrelu_0.01"

if [[ ${model} == "7b" ]]; then
    num_layer=31
else
    num_layer=39
fi

mkdir -p logs/${model_name}

for l in $(seq 0 4 $num_layer)
do  
    (trap 'kill 0' SIGINT; \
    python main_mlp.py --dataset ${data_path} --model_name ${model_name} --model ${model} --lr 0.001 --L ${l}     > logs/${model_name}/c4_mlp_out_${l}.txt & \
    python main_mlp.py --dataset ${data_path} --model_name ${model_name} --model ${model} --lr 0.001 --L $((l+1)) > logs/${model_name}/c4_mlp_out_$((l+1)).txt & \
    python main_mlp.py --dataset ${data_path} --model_name ${model_name} --model ${model} --lr 0.001 --L $((l+2)) > logs/${model_name}/c4_mlp_out_$((l+2)).txt & \
    python main_mlp.py --dataset ${data_path} --model_name ${model_name} --model ${model} --lr 0.001 --L $((l+3)) > logs/${model_name}/c4_mlp_out_$((l+3)).txt & \
    # python main_mlp.py --dataset ${data_path} --model_name ${model_name} --model ${model} --lr 0.001 --L $((l+4)) > logs/${model_name}/c4_mlp_out_$((l+4)).txt & \
    # python main_mlp.py --dataset ${data_path} --model_name ${model_name} --model ${model} --lr 0.001 --L $((l+5)) > logs/${model_name}/c4_mlp_out_$((l+5)).txt & \
    # python main_mlp.py --dataset ${data_path} --model_name ${model_name} --model ${model} --lr 0.001 --L $((l+6)) > logs/${model_name}/c4_mlp_out_$((l+6)).txt & \
    # python main_mlp.py --dataset ${data_path} --model_name ${model_name} --model ${model} --lr 0.001 --L $((l+7)) > logs/${model_name}/c4_mlp_out_$((l+7)).txt & \
    wait)
done
