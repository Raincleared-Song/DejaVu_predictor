import os
import torch
import jsonlines
from hf_llama_module import LlamaForCausalLM, LlamaTokenizer, ActivationRecorder, ACTIVATION_RECORDERS


def test_generation():
    torch.set_grad_enabled(False)
    ActivationRecorder.set_record_flag(False)
    ActivationRecorder.set_record_path("ReluLLaMA-7B-c4-data")
    tokenizer = LlamaTokenizer.from_pretrained("/home/jeeves/ReluLLaMA-7B")
    model = LlamaForCausalLM.from_pretrained("/home/jeeves/ReluLLaMA-7B").to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    inputs = "Once upon a time,"
    inputs = tokenizer(inputs, return_tensors="pt").to("cuda")
    inputs["max_length"] = 10
    print(tokenizer.decode(model.generate(**inputs)[0]))
    for rid, recorder in enumerate(ACTIVATION_RECORDERS):
        print(rid, recorder.fp_mlp_id)


def main():
    """
    hdfs dfs -get /user/tc_agi/songchenyang/job_576835_ckpt_16000/ /home/jeeves/sparse_llama_13b_576835_16000
    hdfs dfs -get /user/tc_agi/songchenyang/job_563709_ckpt_17000/ /home/jeeves/vanilla_relu_7b_563709_17000
    hdfs dfs -get /user/tc_agi/songchenyang/job_563716_ckpt_17000/ /home/jeeves/fixed_lambda_7b_563716_17000
    hdfs dfs -get /user/tc_agi/songchenyang/job_567496_ckpt_17000/ /home/jeeves/shift300_7b_567496_17000
    """
    torch.set_grad_enabled(False)
    ActivationRecorder.set_record_flag(True)
    hidden_act = "relu"

    # model_path = "/home/jeeves/ReluLLaMA-7B/"
    # data_path = "predictor_data/Mixed-ReluLLaMA-7B-c4-self-data"

    # model_path = "/home/jeeves/sparse_llama_7b_505074_16500/model_hf/"
    # data_path = "predictor_data/Mixed-ReluLLaMA-7B-c4-505074-16500-data"

    # model_path = "/home/jeeves/ReluLLaMA-13B/"
    # data_path = "predictor_data/Mixed-ReluLLaMA-13B-c4-self-data"

    # model_path = "/home/jeeves/sparse_llama_13b_576835_16000/model_hf/"
    # data_path = "predictor_data/sparse_llama_13b_576835_16000-data"

    # model_path = "/home/jeeves/vanilla_relu_7b_563709_17000/model_hf/"
    # data_path = "predictor_data/vanilla_relu_7b_563709_17000-data"

    # model_path = "/home/jeeves/fixed_lambda_7b_563716_17000/model_hf/"
    # data_path = "predictor_data/fixed_lambda_7b_563716_17000-data"

    # model_path = "/home/jeeves/shift300_7b_567496_17000/model_hf/"
    # data_path = "predictor_data/shift300_7b_567496_17000-data"

    # model_path = "/home/jeeves/sparse_llama_7b_505074_16500/model_hf/"
    # data_path = "predictor_data/sparse_llama_7b_505074_16500-fatrelu-0.01-data"
    # hidden_act = "fatrelu_0.01"

    model_path = "/home/jeeves/sparse_llama_13b_576835_16000/model_hf/"
    data_path = "predictor_data/sparse_llama_13b_576835_16000-fatrelu-0.01-data"
    hidden_act = "fatrelu_0.01"

    ActivationRecorder.set_record_path(data_path)
    os.makedirs(data_path, exist_ok=True)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, hidden_act=hidden_act).to(dtype=torch.bfloat16, device="cuda")
    model.eval()

    reader = jsonlines.open("/home/jeeves/sparse_test_data.jsonl")
    for lid, line in enumerate(reader):
        text = line["prompt"]
        inputs = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        mlp_id = ACTIVATION_RECORDERS[0].fp_mlp_id
        print(f"Processed: {lid+1:03d} Sample Number: {mlp_id:06d}", end="\r")
        if mlp_id >= 400000:
            print("\ncomplete")
            break
    reader.close()
    print()

    for rid, recorder in enumerate(ACTIVATION_RECORDERS):
        print(rid, recorder.fp_mlp_id)


if __name__ == "__main__":
    # test_generation()
    main()
