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
        print(rid, recorder.fp_att_id, recorder.fp_mlp_id)


def main():
    torch.set_grad_enabled(False)
    ActivationRecorder.set_record_flag(True)
    ActivationRecorder.set_record_path("ReluLLaMA-7B-c4-data")
    tokenizer = LlamaTokenizer.from_pretrained("/home/jeeves/ReluLLaMA-7B")
    model = LlamaForCausalLM.from_pretrained("/home/jeeves/ReluLLaMA-7B").to(dtype=torch.bfloat16, device="cuda")
    model.eval()

    reader = jsonlines.open("c4_train.jsonl")
    for lid, line in enumerate(reader):
        text = line["prompt"]
        inputs = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        mlp_id, att_id = ACTIVATION_RECORDERS[0].fp_mlp_id, ACTIVATION_RECORDERS[0].fp_att_id
        print(f"Processed: {lid+1:03d} Sample Number: {mlp_id:06d} | {att_id:06d}", end="\r")
        if mlp_id >= 400000:
            print("\ncomplete")
            break
    reader.close()
    print()

    for rid, recorder in enumerate(ACTIVATION_RECORDERS):
        print(rid, recorder.fp_att_id, recorder.fp_mlp_id)


if __name__ == "__main__":
    # test_generation()
    main()
