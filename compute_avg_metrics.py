import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dim_ff", type=int, default=11008)
    args = parser.parse_args()

    # Mixed-ReluLLaMA-7B-c4-self-full
    # Mixed-ReluLLaMA-7B-c4-505074-16500-full
    # Mixed-ReluLLaMA-13B-c4-self

    checkpoints = [ch for ch in os.listdir(f"checkpoints/{args.model_name}") if ch.startswith("layer")]
    checkpoints.sort(key=lambda x: int(x.split("-")[0][5:]))

    recalls, act_rates, sparsities = [], [], []
    for ch in checkpoints:
        _, recall, act_rate = ch[:-3].split("-")
        recall, act_rate = float(recall), float(act_rate) / args.dim_ff
        recalls.append(recall)
        act_rates.append(act_rate)
        sparsities.append(1 - act_rate)
    print(f">>> Model Name {args.model_name} <<<")
    print("Recall:", round(np.mean(recalls) * 100, 2))
    print("Act Rate:", round(np.mean(act_rates) * 100, 2))
    print("Sparsity:", round(np.mean(sparsities) * 100, 2))


if __name__ == "__main__":
    main()
