import os
import re
import subprocess
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_sample", type=int, default=400)
    args = parser.parse_args()

    reader = jsonlines.open("/home/jeeves/sparse_test_data.jsonl")
    all_data = [entry for entry in reader]
    reader.close()

    record_sample, record_prompt_eval, record_eval = [], [], []
    total_num = min(args.num_sample, len(all_data))
    for idx, entry in tqdm(enumerate(all_data), total=total_num):
        prompt = entry["prompt"]
        prompt = " ".join(prompt.split(" ")[:5])
        cmd = ["../PowerInfer/build/bin/main", "-m", args.model_path, "-n", "128", "-t", "8", "-p", prompt]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        exitcode = proc.returncode
        assert exitcode == 0, " ".join(cmd)
        lines = err.decode("utf-8").split("\n")
        lines = [line for line in lines if "tokens per second" in line]
        assert len(lines) == 3, cmd
        for line, ls in zip(lines, [record_sample, record_prompt_eval, record_eval]):
            tokens = re.findall(r" ([0-9\.]+) tokens per second", line)
            assert len(tokens) == 1, line
            ls.append(float(tokens[0]))
        if idx >= args.num_sample:
            break
    print("sample speed:", round(np.mean(record_sample), 2), "tokens per second")
    print("prompt eval speed:", round(np.mean(record_prompt_eval), 2), "tokens per second")
    print("eval speed:", round(np.mean(record_eval), 2), "tokens per second")


if __name__ == "__main__":
    main()
