import os
import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from trainer_mlp import train


MODEL_CHOICES = ['7b', '13b']
CONFIG = {
    '7b':{
        'num_layer': 32,
        'ckt_storage': "bylayer",
        'd': 4096,
        'f': 11008,
        'h': 32,
        'N': 400000,  # 313336
    },
    '13b':{
        'num_layer': 40,
        'ckt_storage': "bylayer",
        'd': 5120,
        'f': 13824,
        'h': 40,
        'N': 400000,  # 313336
    },
}

class BasicDataset(Dataset):
    def __init__(self, X, Y, n, train ):
        self.X = X
        self.Y = Y 
        self.n = n
        self.train = train

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.train:
            x = torch.Tensor(self.X[idx])
            y = torch.Tensor(self.Y[idx])
        else:
            x = torch.Tensor(self.X[-idx])
            y = torch.Tensor(self.Y[-idx])
        # if y.sum() == 0:
        #     print(">>> all zero detected <<<")
        return x, y


def get_data(args, l):
    if CONFIG[args.model]['ckt_storage'] == "bylayer":
        path = f"{args.dataset}/mlp_sp_x_{l}.mmap"
        print(f"Reading query from {path}")
        query = np.array(np.memmap(path, dtype='float16', mode='r', shape=(400000, CONFIG[args.model]['d'])))[:CONFIG[args.model]['N']]
    
        path = f"{args.dataset}/mlp_label_{l}.mmap"
        print(f"Reading MLP label from {path}")
        label = np.array(np.memmap(path, dtype='float16', mode='r', shape=(400000, CONFIG[args.model]['f'])))[:CONFIG[args.model]['N']]
    
        return  query, label


def create_dataset(query, labels, args):

    total = len(query)
    num_train = int(0.95 * total)
    num_test = int(0.05 * total)

    print(f"Query shape: {query.shape}, Label shape: {labels.shape}")
    print(f"# training data: {num_train}, # test data: {num_test}")

    train_ds = BasicDataset(query, labels, num_train, True)
    test_ds = BasicDataset(query, labels, num_test, False)

    train_dataloader = DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)
    return train_dataloader, test_dataloader


def main():
    parser = argparse.ArgumentParser(description="PyTorch OPT Full Model")
    parser.add_argument("--model", type=str, default="7b", choices = MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--L",
        type=int,
        default=0,
        help="which layer",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=1024,
        help="low rank dimension",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate",
    )
    args = parser.parse_args()

    print(args)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda"

    print("=" * 40, "Layer", args.L, "=" * 40)

    query, labels = get_data(args, args.L)

    train_loader, test_loader = create_dataset(query, labels, args)

    query_layer = torch.nn.Sequential(
        torch.nn.Linear(CONFIG[args.model]['d'], args.D, bias=None),
        torch.nn.ReLU(),
        torch.nn.Linear(args.D, CONFIG[args.model]['f'], bias=None),
    )

    print("Start Training")
    best_model, eval_result = train(
        query_layer,  train_loader, test_loader, args, device, verbal=True
    )

    file_name = f"layer{args.L}-{eval_result['Recall']:.4f}-{eval_result['Classifier Sparsity']:.0f}.pt"
    model_path = os.path.join("checkpoints", args.model_name)
    os.makedirs(model_path, exist_ok=True)
    path = f"{model_path}/{file_name}"
    new_best_model = {"fc1.weight": best_model["0.weight"], "fc2.weight": best_model["2.weight"]}
    torch.save(new_best_model, path)
    if os.path.exists(f"{model_path}/model_{args.L}.pt"):
        os.system(f"rm {model_path}/model_{args.L}.pt")
    os.system(f"ln -s {file_name} {model_path}/model_{args.L}.pt")


if __name__ == "__main__":
    main()
