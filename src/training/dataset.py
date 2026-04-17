import torch
from torch.utils.data import Dataset


class RecSysDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
            "history": sample["history"],
            "target": sample["target"]
        }
    
def collate_fn(batch, max_history_len):
    histories = []
    targets = []

    for sample in batch:
        history = sample["history"]
        target = sample["target"]

        # truncate
        history = history[-max_history_len:]

        # pad (LEFT padding)
        padding = [0] * (max_history_len - len(history))
        history = padding + history

        histories.append(history)
        targets.append(target)

    histories = torch.tensor(histories, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return {
        "history": histories,   # (B, max_len)
        "target": targets       # (B,)
    }
import json
from torch.utils.data import DataLoader


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_json("data/processed/train.json")

    dataset = RecSysDataset(data)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, max_history_len=5)
    )

    for batch in dataloader:
        print("History shape:", batch["history"].shape)
        print("Target shape:", batch["target"].shape)
        print("\nHistory:")
        print(batch["history"])
        print("\nTarget:")
        print(batch["target"])
        break