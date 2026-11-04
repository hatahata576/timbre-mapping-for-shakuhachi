import os
import pickle
import collections

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from config import CACHE_PKL, SEGMENT_FRAMES, TRAIN_RATIO, BATCH_SIZE


def load_cached_dataset(cache_pkl=CACHE_PKL, segment_frames=SEGMENT_FRAMES):
    #pklからデータを読み込み、指定フレーム長のデータのみ抽出する。
    if not os.path.exists(cache_pkl):
        raise FileNotFoundError(f"pklが見つかりません: {cache_pkl}")

    print("pklを読み込み開始:", cache_pkl)
    with open(cache_pkl, "rb") as f:
        cache = pickle.load(f)

    data = cache["dataset"]
    label_to_id = cache["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}

    dataset = []
    for i, (inp, cond, tgt, y_cls) in enumerate(data):
        if inp.shape[1] == segment_frames:
            dataset.append((inp, cond, tgt, y_cls))
        else:
            print(i, inp.shape, cond.shape, tgt.shape)

    print("読み込完了。セグメント数:", len(dataset))
    print("ラベル数:", len(label_to_id))
    print(collections.Counter([it[3] for it in dataset]))
    for key, value in label_to_id.items():
        print(key, value)

    return dataset, label_to_id, id_to_label


def np2torch(inp, cond, tgt):
    """
    inp/tgt: [F,T] -> x:[T,1,F], y:[T,F]
    cond:    [2,T] -> c:[T,2]
    """
    x = torch.from_numpy(inp.T).unsqueeze(1).float()
    c = torch.from_numpy(cond.T).float()
    y = torch.from_numpy(tgt.T).float()
    return x, c, y


class SpecDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        inp, cond, tgt, y_id = self.items[i]
        x, c, y = np2torch(inp, cond, tgt)
        y_cls = torch.tensor(y_id, dtype=torch.long)
        return x, c, y, y_cls


def create_dataloaders(
    items,
    train_ratio=TRAIN_RATIO,
    batch_size=BATCH_SIZE,
    generator=None,
):
    ds = SpecDataset(items)
    n_train = int(len(ds) * train_ratio)
    n_val = len(ds) - n_train

    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader
