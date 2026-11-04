# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


@torch.no_grad()
def encode_segments(model, loader, device, max_batches=200):
    model.eval()
    Z = []
    Y = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        xb, cb, yb, y_cls = batch
        T = min(xb.shape[1], cb.shape[1], yb.shape[1])
        xb = xb[:, :T].to(device)

        mu, _ = model.enc(xb)
        z_seg = mu.mean(dim=1).cpu().numpy()
        Z.append(z_seg)
        Y.append(y_cls.numpy())

    if len(Z) == 0:
        return np.empty((0, 0)), None

    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0) if len(Y) > 0 else None
    return Z, Y


def plot_latent_progress(
    model,
    loader,
    id_to_label,
    epoch,
    colors,
    device,
    title="PCA of latent (per segment)",
    max_batches=None,
):
    Z, Y = encode_segments(model, loader, device, max_batches=max_batches)

    if len(Z) == 0:
        print("可視化対象サンプルがありません。")
        return None, None, None, None

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)

    if Y is None:
        print("警告: y_cls が loader から取れませんでした（色分け不可）。")
        plt.figure(figsize=(6, 6))
        plt.scatter(Z2[:, 0], Z2[:, 1], s=10, alpha=0.6)
        plt.title(title)
        plt.grid(True)
        plt.axis("equal")
        plt.show()
        return Z, Z2, Y, pca

    labs = np.array([id_to_label[int(i)] for i in Y])
    uniq = sorted(set(labs))
    print("可視化対象に含まれるクラス:", uniq)

    if epoch % 5 == 0 or epoch < 5:
        plt.figure(figsize=(6, 6))
        plt.scatter(
            Z2[:, 0],
            Z2[:, 1],
            s=10,
            alpha=0.6,
            color=[colors[i] for i in Y],
        )
        plt.title(title)
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    return Z, Z2, Y, pca
