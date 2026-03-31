# -*- coding: utf-8 -*-
import os, pickle, numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
import collections
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

SR = 44100      # サンプリング周波数
N_FFT = 2048    # STFTのフレーム長
HOP = 512       # STFTのホップサイズ
WIN = 2048      # STFTの窓長
REF_F0 = 530.0     # ピッチ正規化の参照

F_BINS = N_FFT // 2 + 1 # スペクトルの周波数ビン数
HARMONICS = 32          # 調波成分の本数
NOISE_BANDS = 64        # ノイズ成分のバンド数
LATENT_D = 16           # PCAで削減する前の潜在空間の次元

# キャッシュの保存先
CACHE_PKL = ""
PCA_PATH = ""
BEST_PATH = ""
NPZ_PATH = ""

if os.path.exists(CACHE_PKL):
    print("[LOAD] 既存の pkl を読み込み:", CACHE_PKL)
    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    data = cache["dataset"]          # [(inp, cond, tgt, y_cls), ...] -> (入力スペクトログラム、条件特徴量、ターゲット、奏法ID)
    label_to_id = cache["label_to_id"]  # {"奏法名": id, ...}
    id_to_label = {v:k for k,v in label_to_id.items()}
    print("セグメント数:", len(data))
    print("ラベル数:", len(label_to_id))
dataset = []
for a in range(len(data)):
    inp, cond, tgt, y_cls = data[a]
    if inp.shape[1] == 104:
        dataset.append((inp, cond, tgt, y_cls))
    else:
        print(a, inp.shape, cond.shape, tgt.shape)


print("読み込みました。セグメント数:", len(dataset))
print("ラベル数:", len(label_to_id))
print(collections.Counter([it[3] for it in dataset]))
for key, value in label_to_id.items():
    print(key, value)


# データセットを PyTorch の Dataset として定義

def np2torch(inp, cond, tgt):
    # inp/tgt: [F,T] -> [T, F]
    # cond: [2,T] -> [T,2]
    x = torch.from_numpy(inp.T).unsqueeze(1).float()  # [T, 1, F]
    c = torch.from_numpy(cond.T).float()              # [T, 2]
    y = torch.from_numpy(tgt.T).float()               # [T, F]
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

ds = SpecDataset(dataset)
n_train = int(len(ds)*0.9)
n_val   = len(ds) - n_train
train_ds, val_ds = random_split(ds, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, drop_last=False)

xb, cb, yb, y_cls = next(iter(train_loader))
print("batch shapes:", xb.shape, cb.shape, yb.shape, y_cls.shape)

# モデル全体の定義（Encoder、Decoder、再パラメータ化トリック）


# 再パラメータ化トリック
def reparam(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps

# --------- Encoder ------------

class Encoder(nn.Module):
    def __init__(self, latent_d=LATENT_D,
                 fb_fmin=100.0, fb_fmax=None):
        super().__init__()

        # 畳み込み層
        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=(5,5), padding=(2,2)), nn.ReLU(), #(batch, 1, 104, 256) -> (batch, 32, 104, 128)
            nn.Conv2d(32, 64, kernel_size=(5,5), padding=(2,2)), nn.ReLU(), #(batch, 64, 104, 64)
            nn.Conv2d(64,128, kernel_size=(3,3), padding=(1,1)), nn.ReLU(), #(batch, 128, 104, 32)
        )

        # μ / logvar を各時刻で出力
        self.proj_mu = nn.LazyLinear(latent_d)
        self.proj_lv = nn.LazyLinear(latent_d)

    def forward(self, x):  # x:[B,T,1,F]
        B, T, _, Fbins = x.shape
        h = x.permute(0,2,1,3)           # [B,1,T,F]
        h = self.conv(h)                 # [B,128,T,F]

        # Conv出力のチャネル平均で「スペクトルの時間推移」を要約 [B,T,F]
        h = h.mean(dim=1)
        mu = self.proj_mu(h)        # [B,T,latent]
        lv = self.proj_lv(h)        # [B,T,latent]
        return mu, lv



# --------- Decoder ------------

class Decoder(nn.Module):
    def __init__(self, harmonics=HARMONICS, bands=NOISE_BANDS,
                 f_bins=F_BINS, mel_bins=256,
                 mel_fmin=0.0, mel_fmax=None):
        super().__init__()

        in_d = LATENT_D + 2     # 潜在次元 + 条件特徴量（f0_n, l_n）

        # 二層の全結合層
        self.mlp = nn.Sequential(
            nn.Linear(in_d, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head_harm  = nn.Linear(256, harmonics)     # 調波成分の振幅を推測
        self.head_noise = nn.Linear(256, bands)         # ノイズ成分のバンド振幅を推測

        self.register_buffer("freq_index", torch.arange(f_bins).float())
        self.harm_width = nn.Parameter(torch.tensor(1.5))   # 調波のスペクトル幅 (学習可能)
        self.f_bins = f_bins
        self.bands  = bands

        # melフィルタバンクの計算
        fmax = mel_fmax if mel_fmax is not None else SR / 2.0
        mel = librosa.filters.mel(
            sr=SR,
            n_fft=N_FFT,
            n_mels=mel_bins,
            fmin=mel_fmin,
            fmax=fmax
        ).astype(np.float32)              # [mel, F]

        mel = torch.from_numpy(mel)
        mel = mel / (mel.sum(dim=1, keepdim=True) + 1e-8)  # 正規化
        self.register_buffer("mel_fb", mel)                # [mel, F]


    def harmonic_splat(self, f0_hz, amps):
        B,T,H = amps.shape
        F = self.f_bins
        # 周波数binあたりのHz（FFT解像度）
        hz_per_bin = SR / N_FFT
        # # 調波成分の番号、F0から (1,2,...,H)
        k = torch.arange(1, H+1, device=amps.device).view(1,1,H)   # [1,1,H]
        # 各調波の中心周波数をbin単位に変換
        pk = (k * f0_hz.unsqueeze(-1)) / hz_per_bin                # [B,T,H]
        pk = pk.unsqueeze(-1)                                      # [B,T,H,1]
        f  = self.freq_index.view(1,1,1,F).to(amps.device)         # [1,1,1,F]
        # 調波スペクトルをガウス関数でスプラット
        sigma = torch.clamp(self.harm_width, 0.5, 4.0)
        # スペクトログラム上に配置
        g = torch.exp(-(f - pk)**2 / (2*sigma**2))                 # [B,T,H,F]
        spec = (amps.unsqueeze(-1) * g).sum(dim=2)                 # [B,T,F]
        return spec

    def band_interp(self, bands):
        B,T,Bn = bands.shape
        x = bands.reshape(B*T, 1, Bn)                 # [BT,1,B]
        out = Fnn.interpolate(x, size=self.f_bins, mode='linear', align_corners=True)  # [BT,1,F]
        return out.reshape(B, T, self.f_bins)

    def forward(self, z, cond):
        # 入力整形
        h = torch.cat([z, cond], dim=-1)             # [B,T,LATENT_D+2]
        h = self.mlp(h)                               # [B,T,256]
        harm_a  = Fnn.softplus(self.head_harm(h))  + 1e-4   # [B,T,H]
        noise_b = Fnn.softplus(self.head_noise(h)) + 1e-4   # [B,T,B]

        # cond の f0_n を Hz に戻す
        f0_hz = cond[...,0]*400.0 + REF_F0           # [B,T]
        harm_spec  = self.harmonic_splat(f0_hz, harm_a)     # [B,T,F], 調波成分
        noise_spec = self.band_interp(noise_b)               # [B,T,F]、雑音成分
        mag = harm_spec + noise_spec   # [B,T,F], 線形スペクトル

        # メルスペクトログラムに変換
        # [B,T,F] x [F,mel] -> [B,T,mel]
        mel_mag   = torch.matmul(mag,        self.mel_fb.T)
        mel_harm  = torch.matmul(harm_spec,  self.mel_fb.T)
        mel_noise = torch.matmul(noise_spec, self.mel_fb.T)

        # dBに変換
        eps = 1e-10
        ref = mel_mag.amax(dim=(1,2), keepdim=True).clamp_min(eps)  # [B,1,1]  (ref=np.max 相当)

        yhat_db  = 10.0 * torch.log10((mel_mag   / ref).clamp_min(eps))   # [B,T,256]
        harm_db  = 10.0 * torch.log10((mel_harm  / ref).clamp_min(eps))
        noise_db = 10.0 * torch.log10((mel_noise / ref).clamp_min(eps))

        return yhat_db, harm_db, noise_db

# -------- 学習ループ --------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(model, loader, opt, beta_kl=3e-4):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        xb, cb, yb, _ = batch
        T = min(xb.shape[1], cb.shape[1], yb.shape[1])
        xb = xb[:, :T].to(device)
        cb = cb[:, :T].to(device)
        yb = yb[:, :T].to(device)

        opt.zero_grad()
        out = model(xb, cb)
        yhat, mu, lv, _, _ = out

        l_rec = Fnn.l1_loss(yhat, yb)
        l_kl  = kl_loss(mu, lv)
        loss = l_rec + beta_kl*l_kl
        pbar.set_postfix(rec=float(l_rec.item()), kl=float(l_kl.item()),
                             tot=float(loss.item()))

        loss.backward()
        opt.step()
        total += loss.item(); n += 1
    return total/max(n,1)


def eval_epoch(model, loader, beta_kl=3e-4):
    model.eval()
    total, n = 0.0, 0
    acc_sum, acc_n = 0.0, 0
    pbar = tqdm(loader, desc="valid", leave=False)
    for batch in pbar:
        xb, cb, yb, _ = batch
        T = min(xb.shape[1], cb.shape[1], yb.shape[1])
        xb = xb[:, :T].to(device)
        cb = cb[:, :T].to(device)
        yb = yb[:, :T].to(device)

        out = model(xb, cb)
        yhat, mu, lv, _, _ = out

        l_rec = Fnn.l1_loss(yhat, yb)
        l_kl  = kl_loss(mu, lv)
        loss = l_rec + beta_kl*l_kl
        pbar.set_postfix(rec=float(l_rec.item()), kl=float(l_kl.item()))

        total += loss.item(); n += 1

    metrics = {"loss": total/max(n,1)}
    if acc_n > 0: metrics["acc"] = acc_sum/max(acc_n,1)
    return metrics

@torch.no_grad()
def encode_segments(model, loader, max_batches=200):
    model.eval()
    Z = []
    Y = []
    for i, batch in enumerate(loader):
        if (max_batches is not None) and (i >= max_batches):
            break
        y_cls = None
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            xb, cb, yb, y_cls = batch
        else:
            xb, cb, yb = batch[:3]
        T = min(xb.shape[1], cb.shape[1], yb.shape[1])
        xb = xb[:, :T].to(device)
        mu, lv = model.enc(xb)                # [B,T,latent]
        z_seg = mu.mean(dim=1).cpu().numpy()  # [B,latent]
        Z.append(z_seg)
        if y_cls is not None:
            Y.append(y_cls.numpy())            # [B]
    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0) if len(Y)>0 else None
    return Z, Y


def plot_latent_progress(model, loader, id_to_label, epoc, colors, title="PCA of latent (per segment)", max_batches=None):
    Z, Y = encode_segments(model, loader, max_batches=max_batches)
    if len(Z) == 0:
        print("可視化対象サンプルがありません。")
        return
    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)

    if Y is None:
        print("警告: y_cls が loader から取れませんでした（色分け不可）。")
        plt.figure(figsize=(6,6))
        plt.scatter(Z2[:,0], Z2[:,1], s=10, alpha=0.6)
        plt.title(title); plt.grid(True); plt.axis('equal'); plt.show()
        return

    # y_id → ラベル名
    labs = np.array([id_to_label[int(i)] for i in Y])
    uniq = sorted(set(labs))
    print("可視化対象に含まれるクラス:", uniq)

    if epoc % 5 == 0 or epoc == 0 or epoc == 1 or epoc == 2 or epoc == 3 or epoc == 4:
        plt.figure(figsize=(6,6))
        plt.scatter(Z2[:,0], Z2[:,1], s=10, alpha=0.6, color=[colors[i] for i in Y])
        plt.title(title); plt.grid(True); plt.axis('equal'); plt.show()
    return Z, Z2, Y, pca

class VAEHarmNoise(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x, cond):
        mu, lv = self.enc(x)                      # [B,T,latent]
        z = reparam(mu, lv)
        yhat, hs, ns = self.dec(z, cond)
        return yhat, mu, lv, hs, ns

# --------実行部分--------
model = VAEHarmNoise(num_classes=None).to(device)

opt = torch.optim.Adam(model.parameters(), lr=2e-3)
EPOCHS = 50
best = 1e9
colors = ["#ff0000", "#ffaaaa", "#ff7777"]
ep = 0
count = 0

while ep < EPOCHS and count < 5:
    tr = train_epoch(model, train_loader, opt, beta_kl=3e-4)
    va = eval_epoch(model, val_loader,   beta_kl=3e-4)
    if isinstance(va, dict):
        print(f"[{ep:02d}] train_loss={tr:.4f}  valid_loss={va['loss']:.4f}  " +
              (f"acc={va['acc']:.3f}" if 'acc' in va else ""))
        cur = va["loss"]
    else:
        print(f"[{ep:02d}] train_loss={tr:.4f}  valid_loss={va:.4f}")
        cur = va

    # 5エポックごとに可視化
    Z, Z2, Y, pca = plot_latent_progress(model, train_loader, id_to_label, ep, colors, title=f"PCA latent @ epoch {ep}", max_batches=None)
    print(len(Y))
    ep += 1

    if cur < best:
        best = cur
        torch.save(model.state_dict(), BEST_PATH)
        print("  ↳ best updated, saved to" , BEST_PATH)

        np.savez(NPZ_PATH,
        Z=Z,              # (N, latent)
        Z2=Z2,            # (N, 2)
        y_labels=Y,              # (N,)  (整数ID)
        label_to_id=label_to_id,  # (N,)  (文字列)
        colors=colors # (N,)  (色コード)
        )
        print("   ↳ npz, saved to",  NPZ_PATH)

        joblib.dump(pca, PCA_PATH)
        print("[PCA] Saved to ", PCA_PATH)

        count = 0
    else:
        count += 1
