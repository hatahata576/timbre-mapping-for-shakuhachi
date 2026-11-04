# -*- coding: utf-8 -*-

# 音響設定
SR = 44100
N_FFT = 2048
HOP = 512
WIN = 2048
REF_F0 = 530.0

# モデル設定
F_BINS = N_FFT // 2 + 1
HARMONICS = 32
NOISE_BANDS = 64
LATENT_D = 16
MEL_BINS = 256

# データ・モデル保存先

CACHE_PKL = "/content/drive/MyDrive/お試し版/dataset_me20251230_mel.pkl"  # 学習用データ(pkl)
PCA_PATH = "/content/drive/MyDrive/お試し版/自分の音声_20251230/20251230.pca"   # PCAモデル
BEST_PATH = "/content/drive/MyDrive/お試し版/result/20251230.pt"  # 学習済みモデル
NPZ_PATH = "/content/drive/MyDrive/お試し版/result/20251230.npz"   # 潜在空間データ(npz)

# 学習設定
BATCH_SIZE = 16
TRAIN_RATIO = 0.9
LEARNING_RATE = 2e-3
EPOCHS = 50
BETA_KL = 3e-4
EARLY_STOP_PATIENCE = 5
SEGMENT_FRAMES = 104

# 可視化用
COLORS = ["#ff0000", "#ffaaaa", "#ff7777"]
