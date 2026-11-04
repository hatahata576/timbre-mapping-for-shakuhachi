#データセット読み込み
import numpy as np
import librosa
import os
import pickle

SR = 44100              #サンプリング周波数
N_FFT = 2048            #フーリエ変換フレーム長
HOP = 512               #ホップ数
WIN = 2048              #窓長

REF_F0 = 530.0          # 正規化の音高（C5)
TARGET_RMS = 0.1        # RMS正規化の値
SEG_SEC = 1.2           # セグメント長
STEP_SEC = 0.4          # ステップ長

# 音源フォルダパス
root = ""
# 保存先パス
save_path = "~.pkl"

#wavファイルをモノラルでロード
def load_mono(path, sr=SR):
    x, _ = librosa.load(path, sr=sr, mono=True)
    x = librosa.util.normalize(x)
    return x

#セグメント長に従って分割
def slice_wav(x, sr=SR, seg_sec=SEG_SEC, step_sec=STEP_SEC):
    seg_len = int(seg_sec*sr)
    step = int(step_sec*sr)
    out = []
    for s in range(0, max(1, len(x)-seg_len), step):
        e = s+seg_len
        f0 = s//HOP
        f1 = e//HOP
        out.append(x[s:e])
    return out

#F0抽出
def extract_f0(x):
    f0, _, _ = librosa.pyin(
        x, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=SR, frame_length=WIN, hop_length=HOP
    )
    f0 = np.array(f0)
    if np.isnan(f0).all():
        return np.zeros_like(f0)
    nans = np.isnan(f0)
    f0[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), f0[~nans])
    return f0

#メルスペクトログラム変換
def stft_mel(x):
    S_mel = librosa.feature.melspectrogram(
    y=x, sr=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    win_length=N_FFT,
    window="hann",
    n_mels=256,      # メルバンド数
    fmin=0,
    fmax=SR/2
    )
    log_mel = librosa.power_to_db(S_mel, ref=np.max, top_db=None)
    return log_mel.astype(np.float32)


dataset = []    #入力スペクトログラム、奏法、奏法のIDを収納するリスト
labels = []     #奏法の文字ラベルを収納するリスト
number = 0

files = sorted([f for f in os.listdir(root) if f.lower().endswith(".wav")])

for fname in files:
    path = os.path.join(root, fname)
    x = load_mono(path)
    segs = slice_wav(x)
    tech = os.path.basename(path).split("_")[0]     #ファイル名の先頭から奏法名を抽出

    for seg in segs:
        # ---- ピッチ正規化 ----
        f0_raw = extract_f0(seg)
        f0_med = np.median(f0_raw[f0_raw > 0]) if np.any(f0_raw > 0) else REF_F0
        semi = 12 * np.log2((f0_med + 1e-6) / REF_F0)
        seg_ps = librosa.effects.pitch_shift(seg, sr=SR, n_steps=-semi)

        # ---- RMS正規化 ----
        rms = np.sqrt(np.mean(seg_ps**2)) + 1e-8
        seg_ps = seg_ps * (TARGET_RMS / rms)

        # ---- 特徴抽出 ----
        f0   = extract_f0(seg_ps)
        spec = stft_mel(seg_ps)     #[F,T]
        rms_frame = librosa.feature.rms(y=seg_ps, frame_length=WIN, hop_length=HOP).flatten()

        T = spec.shape[1]
        L = min(T, len(f0), len(rms_frame))
        if L < 4:
            continue

        spec = spec[:, :L]
        f0   = f0[:L]
        rms_frame = rms_frame[:L]

        f0_n  = (f0 - REF_F0) / 400.0
        rms_n = (np.log(rms_frame + 1e-8) + 5.0) / 5.0
        cond  = np.stack([f0_n.astype(np.float32), rms_n.astype(np.float32)], axis=0)  # [2,T]

        inp = spec.copy()
        tgt = spec
        if spec.shape[1] != 104:
            print(spec.shape[1])
            continue

        dataset.append((inp, cond, tgt, tech))      #（入力スペクトログラム、条件特徴量、ターゲットスペクトログラム、奏法名（文字列））
        labels.append(tech)
        number += 1
        if number % 10 == 0:
            print(f"{number}個完了")



print("セグメント数(文字列ラベル付き):", len(dataset))

# 奏法名 <--> IDの辞書
unique_labels = sorted(set(labels))
label_to_id = {lab: i for i, lab in enumerate(unique_labels)}

#奏法名（文字列）をIDに変換
dataset2 = [(inp, cond, tgt, label_to_id[tech]) for (inp, cond, tgt, tech) in dataset]

print("ラベル → ID 対応表:")
for k, v in label_to_id.items():
    print(f"  {k} -> {v}")
print("最終セグメント数:", len(dataset2))




with open(save_path, "wb") as f:
    pickle.dump({"dataset": dataset2, "label_to_id": label_to_id}, f)

print("保存しました:", save_path)
