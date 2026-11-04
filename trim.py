#音声ファイルおよび出力先のファイルを指定
wav_dir       = ""       # wavファイルが入ったフォルダ
seg_wav_dir   = ""       # 出力先のフォルダ
sr            = 44100    # サンプリング周波数
top_db_trim   = 20       # 無声部分のトリミングの閾値 [dB]
min_len_sec   = 1.2      # 保存する最短長さ [s]

# --- 準備 ----------------------------------------------------------------------
import os
import glob
import librosa
import soundfile as sf
import warnings
import tqdm
warnings.filterwarnings("ignore")

os.makedirs(seg_wav_dir, exist_ok=True)

wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
print(f"累計：{len(wav_files)} ファイル\n")

# --- 実行部 ------------------------------------------------------
for wav_path in tqdm.tqdm(wav_files):
    basename = os.path.splitext(os.path.basename(wav_path))[0]

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    orig_len_sec = librosa.get_duration(y=y, sr=sr)

    #top_db_trim以下の部分をトリミング
    intervals = librosa.effects.split(y, top_db=top_db_trim)

    print(f"\n{basename}.wav  (全長 {orig_len_sec:.2f} s, 区間 {len(intervals)} 個)")

    kept = 0
    for idx, (start, end) in enumerate(intervals):
        seg_len_sec = (end - start) / sr
        if seg_len_sec < min_len_sec:
            # min_len_sec 秒未満は保存しない
            print(f"  └─ {basename}_{idx:03d}.wav  {seg_len_sec:.2f} s  →  skip (<{min_len_sec} s)")
            continue

        seg = y[start:end]
        out_path = os.path.join(seg_wav_dir, f"{basename}_{idx:03d}.wav")
        sf.write(out_path, seg, sr, subtype="PCM_16")
        print(f"{basename}_{idx:03d}.wav  {seg_len_sec:.2f} s")
        kept += 1

    print(f"  保存数: {kept} / {len(intervals)}")

print(f"{min_len_sec} 秒以上のセグメントのみを {seg_wav_dir} に保存完了")
