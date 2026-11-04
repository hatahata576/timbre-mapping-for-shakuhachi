import os
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import pickle, joblib
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output, display
from pathlib import Path
import sounddevice as sd
import queue
import threading



SR = 44100       # 学習時と同じサンプリングレート
SEG_SEC = 1.2    # セグメント長（秒）
STEP_SEC = 0.4   # セグメント更新間隔（秒）
N_FFT = 2048
HOP = 512
WIN = 2048
LATENT_D = 16
CHANNELS = 1
INPUT_DEVICE = None  # 特定の入力デバイスIDを使うなら整数IDを指定
seg_len  = int(SEG_SEC * SR)     # 例: 1.2秒 * 22050 = 26460 サンプル
step_len = int(STEP_SEC * SR)    # 例: 0.4秒 * 22050 = 8820 サンプル
stop_flag = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = Path("~")

BEST_PATH     = cwd / "~.pt"
PCA_PATH  = cwd / "~.pkl"
NPZ_PATH = cwd / "~.npz"


# 新：マイク入力用キュー
q_frames = queue.Queue()

def wait_key():
    global stop_flag
    input("停止したいときは Enter を押してください。\n")
    stop_flag = True

threading.Thread(target=wait_key, daemon=True).start()


# コールバック関数
def audio_cb(indata, frames, time_info, status):
    # モノラル前提（ステレオなら平均または片chを選ぶ）
    x = indata[:, 0].astype(np.float32).copy()
    q_frames.put(x)

# リングバッファ実体
buf = np.zeros(seg_len, dtype=np.float32)
w = 0           # 書き込み位置
filled = False  # 一周回って満杯になったかどうか


#キューに入った音声データをリングバッファに追記する関数

def rb_push(x: np.ndarray):
    """リングバッファに追記"""
    global buf, w, filled
    n = len(x)
    if n >= seg_len:
        buf[:] = x[-seg_len:]
        w = 0
        filled = True
        return
    end = w + n
    if end <= seg_len:
        buf[w:end] = x
    else:
        k = seg_len - w
        buf[w:]   = x[:k]
        buf[:n-k] = x[k:]
    w = (w + n) % seg_len
    if w == 0:
        filled = True


#キューから音声データを取り出し、直近seg_len分を返す関数

def get_latest_segment():
    """直近 SEG_SEC 秒（seg_len サンプル）を返す。まだ足りなければ None を返す。"""
    global buf, w, filled
    # 溜まっているフレームを全部取り出して追記
    while True:
        try:
            chunk = q_frames.get_nowait()
        except queue.Empty:
            break
        rb_push(chunk)

    if not filled and w < seg_len:
        return None  # まだ1.2秒分たまっていない

    # 最新 seg_len サンプルを連続化して返す
    start = (w - seg_len) % seg_len
    if start == 0:
        return buf.copy()
    else:
        k = seg_len - start
        return np.concatenate([buf[start:], buf[:seg_len-k]]).copy()


#波形を読み込み、セグメント長に切り出す関数
def load_mono(path, sr=SR):
    x, _ = librosa.load(path, sr=sr, mono=True)
    x = librosa.util.normalize(x)
    return x

#基本周波数抽出の関数定義
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

#スペクトログラムに変換の関数定義
def stft_mel(x):
    S_mel = librosa.feature.melspectrogram(
    y=x, sr=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    win_length=N_FFT,
    window="hann",
    n_mels=256,
    fmin=0,
    fmax=SR/2
    )
    log_mel = librosa.power_to_db(S_mel, ref=np.max, top_db=None)
    return log_mel.astype(np.float32)

#エンコーダ定義
class Encoder(nn.Module):
    def __init__(self, latent_d=LATENT_D, n_bands=5,
                 fb_fmin=100.0, fb_fmax=None):
        super().__init__()
        self.n_bands = n_bands
        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=(5,5), padding=(2,2)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), padding=(2,2)), nn.ReLU(),
            nn.Conv2d(64,128, kernel_size=(3,3), padding=(1,1)), nn.ReLU(),
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

#エンコーダ再構成
state_full = torch.load(BEST_PATH, map_location=device)
enc_state = {}
for k, v in state_full.items():
    if not k.startswith("enc."):
        continue
    kk = k.replace("enc.", "", 1)
    if kk.startswith("proj_lv."):
        continue  # ← ここで除外
    enc_state[kk] = v
encoder = Encoder(latent_d=LATENT_D).to(device)
missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
if missing or unexpected:
    print("[Warn] missing:", missing, " unexpected:", unexpected)
encoder.eval()

#pca読み込み
pca = joblib.load(PCA_PATH)

# --- 既存データの読み込み ---
dat = np.load(NPZ_PATH, allow_pickle=True)
Z  = dat["Z"]      # (N, latent)
Z2 = dat["Z2"]     # (N, 2)
Y  = dat["y_labels"]      # (N,)
colors = ["#ff0000", "#ffdddd", "#ff7777"]
# --- Figure を作成（1つだけ使う） ---
fig, ax = plt.subplots(figsize=(6,5))

# 既存データをプロット
ax.scatter(Z2[:,0], Z2[:,1], s=5, c=[colors[i] for i in Y], alpha=0.85)
# ====== 既存点(Z2)プロット直後に追加 ======

# 既存点の中心・スケール（「近い/遠い」判定に使う）
Z2_center = Z2.mean(axis=0)             # (2,)
Z2_std    = Z2.std(axis=0) + 1e-9       # (2,) 0割防止

# 「近い」とみなす閾値（大きいほど判定が甘くなる）
NEAR_K = 2.5  # 2.0〜4.0あたりで調整

# 余白
PAD = 0.10

# 新規点を溜める（全体点=既存+新規 でズームインしたいので履歴が必要）
Z2_new = []   # list of (x,y)

# 表示範囲（初期は既存点に合わせる）
xmin, xmax = float(Z2[:,0].min()), float(Z2[:,0].max())
ymin, ymax = float(Z2[:,1].min()), float(Z2[:,1].max())
dx = (xmax - xmin) if xmax > xmin else 1.0
dy = (ymax - ymin) if ymax > ymin else 1.0

cur_xlim = [xmin - PAD*dx, xmax + PAD*dx]
cur_ylim = [ymin - PAD*dy, ymax + PAD*dy]
ax.set_xlim(*cur_xlim)
ax.set_ylim(*cur_ylim)

# ズームを滑らかにする係数（1.0=即時、0.1=ゆっくり追従）
ZOOM_ALPHA = 0.25

# =========================================
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("Latent space + new mapped points")
fig.tight_layout()
# 最初の1回表示
clear_output(wait=True)
display(fig)



#新：マイクを開く
stream = sd.InputStream(
    samplerate=SR,
    channels=CHANNELS,
    device=INPUT_DEVICE,
    callback=audio_cb,
    blocksize=step_len  # STEP_SECごとにコールバック
)
stream.start()
print("Mic stream started. Ctrl-C で停止 / ウィンドウを閉じても可。")

# ========= GUI 実装部分 =========



running = False         # Start/Stop 用フラグ
last_point = None       # 直近の追加ポイント（描画し直す用）


def start_plot():
    """Start ボタンが押されたときに呼ばれる"""
    global running
    running = True
    print("リアルタイムプロット開始")


def stop_plot():
    """Stop ボタンが押されたときに呼ばれる"""
    global running
    running = False
    print("リアルタイムプロット停止")


def on_close():
    """ウィンドウ右上の× or Quit ボタン"""
    global running, stream
    running = False
    try:
        stream.stop()
        stream.close()
    except Exception as e:
        print("stream close error:", e)
    print("終了しました。")

REF_F0 = 530.0
TARGET_RMS = 0.1

def seikika(x):
    f0_raw = extract_f0(x)
    f0_med = np.median(f0_raw[f0_raw > 0]) if np.any(f0_raw > 0) else REF_F0
    semi = 12 * np.log2((f0_med + 1e-6) / REF_F0)
    seg_ps = librosa.effects.pitch_shift(x, sr=SR, n_steps=-semi)

    rms = np.sqrt(np.mean(seg_ps**2)) + 1e-8
    seg_ps = seg_ps * (TARGET_RMS / rms)
    return seg_ps

def _is_near_existing(x, y):
    # 既存点中心からの正規化距離（楕円距離）
    dx = (x - float(Z2_center[0])) / float(Z2_std[0])
    dy = (y - float(Z2_center[1])) / float(Z2_std[1])
    d = (dx*dx + dy*dy) ** 0.5
    return d <= NEAR_K


def _smooth_set_limits(target_xlim, target_ylim):
    # いまの範囲(cur_*)から目標(target_*)へ滑らかに寄せる
    global cur_xlim, cur_ylim
    cur_xlim[0] = (1 - ZOOM_ALPHA)*cur_xlim[0] + ZOOM_ALPHA*target_xlim[0]
    cur_xlim[1] = (1 - ZOOM_ALPHA)*cur_xlim[1] + ZOOM_ALPHA*target_xlim[1]
    cur_ylim[0] = (1 - ZOOM_ALPHA)*cur_ylim[0] + ZOOM_ALPHA*target_ylim[0]
    cur_ylim[1] = (1 - ZOOM_ALPHA)*cur_ylim[1] + ZOOM_ALPHA*target_ylim[1]
    ax.set_xlim(cur_xlim[0], cur_xlim[1])
    ax.set_ylim(cur_ylim[0], cur_ylim[1])


def update_zoom_by_point(x, y):
    """
    - 遠い時：ズームアウトしてOK（必要なら点が入るように拡げる）
    - 近い時：全体（既存+新規）に合わせてズームイン（狭める）
    """
    global Z2_new

    # 新規点履歴に追加（全体範囲を作るため）
    Z2_new.append((x, y))

    if _is_near_existing(x, y):
        # ====== ズームイン：全体点（既存+新規）の範囲で「狭める」 ======
        all_pts = np.vstack([Z2, np.array(Z2_new, dtype=np.float32)])  # (N,2)
        xmin, xmax = float(all_pts[:,0].min()), float(all_pts[:,0].max())
        ymin, ymax = float(all_pts[:,1].min()), float(all_pts[:,1].max())
        dx = (xmax - xmin) if xmax > xmin else 1.0
        dy = (ymax - ymin) if ymax > ymin else 1.0

        target_xlim = (xmin - PAD*dx, xmax + PAD*dx)
        target_ylim = (ymin - PAD*dy, ymax + PAD*dy)

        _smooth_set_limits(target_xlim, target_ylim)

    else:
        # ====== ズームアウトは許容：点が画面外なら外側へ広げる（縮めない） ======
        changed = False
        if x < cur_xlim[0]:
            cur_xlim[0] = x; changed = True
        if x > cur_xlim[1]:
            cur_xlim[1] = x; changed = True
        if y < cur_ylim[0]:
            cur_ylim[0] = y; changed = True
        if y > cur_ylim[1]:
            cur_ylim[1] = y; changed = True

        if changed:
            # 少し余白
            dx = (cur_xlim[1] - cur_xlim[0]) if cur_xlim[1] > cur_xlim[0] else 1.0
            dy = (cur_ylim[1] - cur_ylim[0]) if cur_ylim[1] > cur_ylim[0] else 1.0
            ax.set_xlim(cur_xlim[0] - PAD*dx, cur_xlim[1] + PAD*dx)
            ax.set_ylim(cur_ylim[0] - PAD*dy, cur_ylim[1] + PAD*dy)


def update_plot():
    """一定間隔で呼ばれて、最新セグメントがあればプロットする"""
    global last_point

    if running:
        seg = get_latest_segment()
        if seg is not None:
            # STFT → mel スペクトログラム

            seg = seikika(seg)
            spec = stft_mel(seg)  # [F, T]
            spect = torch.from_numpy(spec.T).unsqueeze(0).unsqueeze(2).float().to(device)  # [1,T,1,F]

            # エンコーダで潜在ベクトルを取得
            with torch.no_grad():
                mu, lv = encoder(spect)
            z_seg = mu.mean(dim=1)                # [1, latent]
            z_seg_num = z_seg.cpu().numpy()       # (1, latent)

            # PCA で 2 次元に
            Z_segment = pca.transform(z_seg_num)  # (1, 2)

            # 前回のポイントを消したければ remove()
            if last_point is not None:
                try:
                    last_point.remove()
                except ValueError:
                    # すでに消えている等でエラー時は無視
                    pass

            # 新しいポイントを描画（マゼンタ + 黒枠）
            last_point = ax.scatter(
                Z_segment[:, 0], Z_segment[:, 1],
                s=80, c="magenta", edgecolors="black",
                linewidth=1.0, alpha=1.0
            )
            # Z_segment は (1,2) の想定
            x = float(Z_segment[0, 0])
            y = float(Z_segment[0, 1])

            update_zoom_by_point(x, y)      

            # キャンバスを再描画
            canvas.draw()



# ==== Tkinter ウィンドウの準備 ====
import sys
import matplotlib
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class RealTimeMapWindow(QtWidgets.QMainWindow):
    def __init__(self, fig, parent=None):
        super().__init__(parent)

        self.setWindowTitle("リアルタイム音色マップ")

        # --- Figure を Qt のウィジェットに貼り付け ---
        global canvas
        canvas = FigureCanvas(fig)
        self.canvas = canvas
        self.ax = ax

        central = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(central)
        vlayout.addWidget(canvas)

        # --- ボタン行（Start / Stop / Quit） ---
        btn_layout = QtWidgets.QHBoxLayout()

        btn_start = QtWidgets.QPushButton("Start")
        btn_stop  = QtWidgets.QPushButton("Stop")
        btn_quit  = QtWidgets.QPushButton("Quit")

        btn_layout.addWidget(btn_start)
        btn_layout.addWidget(btn_stop)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_quit)

        vlayout.addLayout(btn_layout)

        self.setCentralWidget(central)

        # コールバックを既存の関数に接続
        btn_start.clicked.connect(start_plot)
        btn_stop.clicked.connect(stop_plot)
        btn_quit.clicked.connect(self.handle_quit)

        # Tk の root.after(100, update_plot) 相当 → QTimer を使う
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)              # 100 ms ごと
        self.timer.timeout.connect(update_plot)  # update_plot() を呼ぶ
        self.timer.start()

        # 必要なら canvas / fig をメンバに保持
        self.fig = fig

    def handle_quit(self):
        """Quit ボタンを押したときの処理"""
        on_close()  # 元の終了処理を呼ぶ
        QtWidgets.QApplication.quit()

    def closeEvent(self, event):
        """ウィンドウの × ボタンが押されたときの処理"""
        on_close()  # 元の終了処理を呼ぶ
        event.accept()


def main():
    print("backend:", matplotlib.get_backend())

    app = QtWidgets.QApplication(sys.argv)

    win = RealTimeMapWindow(fig)
    win.resize(1200, 1000)  # Figure(1200x1000) に合わせておく
    win.show()

    print("Mic stream started. GUI 起動中。Start ボタンでプロット開始。")
    print("E: before app.exec_")
    ret = app.exec_()
    print("F: after app.exec_")
    sys.exit(ret)


if __name__ == "__main__":
    main()

