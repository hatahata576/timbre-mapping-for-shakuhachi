import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from config import (
    SR,
    N_FFT,
    REF_F0,
    F_BINS,
    HARMONICS,
    NOISE_BANDS,
    LATENT_D,
    MEL_BINS,
)


def reparam(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps


class Encoder(nn.Module):
    def __init__(self, latent_d=LATENT_D):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

        self.proj_mu = nn.LazyLinear(latent_d)
        self.proj_lv = nn.LazyLinear(latent_d)

    def forward(self, x):
        # x: [B,T,1,F]
        h = x.permute(0, 2, 1, 3)  # [B,1,T,F]
        h = self.conv(h)            # [B,128,T,F]
        h = h.mean(dim=1)           # [B,T,F]

        mu = self.proj_mu(h)        # [B,T,latent]
        lv = self.proj_lv(h)        # [B,T,latent]
        return mu, lv


class Decoder(nn.Module):
    def __init__(
        self,
        latent_d=LATENT_D,
        harmonics=HARMONICS,
        bands=NOISE_BANDS,
        f_bins=F_BINS,
        mel_bins=MEL_BINS,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()

        in_d = latent_d + 2

        self.mlp = nn.Sequential(
            nn.Linear(in_d, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.head_harm = nn.Linear(256, harmonics)
        self.head_noise = nn.Linear(256, bands)

        self.register_buffer("freq_index", torch.arange(f_bins).float())
        self.harm_width = nn.Parameter(torch.tensor(1.5))
        self.f_bins = f_bins
        self.bands = bands

        fmax = mel_fmax if mel_fmax is not None else SR / 2.0
        mel = librosa.filters.mel(
            sr=SR,
            n_fft=N_FFT,
            n_mels=mel_bins,
            fmin=mel_fmin,
            fmax=fmax,
        ).astype(np.float32)

        mel = torch.from_numpy(mel)
        mel = mel / (mel.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("mel_fb", mel)

    def harmonic_splat(self, f0_hz, amps):
        B, T, H = amps.shape
        F = self.f_bins

        hz_per_bin = SR / N_FFT
        k = torch.arange(1, H + 1, device=amps.device).view(1, 1, H)
        pk = (k * f0_hz.unsqueeze(-1)) / hz_per_bin
        pk = pk.unsqueeze(-1)
        f = self.freq_index.view(1, 1, 1, F).to(amps.device)

        sigma = torch.clamp(self.harm_width, 0.5, 4.0)
        g = torch.exp(-(f - pk) ** 2 / (2 * sigma ** 2))
        spec = (amps.unsqueeze(-1) * g).sum(dim=2)
        return spec

    def band_interp(self, bands):
        B, T, Bn = bands.shape
        x = bands.reshape(B * T, 1, Bn)
        out = Fnn.interpolate(
            x,
            size=self.f_bins,
            mode="linear",
            align_corners=True,
        )
        return out.reshape(B, T, self.f_bins)

    def forward(self, z, cond):
        h = torch.cat([z, cond], dim=-1)
        h = self.mlp(h)

        harm_a = Fnn.softplus(self.head_harm(h)) + 1e-4
        noise_b = Fnn.softplus(self.head_noise(h)) + 1e-4

        f0_hz = cond[..., 0] * 400.0 + REF_F0
        harm_spec = self.harmonic_splat(f0_hz, harm_a)
        noise_spec = self.band_interp(noise_b)
        mag = harm_spec + noise_spec

        mel_mag = torch.matmul(mag, self.mel_fb.T)
        mel_harm = torch.matmul(harm_spec, self.mel_fb.T)
        mel_noise = torch.matmul(noise_spec, self.mel_fb.T)

        eps = 1e-10
        ref = mel_mag.amax(dim=(1, 2), keepdim=True).clamp_min(eps)

        yhat_db = 10.0 * torch.log10((mel_mag / ref).clamp_min(eps))
        harm_db = 10.0 * torch.log10((mel_harm / ref).clamp_min(eps))
        noise_db = 10.0 * torch.log10((mel_noise / ref).clamp_min(eps))

        return yhat_db, harm_db, noise_db


class VAEHarmNoise(nn.Module):
    def __init__(self, latent_d=LATENT_D):
        super().__init__()
        self.enc = Encoder(latent_d=latent_d)
        self.dec = Decoder(latent_d=latent_d)

    def forward(self, x, cond):
        mu, lv = self.enc(x)
        z = reparam(mu, lv)
        yhat, hs, ns = self.dec(z, cond)
        return yhat, mu, lv, hs, ns
