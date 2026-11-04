import torch
import torch.nn.functional as Fnn
from tqdm import tqdm


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(model, loader, optimizer, device, beta_kl=3e-4):
    model.train()
    total, n = 0.0, 0

    pbar = tqdm(loader, desc="train", leave=False)
    for xb, cb, yb, _ in pbar:
        T = min(xb.shape[1], cb.shape[1], yb.shape[1])
        xb = xb[:, :T].to(device)
        cb = cb[:, :T].to(device)
        yb = yb[:, :T].to(device)

        optimizer.zero_grad()
        yhat, mu, lv, _, _ = model(xb, cb)

        l_rec = Fnn.l1_loss(yhat, yb)
        l_kl = kl_loss(mu, lv)
        loss = l_rec + beta_kl * l_kl

        pbar.set_postfix(
            rec=float(l_rec.item()),
            kl=float(l_kl.item()),
            tot=float(loss.item()),
        )

        loss.backward()
        optimizer.step()

        total += loss.item()
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, beta_kl=3e-4):
    model.eval()
    total, n = 0.0, 0

    pbar = tqdm(loader, desc="valid", leave=False)
    for xb, cb, yb, _ in pbar:
        T = min(xb.shape[1], cb.shape[1], yb.shape[1])
        xb = xb[:, :T].to(device)
        cb = cb[:, :T].to(device)
        yb = yb[:, :T].to(device)

        yhat, mu, lv, _, _ = model(xb, cb)

        l_rec = Fnn.l1_loss(yhat, yb)
        l_kl = kl_loss(mu, lv)
        loss = l_rec + beta_kl * l_kl

        pbar.set_postfix(
            rec=float(l_rec.item()),
            kl=float(l_kl.item()),
        )

        total += loss.item()
        n += 1

    return {"loss": total / max(n, 1)}
