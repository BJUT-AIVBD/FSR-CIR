import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets import IncWrap
from src.metrics import eval_ret
from src.models import BaseModel

log = logging.getLogger(__name__)


def load_cfg(p):
    with open(p) as f:
        return json.load(f)


def train_stage(m, trn_ds, val_ds, cfg, stage, dev):
    tc, lc = cfg["train"], cfg["loss"]
    loader = DataLoader(trn_ds, tc["bs"], shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc["epochs"])
    best, wait = float("inf"), 0

    for ep in range(1, tc["epochs"] + 1):
        m.train()
        tot, nb = 0.0, 0
        for v, t, mx, y in loader:
            v, t, mx, y = v.to(dev), t.to(dev), mx.to(dev), y.to(dev)
            d = m.loss(v, t, mx, y, lc["n_unk"], lc["lam"], lc["a_t"], lc["a_v"])
            opt.zero_grad()
            d["loss"].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += d["loss"].item()
            nb += 1
        sched.step()
        avg = tot / max(nb, 1)

        m.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for v, t, mx, y in DataLoader(val_ds, tc["bs"], shuffle=False):
                v, t, mx, y = v.to(dev), t.to(dev), mx.to(dev), y.to(dev)
                vt += m.loss(v, t, mx, y, lc["n_unk"], lc["lam"], lc["a_t"], lc["a_v"])[
                    "loss"
                ].item()
                vn += 1
        va = vt / max(vn, 1)
        log.info(f"[S{stage}] ep {ep}/{tc['epochs']}  trn={avg:.4f}  val={va:.4f}")

        if va < best:
            best, wait = va, 0
        else:
            wait += 1
            if wait >= tc["patience"]:
                log.info(f"[S{stage}] early stop @ ep {ep}")
                break

    # init centers
    m.eval()
    va, ta, ma, ya = [], [], [], []
    for v, t, mx, y in DataLoader(trn_ds, tc["bs"], shuffle=False):
        va.append(v)
        ta.append(t)
        ma.append(mx)
        ya.append(y)
    va = torch.cat(va).to(dev)
    ta = torch.cat(ta).to(dev)
    ma = torch.cat(ma).to(dev)
    ya = torch.cat(ya).to(dev)
    cls = ya.unique(sorted=True)
    lm = {c.item(): i for i, c in enumerate(cls)}
    yr = torch.tensor([lm[y.item()] for y in ya], device=dev)
    m.init_centers(va, ta, ma, yr)
    return m


def run_train(cfg_path="config.json"):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = load_cfg(cfg_path)
    dc, mc = cfg["data"], cfg["model"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {dev}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    w = IncWrap(dc["root"], dc["n_cls"], dc["n_stages"], dc["val_ratio"], dc["seed"])
    sv, st, sm, _ = w.get(0)[0][0]
    vd, td, md = sv.shape[0], st.shape[0], sm.shape[0]
    log.info(f"dims: v={vd} t={td} m={md}")

    m = BaseModel(
        vd, td, md, mc["emb_dim"], 0, mc["temp"], mc["proj_hid"], mc["proj_drop"]
    ).to(dev)
    results = []

    for s in range(dc["n_stages"]):
        kc = w.known_at(s)
        nk = len(kc)
        log.info(f"=== S{s}: {nk} classes {kc} ===")
        m.clf = torch.nn.Linear(mc["emb_dim"] * 3, nk).to(dev)
        m.n_known = nk

        trn, val = w.get(s)
        m = train_stage(m, trn, val, cfg, s, dev)

        m.eval()
        ve, te, me, ye = val.all_features()
        ve, te, me, ye = ve.to(dev), te.to(dev), me.to(dev), ye.to(dev)
        cls = ye.unique(sorted=True)
        lm = {c.item(): i for i, c in enumerate(cls)}
        yr = torch.tensor([lm[y.item()] for y in ye], device=dev)
        with torch.no_grad():
            ev = F.normalize(m.v_proj(ve), -1)
            et = F.normalize(m.t_proj(te), -1)
            em = F.normalize(m.m_proj(me), -1)
        emb = (ev + et + em) / 3.0
        sim = emb @ emb.T
        sim.fill_diagonal_(-float("inf"))
        r = {k: v.item() for k, v in eval_ret(sim, yr, yr).items() if k != "AP_c"}
        results.append(r)
        log.info(f"[S{s}] {r}")

        torch.save(
            {"stage": s, "sd": m.state_dict(), "nk": nk, "kc": kc, "m": r},
            out / f"s{s}.pt",
        )

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"saved {out / 'results.json'}")


if __name__ == "__main__":
    run_train()
