import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.datasets import IncWrap
from src.metrics import eval_ret
from src.models import BaseModel

log = logging.getLogger(__name__)


def load_cfg(p):
    with open(p) as f:
        return json.load(f)


def run_test(cfg_path="config.json"):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = load_cfg(cfg_path)
    dc, mc = cfg["data"], cfg["model"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {dev}")

    out = Path(cfg["output_dir"])
    w = IncWrap(dc["root"], dc["n_cls"], dc["n_stages"], dc["val_ratio"], dc["seed"])
    sv, st, sm, _ = w.get(0)[0][0]
    vd, td, md = sv.shape[0], st.shape[0], sm.shape[0]

    results = []
    for s in range(dc["n_stages"]):
        kc = w.known_at(s)
        nk = len(kc)
        log.info(f"=== S{s}: {nk} classes {kc} ===")

        ckpt = out / f"s{s}.pt"
        if not ckpt.exists():
            log.warning(f"no {ckpt}, skip")
            continue

        m = BaseModel(
            vd, td, md, mc["emb_dim"], nk, mc["temp"], mc["proj_hid"], mc["proj_drop"]
        ).to(dev)
        m.load_state_dict(torch.load(ckpt, dev, weights_only=True)["sd"])
        m.eval()

        _, val = w.get(s)
        ve, te, me, ye = val.all_features()
        ve, te, me, ye = ve.to(dev), te.to(dev), me.to(dev), ye.to(dev)

        with torch.no_grad():
            ev = F.normalize(m.v_proj(ve), -1)
            et = F.normalize(m.t_proj(te), -1)
            em = F.normalize(m.m_proj(me), -1)
        emb = (ev + et + em) / 3.0
        sim = emb @ emb.T
        sim.fill_diagonal_(-float("inf"))

        r = {k: v.item() for k, v in eval_ret(sim, ye, ye).items() if k != "AP_c"}
        results.append(r)
        log.info(f"[S{s}] {r}")

    log.info("=== summary ===")
    for i, r in enumerate(results):
        log.info(f"S{i}: {r}")
    with open(out / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_test()
