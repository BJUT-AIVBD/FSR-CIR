import torch


def ap(sim, ql, gl):
    Q = ql.size(0)
    aps = torch.zeros(Q, device=sim.device)
    idx = sim.argsort(1, descending=True)
    sl = gl[idx]
    for q in range(Q):
        m = (sl[q] == ql[q]).float()
        nr = m.sum()
        if nr == 0:
            continue
        pos = torch.arange(1, m.size(0) + 1, device=m.device).float()
        p = m.cumsum(0) / pos
        aps[q] = (p * m).sum() / nr
    return aps


def mAP(sim, ql, gl):
    ul = gl.unique(sorted=True)
    aps = []
    for c in ul.tolist():
        mask = ql == c
        if mask.sum() == 0:
            continue
        aps.append(ap(sim[mask], ql[mask], gl).mean())
    if not aps:
        return {"mAP": torch.tensor(0.0, device=sim.device), "AP_c": torch.tensor([])}
    t = torch.stack(aps)
    return {"mAP": t.mean(), "AP_c": t}


def rpk(sim, ql, gl, ks=(1, 5, 10)):
    idx = sim.argsort(1, descending=True)
    sl = gl[idx]
    r = {}
    for k in ks:
        tk = sl[:, :k]
        qe = ql.unsqueeze(1).expand_as(tk)
        h = (tk == qe).float().sum(1)
        nr = torch.zeros_like(ql, dtype=torch.float)
        for q in range(ql.size(0)):
            nr[q] = (gl == ql[q]).float().sum()
        r[f"R@{k}"] = (h / nr.clamp(min=1)).mean()
    return r


def eval_ret(sim, ql, gl):
    return {**mAP(sim, ql, gl), **rpk(sim, ql, gl)}
