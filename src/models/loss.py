import torch
import torch.nn.functional as F


def fsr_loss(v, t, labels, alpha_t=1.0, alpha_v=1.0, lam=10.0):
    s_a = -(lam * F.cosine_similarity(v, t, dim=-1)).mean()

    classes = labels.unique()
    su = torch.tensor(0.0, device=v.device)
    for c in classes:
        m = labels == c
        su += torch.exp(lam * F.cosine_similarity(v[m].mean(0), t[m].mean(0), dim=0))
    s_u = -torch.log(su).mean()

    cv, ct = [], []
    for c in classes:
        m = labels == c
        cv.append(v[m].mean(0))
        ct.append(t[m].mean(0))
    cv, ct = torch.stack(cv), torch.stack(ct)
    I = cv.size(0)
    if I > 1:
        eye = torch.eye(I, device=cv.device, dtype=torch.bool)
        s_p = (
            alpha_v * (cv @ cv.T).masked_fill(eye, 0).sum()
            + alpha_t * (ct @ ct.T).masked_fill(eye, 0).sum()
        ) / (I * (I - 1))
    else:
        s_p = torch.tensor(0.0, device=v.device)

    return s_a + s_u + s_p


def dpfg_loss(v, t, n_unk=6, lam=10.0):
    d = v.device
    i1 = torch.randint(0, v.size(0), (n_unk,), device=d)
    i2 = torch.randint(0, v.size(0), (n_unk,), device=d)
    a = torch.rand(n_unk, 1, device=d)
    theta = F.normalize(a * v[i1] + (1 - a) * v[i2], dim=-1)
    cn = F.normalize(t[torch.randint(0, t.size(0), (n_unk,), device=d)], dim=-1)
    s_a2 = -(lam * F.cosine_similarity(theta, cn, dim=-1)).mean()
    S = F.cosine_similarity(theta.unsqueeze(1), cn.unsqueeze(0), dim=-1)
    s_u2 = -torch.log(S.diag().exp() / S.exp().sum(1)).mean()
    return s_a2 + s_u2
