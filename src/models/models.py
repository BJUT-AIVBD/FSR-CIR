import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.loss import dpfg_loss, fsr_loss


class Proj(nn.Module):
    def __init__(self, d_in, d_out, hid=None, drop=0.0):
        super().__init__()
        if hid is None:
            hid = [d_in * 2, d_in, d_out * 2]
        ls = []
        p = d_in
        for h in hid:
            ls += [nn.Linear(p, h), nn.ReLU(True)]
            if drop > 0:
                ls.append(nn.Dropout(drop))
            p = h
        ls.append(nn.Linear(p, d_out))
        self.net = nn.Sequential(*ls)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class BaseModel(nn.Module):
    def __init__(
        self,
        v_dim,
        t_dim,
        m_dim,
        emb_dim=256,
        n_known=20,
        temp=0.07,
        proj_hid=None,
        proj_drop=0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.temp = temp
        self.n_known = n_known
        self.v_proj = Proj(v_dim, emb_dim, proj_hid, proj_drop)
        self.t_proj = Proj(t_dim, emb_dim, proj_hid, proj_drop)
        self.m_proj = Proj(m_dim, emb_dim, proj_hid, proj_drop)
        self.clf = nn.Linear(emb_dim * 3, n_known)

    def forward(self, v, t, mx):
        return {"v": self.v_proj(v), "t": self.t_proj(t), "m": self.m_proj(mx)}

    def _cv(self):
        return F.normalize(self.clf.weight.data[:, : self.emb_dim], -1)

    def _ct(self):
        return F.normalize(self.clf.weight.data[:, self.emb_dim : self.emb_dim * 2], -1)

    def _cm(self):
        return F.normalize(self.clf.weight.data[:, self.emb_dim * 2 :], -1)

    def loss(self, v, t, mx, labels, n_unk=6, lam=10.0, a_t=1.0, a_v=1.0):
        out = self.forward(v, t, mx)
        ve, te, me = out["v"], out["t"], out["m"]
        l1 = fsr_loss(ve, te, labels, a_t, a_v, lam)
        l1 += fsr_loss(ve, me, labels, a_t, a_v, lam)
        l1 += fsr_loss(te, me, labels, a_t, a_v, lam)
        l2 = dpfg_loss(ve, te, n_unk, lam)
        l2 += dpfg_loss(ve, me, n_unk, lam)
        l2 += dpfg_loss(te, me, n_unk, lam)
        logits = self.clf(torch.cat([ve, te, me], -1))
        l3 = F.cross_entropy(logits, labels)
        return {"loss": l1 + l2 + l3, "l_u1": l1, "l_u2": l2, "l_cls": l3}

    @torch.no_grad()
    def init_centers(self, vf, tf, mf, labels):
        ve = F.normalize(self.v_proj(vf), -1)
        te = F.normalize(self.t_proj(tf), -1)
        me = F.normalize(self.m_proj(mf), -1)
        cls = labels.unique(sorted=True)
        cv, ct, cm = [], [], []
        for c in cls:
            m = labels == c
            cv.append(ve[m].mean(0))
            ct.append(te[m].mean(0))
            cm.append(me[m].mean(0))
        with torch.no_grad():
            self.clf.weight.data[:, : self.emb_dim] = torch.stack(cv)
            self.clf.weight.data[:, self.emb_dim : self.emb_dim * 2] = torch.stack(ct)
            self.clf.weight.data[:, self.emb_dim * 2 :] = torch.stack(cm)
            nn.init.zeros_(self.clf.bias)

    @torch.no_grad()
    def retrieve(self, v, t, mx, thr=0.5):
        ve = F.normalize(self.v_proj(v), -1)
        te = F.normalize(self.t_proj(t), -1)
        me = F.normalize(self.m_proj(mx), -1)
        cv, ct, cm = self._cv(), self._ct(), self._cm()
        pv = F.softmax((ve @ cv.T) / self.temp, -1)
        pt = F.softmax((te @ ct.T) / self.temp, -1)
        pm = F.softmax((me @ cm.T) / self.temp, -1)
        P = (pv + pt + pm) / 3.0
        mx, pred = P.max(-1)
        pred[mx < thr] = -1
        return pred, mx

    @torch.no_grad()
    def add_center(self, v, t, mx):
        ve = F.normalize(self.v_proj(v), -1)
        te = F.normalize(self.t_proj(t), -1)
        me = F.normalize(self.m_proj(mx), -1)
        nc = torch.cat([ve.mean(0, True), te.mean(0, True), me.mean(0, True)], -1)
        n = self.clf.out_features
        old_w, old_b = self.clf.weight.data, self.clf.bias.data
        cls = nn.Linear(self.emb_dim * 3, n + 1, bias=True, device=old_w.device)
        cls.weight[:n] = old_w
        cls.weight[n:] = nc
        cls.bias[:n] = old_b
        cls.bias[n] = 0.0
        self.clf = cls
        self.n_known = n + 1
