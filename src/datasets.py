import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_RE = re.compile(r"^(\d+)_(\d+)\.npy$")


class BJUTHCDMeta(Dataset):
    def __init__(self, root, classes, split="train", val_ratio=0.2, seed=42):
        super().__init__()
        self.root = Path(root)
        self.classes = sorted(classes)

        vd, td, md = self.root / "v", self.root / "t", self.root / "mix"
        samp = {}
        for vf in sorted(vd.glob("*.npy")):
            m = _RE.match(vf.name)
            if not m:
                continue
            vid, cid = int(m.group(1)), int(m.group(2))
            tf, mf = td / vf.name, md / vf.name
            if not tf.exists() or not mf.exists() or cid not in self.classes:
                continue
            samp.setdefault(cid, []).append((vid, str(vf), str(tf), str(mf)))

        rng = np.random.RandomState(seed)
        self._vp, self._tp, self._mp, self._y = [], [], [], []
        for c in sorted(samp):
            items = sorted(samp[c])
            rng.shuffle(items)
            nv = max(1, int(len(items) * val_ratio))
            ch = (
                items[:nv]
                if split == "val"
                else (items[nv:] if split == "train" else items)
            )
            for _, vp, tp, mp in ch:
                self._vp.append(vp)
                self._tp.append(tp)
                self._mp.append(mp)
                self._y.append(c)
        self.n_cls = len(self.classes)

    def __len__(self):
        return len(self._y)

    def __getitem__(self, i):
        v = torch.from_numpy(np.load(self._vp[i])).float()
        t = torch.from_numpy(np.load(self._tp[i])).float()
        mx = torch.from_numpy(np.load(self._mp[i])).float()
        return v, t, mx, self._y[i]

    def all_features(self):
        v, t, mx, y = [], [], [], []
        for i in range(len(self)):
            vi, ti, mi, yi = self[i]
            v.append(vi)
            t.append(ti)
            mx.append(mi)
            y.append(yi)
        return torch.stack(v), torch.stack(t), torch.stack(mx), torch.tensor(y)


class IncWrap:
    def __init__(self, root, n_cls=20, n_stages=4, val_ratio=0.2, seed=42):
        self.root = Path(root)
        self.n_cls = n_cls
        self.n_stages = n_stages
        self.vr = val_ratio
        self.seed = seed
        ch = n_cls // n_stages
        self._stages = []
        for s in range(n_stages):
            e = s * ch + ch if s < n_stages - 1 else n_cls
            self._stages.append(list(range(s * ch, e)))

    def known_at(self, s):
        k = []
        for i in range(s + 1):
            k.extend(self._stages[i])
        return sorted(k)

    def get(self, s):
        k = self.known_at(s)
        return (
            BJUTHCDMeta(self.root, k, "train", self.vr, self.seed),
            BJUTHCDMeta(self.root, k, "val", self.vr, self.seed),
        )
