"""Detectors on the v2 (real) feature vectors, with device-disjoint calibration.

Per fold: models are fit on training-device genuine edges only, and every threshold or
score scale is derived from inner leave-one-device-out scores of those same training
devices, so held-out devices never touch fitting or calibration.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

DRIFT_COLS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]   # all but the categorical profile field
METHODS = ("snapshot_conflict", "mismatch_only", "drift_only", "coevasion_only", "full_system",
           "isolation_forest", "ocsvm")


class Core:
    """Fitted models (no thresholds)."""

    def __init__(self, reg=0.01, drift_cols=None, joint_cols=None):
        self.reg = reg
        self._drift_cols, self._joint_cols = drift_cols, joint_cols

    def fit(self, nat, tgt):
        nf = nat.shape[1]
        self.drift_cols = [j for j in range(nf) if j != 3] if self._drift_cols is None else list(self._drift_cols)
        self.joint_cols = list(range(nf)) if self._joint_cols is None else list(self._joint_cols)
        allf = np.vstack([nat, tgt])
        self.scale = allf.std(axis=0) + 1e-6
        d = self._delta(nat, tgt)
        self.gmm = GaussianMixture(1, covariance_type="full", reg_covar=self.reg, random_state=0).fit(d)
        z = (tgt / self.scale)[:, self.joint_cols]
        self.mu = z.mean(axis=0)
        self.prec = LedoitWolf().fit(z).precision_
        self.iso = IsolationForest(random_state=0).fit(d)
        self.oc = OneClassSVM(nu=0.1, gamma="scale").fit(d)
        return self

    def _delta(self, nat, tgt):
        return ((tgt - nat) / self.scale)[:, self.drift_cols]

    def raw(self, nat, tgt):
        d = self._delta(nat, tgt)
        z = (tgt / self.scale)[:, self.joint_cols] - self.mu
        return {
            "base": np.linalg.norm(d, axis=1),
            "nll": -self.gmm.score_samples(d),
            "dist": np.sqrt(np.einsum("ij,jk,ik->i", z, self.prec, z)),
            "iso": -self.iso.decision_function(d),
            "oc": -self.oc.decision_function(d),
        }


class Detector:
    def __init__(self, reg=0.01, hi_pct=90.0, band=(5.0, 95.0), drift_cols=None, joint_cols=None):
        self.reg, self.hi_pct, self.band = reg, hi_pct, band
        self.kw = dict(drift_cols=drift_cols, joint_cols=joint_cols)

    def fit(self, nat, tgt, groups):
        self.core = Core(self.reg, **self.kw).fit(nat, tgt)
        cv = {k: np.zeros(len(nat)) for k in ("base", "nll", "dist", "iso", "oc")}
        for g in np.unique(groups):
            te = groups == g
            c = Core(self.reg, **self.kw).fit(nat[~te], tgt[~te])
            r = c.raw(nat[te], tgt[te])
            for k in cv:
                cv[k][te] = r[k]
        self.cv = cv
        self.cal = {}
        for k in ("base", "nll", "iso", "oc"):
            tau = np.percentile(cv[k], self.hi_pct)
            sc = max(np.percentile(cv[k], 75) - np.percentile(cv[k], 25), 1e-6) / 1.349
            self.cal[k] = (tau, sc)
        lo, hi = np.percentile(cv["dist"], self.band[0]), np.percentile(cv["dist"], self.band[1])
        self.lo, self.hi = lo, hi
        self.sc_d = max(np.percentile(cv["dist"], 75) - np.percentile(cv["dist"], 25), 1e-6) / 1.349
        return self

    def scores(self, nat, tgt):
        r = self.core.raw(nat, tgt)
        sn = lambda k: (r[k] - self.cal[k][0]) / self.cal[k][1]
        s_drift = sn("nll")
        s_coev = np.maximum(r["dist"] - self.hi, self.lo - r["dist"]) / self.sc_d
        return {
            "snapshot_conflict": sn("base"),
            "mismatch_only": (r["dist"] - self.hi) / self.sc_d,
            "drift_only": s_drift,
            "coevasion_only": s_coev,
            "full_system": np.maximum(s_drift, s_coev),
            "isolation_forest": sn("iso"),
            "ocsvm": sn("oc"),
        }
