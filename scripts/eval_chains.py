"""Multi-hop chain evaluation: per-edge drift, path z-test, elevated-hop-count tests, end-to-end check."""
import json, os, sys
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, "scripts")
import eval_v2 as E
from custody.detectors_v2 import Detector
from gen_chains import ROUTE
from extract_chain import chain_files

ROOT = "E:/DF paper/"
NC = E.NCOL
CD = list(np.load("work/chain_done.npz", allow_pickle=True)["features"].item().keys())
CP = np.load("work/chain_pairs.npz", allow_pickle=True)["features"].item()
nrm = lambda p: p.replace(chr(92), "/").replace(ROOT, "")
FE = E._EXT


def f12(p):
    q = p.replace(chr(92), "/")
    return (FE[q] if q in FE else FE[nrm(p)] if nrm(p) in FE else FE[ROOT + nrm(p)])[:NC]


def edge(key, k):
    scen, route, i, op = key
    f = chain_files(*key)
    src = T_native = E.T[i]["native"] if k == 1 else f[k - 2]
    tgt = f[k - 1]
    pair = E._PAIRS[(E.T[i]["native"].replace(chr(92), "/"), tgt.replace(chr(92), "/"))] if k == 1 else CP[key][("hop", k)]
    return np.r_[f12(src), np.zeros(5)], np.r_[f12(tgt), pair]


def e2e(key, k):
    scen, route, i, op = key
    f = chain_files(*key)
    return E.N[i], np.r_[f12(f[k - 1]), CP[key][("e2e", k)]]


def etype(route, k):
    return (ROUTE[route][k - 1], "first" if k == 1 else "later")


KEYS = [k for k in CD if k in CP]
GEN = [k for k in KEYS if k[0] == "genuine"]
DEV = E.DEV
res = {}
cont = {}  # scen -> m -> stat -> list
def rec(scen, m, name, flag):
    res.setdefault(scen, {}).setdefault(m, {}).setdefault(name, []).append(bool(flag))

for d in E.DEVS:
    tr = [k for k in GEN if DEV[k[2]] != d]
    dets, rowmap = {}, {}
    for tp in {etype(r, k) for r in ROUTE for k in (1, 2, 3, 4)}:
        rows = [(k, kk) for k in tr for kk in (1, 2, 3, 4) if etype(k[1], kk) == tp]
        S = np.array([edge(a, b)[0] for a, b in rows]); V = np.array([edge(a, b)[1] for a, b in rows])
        det = Detector(joint_cols=list(range(NC))).fit(S, V, np.array([DEV[a[2]] for a, b in rows]))
        dets[tp] = det
        for j, (a, b) in enumerate(rows):
            rowmap[(a, b)] = det.cv["nll"][j]
    # genuine cumulative null per route / prefix
    null = {}
    for r in ROUTE:
        ch = [k for k in tr if k[1] == r]
        cum = np.cumsum(np.array([[rowmap[(c, kk)] for kk in (1, 2, 3, 4)] for c in ch]), axis=1)
        null[r] = (cum.mean(0), cum.std(0) + 1e-6)
    e2edets = {}
    for r in ROUTE:
        for m in (2, 3, 4):
            ch = [k for k in tr if k[1] == r]
            S = np.array([e2e(c, m)[0] for c in ch]); V = np.array([e2e(c, m)[1] for c in ch])
            e2edets[(r, m)] = Detector(joint_cols=list(range(NC))).fit(S, V, np.array([DEV[c[2]] for c in ch]))
    for key in [k for k in KEYS if DEV[k[2]] == d]:
        scen, route, i, op = key
        xs, sd, ss = [], [], []
        for k in (1, 2, 3, 4):
            S, V = edge(key, k)
            det = dets[etype(route, k)]
            xs.append(det.core.raw(S[None], V[None])["nll"][0])
            sc = det.scores(S[None], V[None])
            sd.append(sc["drift_only"][0]); ss.append(sc["snapshot_conflict"][0])
        xs = np.array(xs)
        ranks = np.array([np.mean(dets[etype(route, k)].cv["nll"] <= xs[k - 1]) for k in (1, 2, 3, 4)])
        cum = np.cumsum(xs)
        for m in (2, 3, 4):
            ea = any(np.array(sd[:m]) > 0); sa = any(np.array(ss[:m]) > 0)
            z = abs((cum[m - 1] - null[route][0][m - 1]) / null[route][1][m - 1]) > 3.0
            ne = int((ranks[:m] > 0.70).sum())
            p_hi = binomtest(ne, m, 0.3, alternative="greater").pvalue; p_lo = binomtest(ne, m, 0.3, alternative="less").pvalue
            e1 = p_hi < 0.05; e2 = (p_hi < 0.025) or (p_lo < 0.025)
            S2, V2 = e2e(key, m)
            ee = e2edets[(route, m)].scores(S2[None], V2[None])["drift_only"][0] > 0
            zc = abs((cum[m - 1] - null[route][0][m - 1]) / null[route][1][m - 1])
            ee_s = e2edets[(route, m)].scores(S2[None], V2[None])["drift_only"][0]
            sc_name = scen if scen != "genuine" else "genuine"
            for nm2, v2 in (("max_edge_drift", max(sd[:m])), ("path_z", zc), ("e2e_drift", ee_s), ("elev_count", float(ne))):
                cont.setdefault(sc_name, {}).setdefault(m, {}).setdefault(nm2, []).append(v2)
            for nm, fl in (("edge_drift", ea), ("edge_snapshot", sa), ("z_test", z), ("elev_1sided", e1), ("elev_2sided", e2),
                           ("path_z_or_elev", z or e2), ("e2e_drift", ee), ("edge_or_path", ea or z or e2), ("edge_or_path_or_e2e", ea or z or e2 or ee)):
                rec(scen if scen != "genuine" else f"genuine_{route}", m, nm, fl)
                if scen == "genuine":
                    rec("genuine_all", m, nm, fl)
out = {sc: {str(m): {n: float(np.mean(v)) for n, v in dd.items()} | {"n": len(next(iter(dd.values())))} for m, dd in mm.items()} for sc, mm in res.items()}
json.dump(out, open("scripts/eval_chains.json", "w"), indent=1)
names = ["edge_drift", "edge_snapshot", "z_test", "elev_1sided", "elev_2sided", "path_z_or_elev", "e2e_drift", "edge_or_path", "edge_or_path_or_e2e"]
for sc in out:
    print("==", sc)
    print("%3s %5s " % ("m", "n") + " ".join("%11s" % n[:11] for n in names))
    for m in sorted(out[sc]):
        print("%3s %5d " % (m, out[sc][m]["n"]) + " ".join("%11.3f" % out[sc][m][n] for n in names))

from sklearn.metrics import roc_auc_score
print("== chain-level AUC, genuine vs tampered (all genuine chains, both routes)")
auc = {}
for scen in ("tamper2", "creep"):
    for m in (2, 3, 4):
        row = {}
        for nm in ("max_edge_drift", "path_z", "e2e_drift", "elev_count"):
            g = cont["genuine"][m][nm]; a = cont[scen][m][nm]
            row[nm] = float(roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(a))], np.r_[g, a]))
        auc[f"{scen}_{m}"] = row
        print(scen, m, {k: round(v, 3) for k, v in row.items()})
json.dump(auc, open("scripts/eval_chains_auc.json", "w"), indent=1)
