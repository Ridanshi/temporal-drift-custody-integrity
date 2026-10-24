import os, sys
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "scripts")
import eval_v2 as E

forged = E.load_forged(("naive", "platform"))
ks = list(forged)
Xp = np.array([forged[k] for k in ks]); dp = np.array([E.DEV[k[1]] for k in ks])
os.makedirs("work/gbm_folds", exist_ok=True)
for d in E.DEVS:
    X = np.vstack([E.W[E.DEV != d], Xp[dp != d]]); y = np.r_[np.zeros((E.DEV != d).sum()), np.ones((dp != d).sum())]
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=0).fit(X, y)
    joblib.dump(clf, f"work/gbm_folds/{d}.joblib")
print("saved", len(E.DEVS), "ncols", X.shape[1])
