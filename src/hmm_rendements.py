"""Version script du notebook 04 : BTC 8 h -> GaussianHMM (choix de N par AIC/BIC) -> régimes -> prédiction.
Utilise les fichiers produits par notebooks/01_donnees.ipynb. À lancer depuis la racine du projet :
    python src/hmm_rendements.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

DATA = Path(__file__).resolve().parent.parent / "data"
PAR_JOUR = 3  # bougies de 8 h : 1 jour = 3 lignes


def charger():
    X_train = pd.read_csv(DATA / "btcusdt_X_train.csv", index_col=0, parse_dates=True)
    X_test = pd.read_csv(DATA / "btcusdt_X_test.csv", index_col=0, parse_dates=True)
    scaler = joblib.load(DATA / "btcusdt_scaler.joblib")
    return X_train, X_test, scaler


def n_params(N, d):  # covariance pleine
    return (N - 1) + N * (N - 1) + N * d + N * d * (d + 1) // 2


def choisir_N(X, Ns=(2, 3, 4, 5), n_init=8):
    out = {}
    for N in Ns:
        best, best_ll = None, -np.inf
        for s in range(n_init):                       # plusieurs initialisations (optimum local)
            m = GaussianHMM(N, covariance_type="full", n_iter=300, tol=1e-4, random_state=s)
            try:
                m.fit(X); ll = m.score(X)
            except Exception:
                continue
            if ll > best_ll: best, best_ll = m, ll
        k, T = n_params(N, X.shape[1]), len(X)
        out[N] = dict(logL=best_ll, AIC=2 * k - 2 * best_ll, BIC=k * np.log(T) - 2 * best_ll, model=best)
    return out


if __name__ == "__main__":
    X_train, X_test, scaler = charger()
    print(f"train {X_train.index[0]} -> {X_train.index[-1]} ({len(X_train)} lignes) | "
          f"test {X_test.index[0]} -> {X_test.index[-1]} ({len(X_test)} lignes)")
    res = choisir_N(X_train.values)
    for N, v in res.items():
        print(f"N={N}  logL {v['logL']:.0f}  AIC {v['AIC']:.0f}  BIC {v['BIC']:.0f}")
    N = min(res, key=lambda n: res[n]["BIC"]); m = res[N]["model"]
    print("N* (BIC) =", N)
    mu = pd.DataFrame(scaler.inverse_transform(m.means_), columns=X_train.columns)
    mu["durée moyenne (jours)"] = 1 / (1 - np.diag(m.transmat_)) / PAR_JOUR
    print(mu.round(5))
    gam = m.predict_proba(X_test.values)[-1]         # dernier point du test (filtré = lissé au dernier instant)
    for k in (1, PAR_JOUR):                          # bougie suivante, puis 1 jour plus tard
        print(f"P(S_T+{k}) =", (gam @ np.linalg.matrix_power(m.transmat_, k)).round(3))
