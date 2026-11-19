"""Données -> rendements -> GaussianHMM (choix de N par BIC) -> régimes -> prédiction de transition."""
import numpy as np, pandas as pd, requests
from hmmlearn.hmm import GaussianHMM

def binance_klines(symbole="BTCUSDT", debut="2017-08-17", fin="2026-11-01"):
    """OHLCV journalier via /api/v3/klines, 1000 bougies max par requête -> pagination. Heures en UTC."""
    url, t0, t1, rows = "https://api.binance.com/api/v3/klines", pd.Timestamp(debut, tz="UTC"), pd.Timestamp(fin, tz="UTC"), []
    cur = int(t0.timestamp() * 1000)
    while cur < t1.timestamp() * 1000:
        r = requests.get(url, params=dict(symbol=symbole, interval="1d", startTime=cur, limit=1000)).json()
        if not r: break
        rows += r; cur = r[-1][0] + 86_400_000
    df = pd.DataFrame(rows, columns=["t","Open","High","Low","Close","Volume","ct","qv","n","tb","tq","i"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df[["Open","High","Low","Close","Volume"]].astype(float)
# Alternative / contrôle : yfinance.download("BTC-USD", start="2017-01-01", auto_adjust=True)

def features(df):
    r = np.log(df["Close"]).diff()
    return pd.DataFrame({"rendement": r, "vol20": r.rolling(20).std()}).dropna()

def n_params(N, d):  # covariance pleine
    return (N - 1) + N * (N - 1) + N * d + N * d * (d + 1) // 2

def choisir_N(X, Ns=(2, 3, 4, 5), n_init=10):
    out = {}
    for N in Ns:
        best = None
        for s in range(n_init):                       # plusieurs initialisations (optimum local)
            m = GaussianHMM(N, covariance_type="full", n_iter=200, random_state=s).fit(X)
            if best is None or m.score(X) > best.score(X): best = m
        L, k, T = best.score(X), n_params(N, X.shape[1]), len(X)
        out[N] = dict(AIC=2*k - 2*L, BIC=k*np.log(T) - 2*L, model=best)
    return out

if __name__ == "__main__":
    # df = binance_klines("BTCUSDT")  # données réelles ; ici démonstration sur données simulées
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.normal(.002,.015,300), rng.normal(-.002,.02,250), rng.normal(0,.06,100), rng.normal(.002,.015,300)])
    df = pd.DataFrame({"Close": 100*np.exp(np.cumsum(r))}, index=pd.date_range("2020-01-01", periods=len(r)))
    F = features(df); X = (F - F.mean()) / F.std()   # NB : sur données réelles, fit du scaler sur le train seulement
    res = choisir_N(X.values)
    for N, v in res.items(): print(N, "AIC %.0f BIC %.0f" % (v["AIC"], v["BIC"]))
    N = min(res, key=lambda n: res[n]["BIC"]); m = res[N]["model"]
    reg = m.predict(X.values)                        # Viterbi
    gam = m.predict_proba(X.values)[-1]              # filtré/lissé en dernier point
    print("N* =", N, "| moyennes rendement par régime:", F.groupby(reg)["rendement"].mean().round(4).to_dict())
    for k in (1, 5): print(f"P(S_T+{k}) =", (gam @ np.linalg.matrix_power(m.transmat_, k)).round(3))
