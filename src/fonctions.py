import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


def charger_donnees(symbole, debut, fin):
    """Télécharge les données OHLCV journalières depuis Yahoo Finance."""
    df = yf.download(symbole, start=debut, end=fin, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


def charger_donnees_8h(symbole="BTCUSDT", debut="2018-01-01", fin="2026-11-01"):
    """Télécharge les bougies de 8h depuis Binance (3 lignes par jour)."""
    import requests, time
    url = "https://api.binance.com/api/v3/klines"
    t = int(pd.Timestamp(debut, tz="UTC").timestamp() * 1000)
    t_fin = int(pd.Timestamp(fin, tz="UTC").timestamp() * 1000)
    lignes = []
    while t < t_fin:
        r = requests.get(url, params={"symbol": symbole, "interval": "8h",
                                      "startTime": t, "endTime": t_fin,
                                      "limit": 1000}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        lignes += data
        t = data[-1][0] + 1
        time.sleep(0.2)
    df = pd.DataFrame(lignes, columns=[
        "Date", "Open", "High", "Low", "Close", "Volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
    df = df.set_index("Date")[["Close", "High", "Low", "Open", "Volume"]].astype(float)
    return df


def calculer_rendement_log(prix):
    """Rendement log : r_t = ln(P_t / P_{t-1})."""
    return np.log(prix / prix.shift(1))


def calculer_volatilite(rendements, fenetre=20):
    """Écart-type des rendements sur une fenêtre glissante."""
    return rendements.rolling(window=fenetre).std()


def calculer_RSI(prix, periode=14):
    """RSI : indicateur entre 0 et 100."""
    variation = prix.diff()
    gain = variation.clip(lower=0)
    perte = -variation.clip(upper=0)

    gain_moyen = gain.ewm(alpha=1/periode, min_periods=periode).mean()
    perte_moyenne = perte.ewm(alpha=1/periode, min_periods=periode).mean()

    rs = gain_moyen / perte_moyenne
    return 100 - (100 / (1 + rs))


def calculer_MACD(prix, court=12, long=26, signal=9):
    """MACD = EMA(12) - EMA(26). Signal = EMA(9) du MACD."""
    ema_court = prix.ewm(span=court, adjust=False).mean()
    ema_long = prix.ewm(span=long, adjust=False).mean()
    macd = ema_court - ema_long
    ligne_signal = macd.ewm(span=signal, adjust=False).mean()
    histogramme = macd - ligne_signal
    return macd, ligne_signal, histogramme


def standardiser(X):
    """Centre et réduit : moyenne 0, écart-type 1."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std, scaler


def charger_binance(symbole="BTCUSDT", intervalle="8h", debut="2017-08-17", fin="2026-11-01"):
    """Télécharge des bougies Binance (1h, 4h, 8h, 1d...) avec pagination."""
    import requests, time
    url = "https://api.binance.com/api/v3/klines"
    t = int(pd.Timestamp(debut, tz="UTC").timestamp() * 1000)
    t_fin = int(pd.Timestamp(fin, tz="UTC").timestamp() * 1000)
    lignes = []
    while t < t_fin:
        r = requests.get(url, params={"symbol": symbole, "interval": intervalle,
                                      "startTime": t, "endTime": t_fin,
                                      "limit": 1000}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        lignes += data
        t = data[-1][0] + 1
        time.sleep(0.2)
    df = pd.DataFrame(lignes, columns=[
        "Date", "Open", "High", "Low", "Close", "Volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
    df = df.set_index("Date")[["Close", "High", "Low", "Open", "Volume"]].astype(float)
    return df
