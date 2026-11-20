import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- DATA LOADING
def charger_donnees(symbole, debut, fin):
    """Daily OHLCV from Yahoo Finance (used as a control source)."""
    df = yf.download(symbole, start=debut, end=fin, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def charger_binance(symbole="BTCUSDT", intervalle="8h", debut="2017-08-17", fin="2026-11-01"):
    """Binance candles (1h, 4h, 8h, 1d...) with pagination (1000 candles per request)."""
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
    return df.set_index("Date")[["Close", "High", "Low", "Open", "Volume"]].astype(float)


def charger_donnees_8h(symbole="BTCUSDT", debut="2018-01-01", fin="2026-11-01"):
    """Kept for compatibility: 8h candles = 3 rows per day."""
    return charger_binance(symbole, "8h", debut, fin)


# ---------------------------------------------------------------- INDICATORS
def calculer_rendement_log(prix):
    """Log-return: r_t = ln(P_t / P_{t-1})."""
    return np.log(prix / prix.shift(1))


def calculer_volatilite(rendements, fenetre=20):
    """Rolling standard deviation of the returns."""
    return rendements.rolling(window=fenetre).std()


def calculer_RSI(prix, periode=14):
    """RSI between 0 and 100."""
    variation = prix.diff()
    gain = variation.clip(lower=0)
    perte = -variation.clip(upper=0)
    gain_moyen = gain.ewm(alpha=1 / periode, min_periods=periode).mean()
    perte_moyenne = perte.ewm(alpha=1 / periode, min_periods=periode).mean()
    return 100 - (100 / (1 + gain_moyen / perte_moyenne))


def calculer_MACD(prix, court=12, long=26, signal=9):
    """MACD = EMA(court) - EMA(long); signal line = EMA(signal) of the MACD."""
    macd = prix.ewm(span=court, adjust=False).mean() - prix.ewm(span=long, adjust=False).mean()
    ligne_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, ligne_signal, macd - ligne_signal


def standardiser(X):
    """Mean 0, std 1 (returns the scaler so it can be reused on the test set)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ---------------------------------------------------------------- FEATURES
def construire_features(df, fen_vol=60, fen_rsi=42, macd_params=(36, 78, 27), fen_volume=20):
    """All features from an OHLCV DataFrame. Defaults are for 8h candles (3 per day).
    Every feature only uses PAST bars (no look-ahead)."""
    f = pd.DataFrame(index=df.index)
    f["rendement"] = calculer_rendement_log(df["Close"])
    f["volatilite"] = calculer_volatilite(f["rendement"], fen_vol)
    f["RSI"] = calculer_RSI(df["Close"], fen_rsi)
    macd, sig, histo = calculer_MACD(df["Close"], *macd_params)
    f["MACD_norm"] = macd / df["Close"]
    f["histo_norm"] = histo / df["Close"]
    f["amplitude"] = (df["High"] - df["Low"]) / df["Close"]
    # relative volume: compared with the same session (00h/08h/16h) over the last days, past only
    v_session = df["Volume"].groupby(df.index.hour).transform(
        lambda s: s.rolling(fen_volume).mean().shift(1))
    f["vol_rel"] = np.log(df["Volume"] / v_session).replace([np.inf, -np.inf], np.nan)
    return f
