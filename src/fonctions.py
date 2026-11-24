"""Fonctions utilitaires du projet : données Bitcoin (bougies de 8 h), indicateurs, features."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PAR_JOUR = 3  # bougies de 8 h : 00h, 08h, 16h UTC -> 1 jour = 3 lignes


# ---------------------------------------------------------------- DONNÉES
def charger_binance(symbole="BTCUSDT", intervalle="8h", debut="2017-08-17", fin="2026-11-01"):
    """Bougies Binance (API publique /api/v3/klines), pagination par 1000 bougies.
    Période [debut, fin[ : la bougie qui commence exactement à `fin` est exclue. Heures en UTC."""
    import requests, time
    url = "https://api.binance.com/api/v3/klines"
    t = int(pd.Timestamp(debut, tz="UTC").timestamp() * 1000)
    t_fin = int(pd.Timestamp(fin, tz="UTC").timestamp() * 1000) - 1   # endTime est inclusif chez Binance
    lignes = []
    while t <= t_fin:
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


def garder_jours_complets(df, par_jour=PAR_JOUR):
    """Garde uniquement les jours qui ont exactement `par_jour` bougies (1 jour = 3 lignes).
    Supprime les jours tronqués (panne de l'exchange, dernier jour incomplet)."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    jour = df.index.normalize()
    taille = pd.Series(jour, index=df.index).map(pd.Series(jour).value_counts())
    return df[taille.values == par_jour]


# ---------------------------------------------------------------- INDICATEURS
def calculer_rendement_log(prix):
    """Rendement log : r_t = ln(P_t / P_{t-1})."""
    return np.log(prix / prix.shift(1))


def calculer_volatilite(rendements, fenetre=20):
    """Écart-type glissant des rendements."""
    return rendements.rolling(window=fenetre).std()


def calculer_RSI(prix, periode=14):
    """RSI entre 0 et 100 (moyenne exponentielle de Wilder)."""
    variation = prix.diff()
    gain = variation.clip(lower=0)
    perte = -variation.clip(upper=0)
    gain_moyen = gain.ewm(alpha=1 / periode, min_periods=periode).mean()
    perte_moyenne = perte.ewm(alpha=1 / periode, min_periods=periode).mean()
    return 100 - (100 / (1 + gain_moyen / perte_moyenne))


def calculer_MACD(prix, court=12, long=26, signal=9):
    """MACD = EMA(court) - EMA(long) ; ligne de signal = EMA(signal) du MACD."""
    macd = prix.ewm(span=court, adjust=False).mean() - prix.ewm(span=long, adjust=False).mean()
    ligne_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, ligne_signal, macd - ligne_signal


def standardiser(X):
    """Moyenne 0, écart-type 1 (renvoie aussi le scaler pour le réutiliser sur le test)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ---------------------------------------------------------------- FEATURES
def construire_features(df, fen_vol=60, fen_rsi=42, macd_params=(36, 78, 27), fen_volume=20):
    """Toutes les features à partir d'un DataFrame OHLCV. Valeurs par défaut pour des bougies
    de 8 h (fenêtres journalières x 3). Chaque feature n'utilise que les bougies PASSÉES."""
    f = pd.DataFrame(index=df.index)
    f["rendement"] = calculer_rendement_log(df["Close"])
    f["volatilite"] = calculer_volatilite(f["rendement"], fen_vol)
    f["RSI"] = calculer_RSI(df["Close"], fen_rsi)
    macd, sig, histo = calculer_MACD(df["Close"], *macd_params)
    f["MACD_norm"] = macd / df["Close"]
    f["histo_norm"] = histo / df["Close"]
    f["amplitude"] = (df["High"] - df["Low"]) / df["Close"]
    # volume relatif : comparé à la même session (00h/08h/16h) sur les derniers jours, passé uniquement
    v_session = df["Volume"].groupby(df.index.hour).transform(
        lambda s: s.rolling(fen_volume).mean().shift(1))
    f["vol_rel"] = np.log(df["Volume"] / v_session).replace([np.inf, -np.inf], np.nan)
    return f
