# Régimes de marché du Bitcoin avec des HMM

Détection des régimes de marché du **Bitcoin (BTCUSDT)** avec des modèles de Markov cachés (HMM) à 3 états (Haussier, Baissier, Panique), puis prédiction du régime suivant **sans look-ahead bias** (filtre forward).

## Données
- Source : API publique Binance, paire BTCUSDT, du 17/08/2017 au 31/10/2026.
- Fréquence : bougies de **8 heures** (00h, 08h, 16h UTC). **1 jour = 3 lignes** : seuls les jours complets sont gardés.
- Features : rendement log, volatilité glissante (60 bougies = 20 jours), RSI, MACD normalisé, amplitude, volume relatif par session.
- Les features brutes **ne sont pas gaussiennes** (rendement : kurtosis en excès ≈ 12, queues épaisses ; volatilité asymétrique). Elles sont donc **transformées vers une loi normale N(0, 1)** par quantiles (`QuantileTransformer`), comme le suppose le `GaussianHMM` ; tests de Jarque-Bera / KS, histogrammes et QQ-plots avant / après dans `results/normalite_avant.png` et `results/normalite_apres.png`.
- Découpage chronologique 70 % train / 30 % test sur des jours entiers ; transformation gaussienne ajustée sur le train seulement (pas de look-ahead).
- Dans chaque notebook, chaque cellule de code est suivie d'une **interprétation** du résultat.

## Organisation
| Dossier / fichier | Contenu |
|---|---|
| `notebooks/01_donnees.ipynb` | téléchargement, contrôle qualité, jours complets, features, tests de normalité, découpage, transformation gaussienne, sauvegarde |
| `notebooks/02_selection_features.ipynb` | feature selection : variables qui influencent chaque cible (rendement, volatilité, prix, MACD), un modèle par cible |
| `notebooks/03_algorithmes_hmm.ipynb` | Forward, Backward, Viterbi, Baum-Welch codés à la main sur un exemple, vérifiés avec `hmmlearn` |
| `notebooks/04_hmm_rendements.ipynb` | GaussianHMM sur BTC (choix de N par AIC/BIC), interprétation, prédiction du prochain régime |
| `notebooks/05_hmm_lstm.ipynb` | initialisation du HMM par un **LSTM** (au lieu du hasard) : le LSTM prévoit la volatilité, ses états cachés donnent 3 régimes, d'où π, A, μ, Σ ; puis Baum-Welch, comparaison avec 10 départs aléatoires, prédiction sans look-ahead |
| `src/fonctions.py` | fonctions communes (données, indicateurs, features, variables candidates et cibles) |
| `src/hmm_exemple_3etats.py`, `src/hmm_rendements.py` | versions script des notebooks 03 et 04 |
| `data/` | `btcusdt_8h.csv` (brut) et fichiers produits par le notebook 01 |
| `results/` | figures et résultats |

## Installation et exécution
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter lab
```
Exécuter les notebooks dans l'ordre : `01` → `02` → `03` → `04` → `05`.
