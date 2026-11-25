"""HMM jouet : 3 états cachés, 2 observations. Forward, Backward, Viterbi, Baum-Welch (itérations).
Mêmes paramètres que notebooks/03_algorithmes_hmm.ipynb. Résultats écrits dans results/hmm_exemple_3etats.json."""
import numpy as np, json
from pathlib import Path
np.set_printoptions(precision=4, suppress=True)
ETATS = ["Haussier", "Baissier", "Panique"]; OBS = ["Hausse", "Baisse"]
pi = np.array([0.5, 0.3, 0.2])
A = np.array([[0.80, 0.15, 0.05], [0.10, 0.80, 0.10], [0.20, 0.30, 0.50]])
B = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
O = np.array([0, 0, 1, 1, 1, 0]); T = len(O); N = 3

def forward(pi, A, B, O):
    al = np.zeros((len(O), N)); al[0] = pi * B[:, O[0]]
    for t in range(1, len(O)): al[t] = (al[t-1] @ A) * B[:, O[t]]
    return al
def backward(A, B, O):
    be = np.ones((len(O), N))
    for t in range(len(O)-2, -1, -1): be[t] = A @ (B[:, O[t+1]] * be[t+1])
    return be
def viterbi(pi, A, B, O):
    d = np.zeros((len(O), N)); psi = np.zeros((len(O), N), int); d[0] = pi * B[:, O[0]]
    for t in range(1, len(O)):
        m = d[t-1][:, None] * A; psi[t] = m.argmax(0); d[t] = m.max(0) * B[:, O[t]]
    path = [int(d[-1].argmax())]
    for t in range(len(O)-1, 0, -1): path.insert(0, int(psi[t][path[0]]))
    return d, psi, path
def baum_welch_step(pi, A, B, O):
    al, be = forward(pi, A, B, O), backward(A, B, O); p = al[-1].sum()
    g = al * be / p
    xi = np.array([al[t][:, None] * A * (B[:, O[t+1]] * be[t+1])[None, :] / p for t in range(len(O)-1)])
    pi2 = g[0]; A2 = xi.sum(0) / g[:-1].sum(0)[:, None]
    B2 = np.array([[g[O == k, i].sum() for k in range(2)] for i in range(N)]) / g.sum(0)[:, None]
    return pi2, A2, B2, np.log(p), g, xi

al = forward(pi, A, B, O); be = backward(A, B, O); P = al[-1].sum()
g = al * be / P; d, psi, path = viterbi(pi, A, B, O)
print("alpha\n", al, "\nP(O|lambda) =", P, "ln =", np.log(P))
print("beta\n", be, "\ngamma\n", g, "\ndelta\n", d, "\npsi\n", psi, "\nchemin:", [ETATS[i] for i in path])
res = dict(al=al.tolist(), be=be.tolist(), g=g.tolist(), d=d.tolist(), psi=psi.tolist(), path=path, P=P, lnP=float(np.log(P)))
p_, A_, B_ = pi, A, B; ll = []
for it in range(1, 11):
    p_, A_, B_, l, gg, xi = baum_welch_step(p_, A_, B_, O); ll.append(float(l))
    if it == 1: res["it1"] = dict(pi=p_.tolist(), A=A_.tolist(), B=B_.tolist(), xi1=xi[0].tolist())
    print(f"iter {it:2d}  ln P(O|lambda) = {l:.5f}")
res["ll"] = ll; res["final"] = dict(pi=p_.tolist(), A=A_.tolist(), B=B_.tolist())
# prédiction à court terme : P(S_{T+k}) = gamma_T . A^k
res["pred"] = [(g[-1] @ np.linalg.matrix_power(A, k)).tolist() for k in (1, 2, 5)]
print("pred", res["pred"])
sortie = Path(__file__).resolve().parent.parent / "results" / "hmm_exemple_3etats.json"
sortie.parent.mkdir(exist_ok=True)
with open(sortie, "w") as fh:
    json.dump(res, fh)

# --- Vérification avec hmmlearn ---
from hmmlearn import hmm
m = hmm.CategoricalHMM(n_components=3, init_params="", params="")
m.startprob_, m.transmat_, m.emissionprob_ = pi, A, B
X = O.reshape(-1, 1)
print("hmmlearn score (ln P) :", m.score(X), "| viterbi :", m.decode(X)[1], "| gamma[-1] :", m.predict_proba(X)[-1])
