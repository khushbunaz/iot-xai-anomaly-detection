"""
Explainable IoT Anomaly Detection - Advanced Artefact (Gradio dashboard)
Khushbunaz Dalal | 2529717 | MSc AI, University of Bedfordshire | CIS144-6

HOW TO RUN IN COLAB (new cell AFTER your pipeline cells):
    !pip install gradio plotly shap -q
    import os; os.chdir("/content/drive/MyDrive/<your project folder>")   # folder holding X_test.npy etc.
    %run iot_artefact_app.py          # or paste this whole file into a cell

Reads (from the working folder / DATA_DIR):
    X_train.npy X_test.npy y_train.npy y_test.npy  (scaled arrays, y: 0 benign / 1 attack)
    scaler.pkl feature_names.pkl
    model_one_class_svm.pkl model_isolation_forest.pkl
    model_autoencoder.keras model_lstm_autoencoder.keras   (optional - skipped if missing)
    shap_values.npy                                        (optional)
Every number in the app is computed live from your saved models and test data.
"""
import os, re, io, json, pickle, inspect, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix)
import gradio as gr

warnings.filterwarnings("ignore")
DATA_DIR = os.environ.get("DATA_DIR", ".")
RNG = np.random.default_rng(42)
EVAL_N = 12000          # natural-distribution evaluation subsample (keeps the app fast)
ARRAYS_ARE_SCALED = True

# ----------------------------------------------------------------------------- palette
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURFACE, CRIT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#d03b3b"
MODEL_ORDER = ["Statistical baseline", "Isolation Forest", "One-Class SVM", "Autoencoder", "LSTM autoencoder"]
MODEL_COLOR = dict(zip(MODEL_ORDER, [MUTED, BLUE, ORANGE, AQUA, MAGENTA]))   # colour follows the model, always
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]               # one-hue sequential ramp

# ----------------------------------------------------------------------------- feature decoder
WINDOWS = {"L5": "100 ms", "L3": "500 ms", "L1": "1.5 s", "L0.1": "10 s", "L0.01": "1 min"}
WIN_ORDER = ["L5", "L3", "L1", "L0.1", "L0.01"]
STREAMS = {"MI_dir": ("Source MAC+IP", "traffic from one device (MAC+IP)"),
           "H": ("Source IP", "traffic from one IP address"),
           "HH": ("Host to host", "traffic between two hosts"),
           "HH_jit": ("Host-to-host jitter", "timing jitter between two hosts"),
           "HpHp": ("Socket", "traffic between two sockets (IP+port)")}
STREAM_ORDER = ["MI_dir", "H", "HH", "HH_jit", "HpHp"]
STATS = {"weight": "packet count", "mean": "mean packet size", "std": "std-dev of size",
         "variance": "variance of size", "magnitude": "magnitude", "radius": "radius",
         "covariance": "covariance", "pcc": "correlation"}
FEAT_RE = re.compile(r"^(MI_dir|HpHp|HH_jit|HH|H)_(L[0-9.]+)_([A-Za-z]+)$")


def parse_feature(name):
    m = FEAT_RE.match(str(name))
    return m.groups() if m else (None, None, None)


def short_name(name):
    s, w, st = parse_feature(name)
    if s is None:
        return str(name)
    return f"{STREAMS[s][0]} | {WINDOWS.get(w, w)} | {STATS.get(st, st)}"


def long_name(name):
    s, w, st = parse_feature(name)
    if s is None:
        return str(name)
    return f"{STATS.get(st, st)} of {STREAMS[s][1]}, over the last {WINDOWS.get(w, w)}"


# ----------------------------------------------------------------------------- model wrapper
class Model:
    """score(X) -> anomaly score (higher = more anomalous); alert if score > thr."""
    def __init__(self, name, fn, thr, note=""):
        self.name, self.fn, self.thr, self.note = name, fn, float(thr), note

    def score(self, X):
        return np.asarray(self.fn(np.asarray(X, dtype=np.float32)), dtype=np.float64)

    def predict(self, X):
        return (self.score(X) > self.thr).astype(int)


def _load_pickle(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _keras_scorer(model):
    shp = model.input_shape

    def fn(X):
        Xi = X
        if len(shp) == 3:
            # Windowed LSTM: same construction as training (window of consecutive rows, scored at its
            # last row). The first W-1 rows are padded by repeating row 0; a single row is repeated W times.
            W = shp[1]
            P = np.concatenate([np.repeat(X[:1], W - 1, axis=0), X]) if len(X) else X
            Xi = np.stack([P[i:i + W] for i in range(len(X))]) if len(X) else X.reshape(0, W, shp[2])
        R = model.predict(Xi, batch_size=2048, verbose=0)
        return ((R - Xi) ** 2).reshape(len(X), -1).mean(1)
    return fn


def load_ae_predict(path):
    import tensorflow as tf
    return _keras_scorer(tf.keras.models.load_model(path, compile=False))


# ----------------------------------------------------------------------------- assets
A = {}


def load_assets():
    p = lambda f: os.path.join(DATA_DIR, f)
    Xtr = np.load(p("X_train.npy"), mmap_mode="r")
    Xte = np.load(p("X_test.npy"), mmap_mode="r")
    ytr = np.load(p("y_train.npy"), mmap_mode="r").astype(int)
    yte = np.load(p("y_test.npy"), mmap_mode="r").astype(int)
    feats = list(_load_pickle(p("feature_names.pkl")))
    scaler = _load_pickle(p("scaler.pkl"))

    ben_tr = np.where(ytr == 0)[0]
    ben_ref = np.asarray(Xtr[np.sort(RNG.choice(ben_tr, min(20000, len(ben_tr)), replace=False))], dtype=np.float32)
    ev_idx = np.sort(RNG.choice(len(yte), min(EVAL_N, len(yte)), replace=False))
    Xev, yev = np.asarray(Xte[ev_idx], dtype=np.float32), yte[ev_idx]

    models = {}
    models["Statistical baseline"] = Model("Statistical baseline", lambda X: np.abs(X).max(1), 4.0,
                                           "flags a row if any feature is more than 4 std-devs from benign")
    if os.path.exists(p("model_isolation_forest.pkl")):
        m = _load_pickle(p("model_isolation_forest.pkl"))
        models["Isolation Forest"] = Model("Isolation Forest", lambda X, m=m: -m.decision_function(X), 0.0,
                                           "isolates rows that are easy to separate from benign")
    if os.path.exists(p("model_one_class_svm.pkl")):
        m = _load_pickle(p("model_one_class_svm.pkl"))
        models["One-Class SVM"] = Model("One-Class SVM", lambda X, m=m: -m.decision_function(X), 0.0,
                                        "learns a boundary around benign behaviour")
    for nm, f in [("Autoencoder", "model_autoencoder.keras"), ("LSTM autoencoder", "model_lstm_autoencoder.keras")]:
        if os.path.exists(p(f)):
            try:
                fn = load_ae_predict(p(f))
                thr = np.quantile(fn(ben_ref[:10000]), 0.95)
                models[nm] = Model(nm, fn, thr, "reconstruction error; threshold = 95th percentile of benign error")
            except Exception as e:
                print(f"[warn] could not load {nm}: {e}")

    A.update(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, feats=feats, scaler=scaler, ben_ref=ben_ref,
             Xev=Xev, yev=yev, models=models,
             ben_pool=np.where(yte == 0)[0], atk_pool=np.where(yte == 1)[0])
    S = {n: m.score(Xev) for n, m in models.items()}
    A["S"] = S
    A["ben_sorted"] = {n: np.sort(s[yev == 0]) for n, s in S.items()}
    shp = p("shap_values.npy")
    if os.path.exists(shp):
        sv = np.load(shp, mmap_mode="r")
        sv = np.asarray(sv[:5000], dtype=np.float64)
        A["imp"] = np.abs(sv).mean(0).ravel()[:len(feats)]
        A["imp_src"] = "SHAP (KernelExplainer on One-Class SVM)"
    else:
        m = models.get("One-Class SVM") or list(models.values())[0]
        imp = np.zeros(len(feats))
        atk = Xev[yev == 1][:40]
        for x in atk:
            imp += np.abs(occlusion(m, x))
        A["imp"] = imp / len(atk)
        A["imp_src"] = "occlusion (SHAP file not found)"
    return A


def occlusion(model, x):
    """Score drop when each feature is set to the benign centre (0 in scaled space)."""
    n = len(x)
    B = np.tile(x, (n + 1, 1)).astype(np.float32)
    B[np.arange(n), np.arange(n)] = 0.0
    s = model.score(B)
    return s[n] - s[:n]


# ----------------------------------------------------------------------------- metrics
def metrics_for(name, scores=None, thr=None):
    m = A["models"][name]
    s = A["S"][name] if scores is None else scores
    t = m.thr if thr is None else thr
    y = A["yev"]
    pred = (s > t).astype(int)
    out = dict(Precision=precision_score(y, pred, zero_division=0), Recall=recall_score(y, pred, zero_division=0),
               F1=f1_score(y, pred, zero_division=0))
    try:
        out["ROC-AUC"] = roc_auc_score(y, s)
        out["AUC-PR"] = average_precision_score(y, s)
    except Exception:
        out["ROC-AUC"] = out["AUC-PR"] = float("nan")
    return out


# ----------------------------------------------------------------------------- charts
def base(fig, title, sub=None, h=420):
    t = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{INK2}'>{sub}</span>" if sub else "")
    fig.update_layout(title=dict(text=t, x=0.01, xanchor="left", font=dict(size=16, color=INK)),
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=h,
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=12),
                      margin=dict(l=60, r=30, t=80 if sub else 60, b=55), hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def names():
    return list(A["models"].keys())


def fig_compare():
    mets = {n: metrics_for(n) for n in names()}
    keys = ["Precision", "Recall", "F1", "ROC-AUC"]
    fig = go.Figure()
    for n in names():
        v = [mets[n][k] for k in keys]
        fig.add_bar(name=n, x=keys, y=v, marker=dict(color=MODEL_COLOR[n], cornerradius=4),
                    text=[f"{x:.2f}" for x in v], textposition="outside", textfont=dict(size=10, color=INK2),
                    hovertemplate=f"<b>{n}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.04)
    fig.update_yaxes(range=[0, 1.12], title="score")
    fig.add_hline(y=0.7, line=dict(color=CRIT, dash="dot", width=1))
    return base(fig, "Five detectors, four metrics",
                "F1 can hide a weak detector: read it next to ROC-AUC. Red dotted line = ROC-AUC 0.70 sanity floor")


def table_compare():
    rows = []
    for n in names():
        m = metrics_for(n)
        rows.append([n] + [round(m[k], 4) for k in ["Precision", "Recall", "F1", "ROC-AUC", "AUC-PR"]]
                    + ["REVIEW: near-random ranking" if m["ROC-AUC"] < 0.70 else "ok"])
    return pd.DataFrame(rows, columns=["Model", "Precision", "Recall", "F1", "ROC-AUC", "AUC-PR", "Flag"])


def fig_roc():
    y = A["yev"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, dash="dash", width=1.5),
                    name="chance", hoverinfo="skip")
    for n in names():
        fpr, tpr, _ = roc_curve(y, A["S"][n])
        auc = roc_auc_score(y, A["S"][n])
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{n} (AUC {auc:.3f})", line=dict(color=MODEL_COLOR[n], width=2.2),
                        hovertemplate=f"<b>{n}</b><br>false alarm %{{x:.3f}}<br>detected %{{y:.3f}}<extra></extra>")
    fig.update_xaxes(title="False-alarm rate (benign flagged)", range=[0, 1])
    fig.update_yaxes(title="Attack detection rate", range=[0, 1.02])
    return base(fig, "ROC curves", "Closer to the top-left corner is better", 430)


def fig_pr():
    y = A["yev"]
    fig = go.Figure()
    for n in names():
        p, r, _ = precision_recall_curve(y, A["S"][n])
        fig.add_scatter(x=r, y=p, mode="lines", name=n, line=dict(color=MODEL_COLOR[n], width=2.2),
                        hovertemplate=f"<b>{n}</b><br>recall %{{x:.3f}}<br>precision %{{y:.3f}}<extra></extra>")
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return base(fig, "Precision-recall curves", "Attacks are ~92% of traffic, so precision starts high; watch the tail", 430)


def fig_cm(name):
    m = A["models"][name]
    pred = (A["S"][name] > m.thr).astype(int)
    cm = confusion_matrix(A["yev"], pred, labels=[0, 1])
    pct = cm / cm.sum(1, keepdims=True)
    lab = ["Benign", "Attack"]
    fig = go.Figure(go.Heatmap(z=pct, x=["Predicted benign", "Predicted attack"], y=lab, colorscale=[[0, SEQ[0]], [1, SEQ[3]]],
                               zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
                               text=[[f"{cm[i, j]:,}<br>{pct[i, j]:.1%}" for j in range(2)] for i in range(2)],
                               texttemplate="%{text}", textfont=dict(size=15),
                               hovertemplate="actual %{y}, %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Actual")
    return base(fig, f"Confusion matrix: {name}", "Rows add to 100%. Top-right = false alarms, bottom-left = missed attacks", 380)


def fig_hist(name):
    m = A["models"][name]
    s, y = A["S"][name], A["yev"]
    lo, hi = np.quantile(s, [0.001, 0.99])
    bins = dict(start=float(lo), end=float(hi), size=float((hi - lo) / 60))
    fig = go.Figure()
    fig.add_histogram(x=s[y == 0], name="Benign", xbins=bins, histnorm="probability", marker=dict(color=BLUE), opacity=0.75)
    fig.add_histogram(x=s[y == 1], name="Attack", xbins=bins, histnorm="probability", marker=dict(color=ORANGE), opacity=0.75)
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=m.thr, line=dict(color=INK, dash="dash", width=1.5), annotation_text="alert threshold",
                  annotation_font=dict(size=11, color=INK))
    fig.update_xaxes(title="Anomaly score (higher = more suspicious)", range=[lo, hi])
    fig.update_yaxes(title="Share of rows")
    return base(fig, f"Score separation: {name}", "The less the two shapes overlap, the better the detector", 380)


def fig_importance(topn=15):
    imp, feats = A["imp"], A["feats"]
    idx = np.argsort(imp)[::-1][:topn][::-1]
    labels = [short_name(feats[i]) for i in idx]
    fig = go.Figure(go.Bar(x=imp[idx], y=labels, orientation="h", marker=dict(color=BLUE, cornerradius=4),
                           customdata=[long_name(feats[i]) for i in idx],
                           hovertemplate="<b>%{y}</b><br>%{customdata}<br>importance %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Mean |contribution| to the anomaly score")
    fig.update_layout(margin=dict(l=300))
    return base(fig, f"Top {topn} features driving detection", f"Source: {A['imp_src']}. Names decoded from the N-BaIoT convention", 520)


def fig_heat():
    imp, feats = A["imp"], A["feats"]
    M = np.zeros((len(STREAM_ORDER), len(WIN_ORDER)))
    for f, v in zip(feats, imp):
        s, w, _ = parse_feature(f)
        if s in STREAM_ORDER and w in WIN_ORDER:
            M[STREAM_ORDER.index(s), WIN_ORDER.index(w)] += v
    tot = M.sum() or 1
    Mp = M / tot * 100
    fig = go.Figure(go.Heatmap(
        z=Mp, x=[f"{WINDOWS[w]}" for w in WIN_ORDER], y=[STREAMS[s][0] for s in STREAM_ORDER],
        colorscale=[[0, SEQ[0]], [0.35, SEQ[1]], [0.6, SEQ[2]], [0.8, SEQ[3]], [1, SEQ[4]]], xgap=3, ygap=3,
        text=[[f"{v:.0f}%" for v in r] for r in Mp], texttemplate="%{text}", textfont=dict(size=13),
        colorbar=dict(title="% of total", thickness=12),
        hovertemplate="%{y}<br>window %{x}<br>%{z:.1f}% of importance<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Time window (short to long)", side="bottom")
    return base(fig, "Where the signal lives: traffic stream x time window",
                "Each cell sums the importance of that group's features (rows add up to the whole model)", 400)


def fig_feature(feat):
    j = A["feats"].index(feat)
    x, y = A["Xev"][:, j], A["yev"]
    lo, hi = np.quantile(x, [0.005, 0.995])
    bins = dict(start=float(lo), end=float(hi), size=float(max((hi - lo) / 50, 1e-6)))
    fig = go.Figure()
    fig.add_histogram(x=x[y == 0], name="Benign", xbins=bins, histnorm="probability", marker=dict(color=BLUE), opacity=0.75)
    fig.add_histogram(x=x[y == 1], name="Attack", xbins=bins, histnorm="probability", marker=dict(color=ORANGE), opacity=0.75)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Standardised value (0 = typical benign)", range=[lo, hi])
    fig.update_yaxes(title="Share of rows")
    return base(fig, short_name(feat), long_name(feat), 380)


# ----------------------------------------------------------------------------- callbacks
def card(label, value, sub="", tone="ink"):
    col = {"ink": INK, "bad": CRIT, "good": "#006300"}[tone]
    return (f"<div class='kpi'><div class='kl'>{label}</div><div class='kv' style='color:{col}'>{value}</div>"
            f"<div class='ks'>{sub}</div></div>")


def kpis():
    best = max(names(), key=lambda n: metrics_for(n)["F1"])
    mb = metrics_for(best)
    y = A["yte"]
    ben = (y == 0).mean()
    return ("<div class='kpis'>" +
            card("Test rows", f"{len(y):,}", "N-BaIoT, 9 IoT devices") +
            card("Benign share", f"{ben:.1%}", f"attack {1 - ben:.1%} (heavy imbalance)") +
            card("Features", f"{len(A['feats'])}", "5 streams x 5 windows x stats") +
            card("Best F1", f"{mb['F1']:.3f}", best, "good") +
            card(f"{best} recall", f"{mb['Recall']:.1%}", f"precision {mb['Precision']:.1%}") + "</div>")


def pct_rank(name, s):
    b = A["ben_sorted"][name]
    return 100 * np.searchsorted(b, s) / max(len(b), 1)


def analyze(kind, idx, focus):
    if kind == "Random benign sample":
        i = int(RNG.choice(A["ben_pool"]))
    elif kind == "Random attack sample":
        i = int(RNG.choice(A["atk_pool"]))
    else:
        i = int(np.clip(idx, 0, len(A["yte"]) - 1))
    x = np.asarray(A["Xte"][i], dtype=np.float32)
    truth = "ATTACK" if A["yte"][i] == 1 else "BENIGN"
    rows, flagged, voters = [], 0, 0
    names_ = names()
    fig = go.Figure()
    for n in names_:
        m = A["models"][n]
        s = float(m.score(x[None])[0])
        a = s > m.thr
        trusted = metrics_for(n)["ROC-AUC"] >= 0.70
        voters += int(trusted); flagged += int(a and trusted)
        r, t = pct_rank(n, s), pct_rank(n, m.thr)
        rows.append([n, round(s, 4), round(m.thr, 4), "ALERT" if a else "normal", f"{r:.0f}%"])
        fig.add_bar(y=[n], x=[r], orientation="h", name=n, showlegend=False, marker=dict(color=MODEL_COLOR[n], cornerradius=4),
                    text=["ALERT" if a else "normal"], textposition="inside", textfont=dict(color="white", size=11),
                    hovertemplate=f"<b>{n}</b><br>more unusual than %{{x:.0f}}% of benign traffic<extra></extra>")
        fig.add_shape(type="line", x0=t, x1=t, y0=names_.index(n) - 0.4, y1=names_.index(n) + 0.4,
                      line=dict(color=INK, width=2))
    fig.update_xaxes(title="How unusual (percentile among benign traffic; black tick = alert threshold)", range=[0, 100])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=150))
    base(fig, "What each detector thinks", "A bar past its black tick means that model raises an alert", 360)

    ok = (flagged > voters / 2) == (truth == "ATTACK")
    verdict = "ATTACK" if flagged > voters / 2 else "BENIGN"
    tone = "#006300" if ok else CRIT
    html = (f"<div class='verdict'><div><div class='kl'>Ensemble verdict (majority of trusted models)</div>"
            f"<div class='kv' style='color:{tone}'>{verdict}</div></div>"
            f"<div><div class='kl'>Ground truth</div><div class='kv'>{truth}</div></div>"
            f"<div><div class='kl'>Detectors alerting</div><div class='kv'>{flagged} / {voters}</div></div>"
            f"<div><div class='kl'>Result</div><div class='kv' style='color:{tone}'>{'correct' if ok else 'wrong'}</div></div>"
            f"<div><div class='kl'>Test row</div><div class='kv'>#{i}</div></div></div>")

    m = A["models"][focus if focus in A["models"] else names_[0]]
    c = occlusion(m, x)
    top = np.argsort(np.abs(c))[::-1][:10][::-1]
    fig2 = go.Figure(go.Bar(x=c[top], y=[short_name(A["feats"][k]) for k in top], orientation="h",
                            marker=dict(color=[ORANGE if v > 0 else BLUE for v in c[top]], cornerradius=4),
                            customdata=[[long_name(A["feats"][k]), float(x[k])] for k in top],
                            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>value %{customdata[1]:.2f} std-devs<br>"
                                          "effect on score %{x:.4f}<extra></extra>"))
    fig2.update_xaxes(title="Change in anomaly score if this feature were normal (orange = pushes toward ATTACK)")
    fig2.update_layout(margin=dict(l=300))
    base(fig2, f"Why? Top features for this row ({m.name})",
         "Occlusion explanation: set one feature to its benign centre and see how the score moves", 430)
    df = pd.DataFrame(rows, columns=["Model", "Score", "Threshold", "Decision", "Unusualness"])
    return html, fig, fig2, df


def run_threshold(name, q):
    s, y = A["S"][name], A["yev"]
    ben = s[y == 0]
    qs = np.linspace(50, 99.9, 60)
    P, R, F = [], [], []
    for qq in qs:
        mm = metrics_for(name, s, np.quantile(ben, qq / 100))
        P.append(mm["Precision"]); R.append(mm["Recall"]); F.append(mm["F1"])
    cur = metrics_for(name, s, np.quantile(ben, q / 100))
    fig = go.Figure()
    for lab, v, col in [("Recall", R, ORANGE), ("Precision", P, BLUE), ("F1", F, AQUA)]:
        fig.add_scatter(x=100 - qs, y=v, name=lab, mode="lines", line=dict(color=col, width=2.2),
                        hovertemplate=lab + " %{y:.3f} at %{x:.1f}% false alarms<extra></extra>")
    fig.add_vline(x=100 - q, line=dict(color=INK, dash="dash", width=1.5), annotation_text="your setting",
                  annotation_font=dict(size=11, color=INK))
    fig.update_xaxes(title="Allowed false-alarm rate on benign traffic (%)", autorange="reversed")
    fig.update_yaxes(title="score", range=[0, 1.02])
    base(fig, f"Sensitivity trade-off: {name}", "Raise sensitivity (more false alarms allowed) to catch more attacks", 400)
    html = ("<div class='kpis'>" + card("False-alarm rate", f"{100 - q:.1f}%", "benign rows flagged") +
            card("Recall", f"{cur['Recall']:.1%}", "attacks caught", "good") +
            card("Precision", f"{cur['Precision']:.1%}", "alerts that are real") +
            card("F1", f"{cur['F1']:.3f}", "balance of the two") + "</div>")
    return html, fig


def run_stress(steps):
    y = A["yev"]
    Xa = A["Xev"][y == 1][:1500]
    Xb = A["Xev"][y == 0][:1500]
    al = np.linspace(0, 0.9, int(steps))
    f1 = make_subplots(rows=1, cols=1)
    for n in names():
        m = A["models"][n]
        det = [float((m.score(Xa * (1 - a)) > m.thr).mean()) for a in al]
        f1.add_scatter(x=al * 100, y=det, name=n, mode="lines+markers", line=dict(color=MODEL_COLOR[n], width=2.2),
                       marker=dict(size=7, line=dict(color=SURFACE, width=2)),
                       hovertemplate=f"<b>{n}</b><br>%{{x:.0f}}% camouflage<br>detected %{{y:.1%}}<extra></extra>")
    f1.update_xaxes(title="Camouflage: attack features shrunk toward benign (%)")
    f1.update_yaxes(title="Attacks still detected", range=[0, 1.02], tickformat=".0%")
    base(f1, "Evasion test: can an attacker blend in?", "Each attack row is scaled toward the benign centre; steeper drop = easier to evade", 430)
    sig = np.linspace(0, 2, int(steps))
    f2 = go.Figure()
    for n in names():
        m = A["models"][n]
        fa = [float((m.score(Xb + RNG.normal(0, s, Xb.shape).astype(np.float32)) > m.thr).mean()) for s in sig]
        f2.add_scatter(x=sig, y=fa, name=n, mode="lines+markers", line=dict(color=MODEL_COLOR[n], width=2.2),
                       marker=dict(size=7, line=dict(color=SURFACE, width=2)),
                       hovertemplate=f"<b>{n}</b><br>noise %{{x:.2f}} std<br>false alarms %{{y:.1%}}<extra></extra>")
    f2.update_xaxes(title="Gaussian sensor noise added to benign traffic (std-devs)")
    f2.update_yaxes(title="Benign wrongly flagged", range=[0, 1.02], tickformat=".0%")
    base(f2, "Noise test: do false alarms explode?", "Lower is better; a flat line means the detector tolerates noisy sensors", 430)
    return f1, f2


def run_upload(file, name):
    if file is None:
        return "<div class='note'>Upload a CSV first.</div>", None, None
    path = file if isinstance(file, str) else getattr(file, "name", file)
    try:
        df = pd.read_csv(path)
        feats = A["feats"]
        if all(f in df.columns for f in feats):
            X = df[feats].to_numpy(np.float32)
        elif df.shape[1] >= len(feats):
            X = df.iloc[:, :len(feats)].to_numpy(np.float32)
        else:
            return f"<div class='note'>Need {len(feats)} feature columns, found {df.shape[1]}.</div>", None, None
        if not ARRAYS_ARE_SCALED or True:
            X = A["scaler"].transform(X).astype(np.float32)
    except Exception as e:
        return f"<div class='note'>Could not read the file: {e}</div>", None, None
    m = A["models"][name]
    s = m.score(X)
    a = s > m.thr
    out = pd.DataFrame({"row": np.arange(len(s)), "anomaly_score": np.round(s, 4), "decision": np.where(a, "ALERT", "normal")})
    fig = go.Figure()
    fig.add_histogram(x=s[~a], name="normal", marker=dict(color=BLUE), opacity=0.8)
    fig.add_histogram(x=s[a], name="alert", marker=dict(color=ORANGE), opacity=0.8)
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=m.thr, line=dict(color=INK, dash="dash"), annotation_text="threshold")
    fig.update_xaxes(title="Anomaly score")
    fig.update_yaxes(title="Rows")
    base(fig, f"Scores for your file ({name})", None, 360)
    html = ("<div class='kpis'>" + card("Rows scored", f"{len(s):,}") +
            card("Alerts", f"{int(a.sum()):,}", f"{a.mean():.1%} of rows", "bad" if a.mean() > 0.5 else "ink") + "</div>")
    return html, fig, out.head(500)


def model_cards():
    h = "<div class='mcards'>"
    for n in names():
        m = A["models"][n]
        k = metrics_for(n)
        flag = k["ROC-AUC"] < 0.70
        h += (f"<div class='mc' style='border-top:4px solid {MODEL_COLOR[n]}'><b>{n}</b>"
              f"<div class='ks'>{m.note}</div><div class='ks'>F1 {k['F1']:.3f} | ROC-AUC {k['ROC-AUC']:.3f}</div>"
              + ("<div class='warn'>Review: ROC-AUC below 0.70</div>" if flag else "") + "</div>")
    return h + "</div>"


CSS = f"""
.gradio-container {{max-width: 1180px !important; margin: auto}}
.kpis,.mcards,.verdict {{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:8px 0}}
.kpi,.mc,.verdict > div {{background:{SURFACE}; border:1px solid {GRID}; border-radius:10px; padding:12px 14px}}
.kl {{font-size:12px; color:{INK2}}} .kv {{font-size:26px; font-weight:650; color:{INK}; line-height:1.25}}
.ks {{font-size:12px; color:{MUTED}; margin-top:2px}} .warn {{font-size:12px; color:{CRIT}; margin-top:4px; font-weight:600}}
.note {{padding:12px; border:1px solid {GRID}; border-radius:8px; background:{SURFACE}; color:{INK2}}}
"""


def build(blocks_kw=None):
    names_ = names()
    best = "One-Class SVM" if "One-Class SVM" in names_ else names_[0]
    feats = A["feats"]
    top_feats = [feats[i] for i in np.argsort(A["imp"])[::-1][:30]]
    with gr.Blocks(title="Explainable IoT Anomaly Detection", **(blocks_kw or {})) as demo:
        gr.Markdown("# Explainable AI-Driven Anomaly Detection for IoT Networks\n"
                    "N-BaIoT botnet traffic (Mirai and Gafgyt) | five detectors trained on benign traffic only | "
                    "Khushbunaz Dalal, 2529717, University of Bedfordshire")
        with gr.Tabs():
            with gr.Tab("1. Overview"):
                gr.HTML(kpis())
                gr.HTML(model_cards())
                gr.Plot(fig_compare())
                with gr.Accordion("Table view (exact numbers)", open=False):
                    gr.Dataframe(table_compare(), interactive=False)
            with gr.Tab("2. Model comparison"):
                with gr.Row():
                    gr.Plot(fig_roc()); gr.Plot(fig_pr())
                sel = gr.Dropdown(names_, value=best, label="Model to inspect")
                with gr.Row():
                    cm = gr.Plot(fig_cm(best)); hs = gr.Plot(fig_hist(best))
                sel.change(lambda n: (fig_cm(n), fig_hist(n)), sel, [cm, hs])
            with gr.Tab("3. Live detector"):
                with gr.Row():
                    kind = gr.Radio(["Random benign sample", "Random attack sample", "Specific test row"],
                                    value="Random attack sample", label="Traffic sample")
                    idx = gr.Number(value=0, precision=0, label="Test row # (for 'Specific')")
                    focus = gr.Dropdown(names_, value=best, label="Explain with")
                btn = gr.Button("Analyse traffic", variant="primary")
                v = gr.HTML(); p1 = gr.Plot(); p2 = gr.Plot()
                with gr.Accordion("Table view", open=False):
                    t = gr.Dataframe(interactive=False)
                btn.click(analyze, [kind, idx, focus], [v, p1, p2, t])
            with gr.Tab("4. Explainability"):
                gr.Markdown("**Reading the feature names.** `HH_jit_L0.1_mean` = the *mean* of the *timing jitter between a pair of hosts*, "
                            "measured over the last *10 seconds*. Windows: L5 = 100 ms, L3 = 500 ms, L1 = 1.5 s, L0.1 = 10 s, L0.01 = 1 min.")
                gr.Plot(fig_importance(15))
                gr.Plot(fig_heat())
                fsel = gr.Dropdown([(short_name(f), f) for f in top_feats], value=top_feats[0], label="Feature explorer (top 30 by importance)")
                fp = gr.Plot(fig_feature(top_feats[0]))
                fsel.change(fig_feature, fsel, fp)
            with gr.Tab("5. Threshold tuner"):
                gr.Markdown("Operators choose how many false alarms they can tolerate. Move the slider and watch recall and precision.")
                tm = gr.Dropdown(names_, value=best, label="Model")
                q = gr.Slider(80, 99.9, value=95, step=0.1, label="Benign-score percentile used as the threshold (95 = about 5% false alarms)")
                kp = gr.HTML(); tp = gr.Plot()
                for c in (tm, q):
                    c.change(run_threshold, [tm, q], [kp, tp])
                demo.load(run_threshold, [tm, q], [kp, tp])
            with gr.Tab("6. Robustness"):
                gr.Markdown("Two stress tests on the saved models. **Evasion:** an attacker tries to look benign. **Noise:** sensors get noisy.")
                steps = gr.Slider(4, 12, value=8, step=1, label="Number of test levels")
                rb = gr.Button("Run stress test", variant="primary")
                s1 = gr.Plot(); s2 = gr.Plot()
                rb.click(run_stress, steps, [s1, s2])
            with gr.Tab("7. Score your own CSV"):
                gr.Markdown(f"Upload a CSV with the {len(feats)} N-BaIoT feature columns (raw, unscaled). The saved scaler is applied.")
                um = gr.Dropdown(names_, value=best, label="Model")
                uf = gr.File(label="CSV file", file_types=[".csv"])
                ub = gr.Button("Score file", variant="primary")
                uh = gr.HTML(); up = gr.Plot(); ut = gr.Dataframe(interactive=False)
                ub.click(run_upload, [uf, um], [uh, up, ut])
            with gr.Tab("About"):
                gr.Markdown(
                    "**Method.** All learned models are trained on benign traffic only (anomaly detection), scaler fitted on benign "
                    "training rows. Metrics are computed live on a random subsample of the held-out test split with the natural class mix.\n\n"
                    "**Explainability.** Global importance comes from SHAP on the One-Class SVM. Per-row explanations use occlusion "
                    "(set a feature to its benign centre and re-score), which works for every model.\n\n"
                    "**Limitations.** Binary framing (benign vs anomalous), single dataset, lab-generated botnet traffic, "
                    "no live network capture. The LSTM autoencoder is flagged because its ranking quality is near random.")
    return demo


def main():
    load_assets()
    kw = dict(theme=gr.themes.Soft(), css=CSS)
    new_api = "css" in inspect.signature(gr.Blocks.launch).parameters
    demo = build(None if new_api else kw)
    demo.launch(share=True, debug=False, **(kw if new_api else {}))


if __name__ == "__main__":
    main()
else:
    try:
        get_ipython  # noqa: F821  (running inside Colab/Jupyter via %run or paste)
        main()
    except NameError:
        pass
