# app.py
# Well LSTM Autoencoder — Streamlit UI (Parity-Safe + K control + Visual Cards)
# -----------------------------------------------------------------------------

import io, os, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Well LSTM Autoencoder (Parity-Safe)", layout="wide")

# -----------------------
# Visual message card helper (HTML) — MOVED UP so it's defined before use
# -----------------------
def _message_card(status:str, ts:str, top_feature:str, rmse:float, per_thr:float, global_thr:float, k_thr:float, action:str) -> str:
    is_anom = (status.upper() == "ANOMALY")
    color = "#ffefef" if is_anom else "#e9f9ee"
    border = "#e53935" if is_anom else "#2e7d32"
    title = "ANOMALY DETECTED" if is_anom else "NORMAL"
    return f"""
    <div style="
        border-left: 6px solid {border};
        background: {color};
        padding: 14px 16px;
        border-radius: 8px;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        ">
      <div style="font-weight:700; font-size:16px; margin-bottom:6px;">{title}</div>
      <div style="font-size:14px; line-height:1.5;">
        <b>Time:</b> {ts}<br/>
        <b>Top feature:</b> {top_feature}<br/>
        <b>Feature RMSE:</b> {rmse:.4f}<br/>
        <b>Per-feature threshold (μ+3σ):</b> {per_thr:.4f}<br/>
        <b>K × Global threshold:</b> {k_thr:.4f}<br/>
        <b>Global threshold:</b> {global_thr:.4f}<br/>
        <b>Recommended action:</b> {action}
      </div>
    </div>
    """

# -----------------------
# Deterministic seed
# -----------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# -----------------------
# Standardization hook
# -----------------------
_ORIG_READ_CSV = pd.read_csv

def _norm_colname(c: str) -> str:
    c = re.sub(r"\(.*?\)|\[.*?\]", " ", str(c))
    c = re.sub(r"[^a-zA-Z0-9]+", " ", c).strip().lower()
    return c

_PATTERNS = {
    "Time":     [r"\btime\b", r"\bdate\s*time\b", r"\bdatetime\b", r"\btimestamp\b", r"\bts\b"],
    "Current":  [r"\bcurrent\b", r"\bmotor\s*current\b", r"\bamps?\b", r"\bampere?s?\b", r"^i$"],
    "Flow":     [r"\bflow\b", r"\bflow\s*rate\b", r"\bdischarge(_| )?flow\b", r"\bq\b"],
    "Pressure": [r"\bpressure\b", r"\bpress\b", r"\bwellhead\s*pressure\b", r"\bdischarge\s*pressure\b", r"^p$"],
}
_UNIT_HINTS = {
    "Current":  [r"\b(a|amp|amps|amperes?)\b"],
    "Flow":     [r"\bm3\s*/?\s*h\b", r"\bl\s*/?\s*s\b", r"\blpm\b"],
    "Pressure": [r"\bbar\b", r"\bpsi\b", r"\bkpa\b", r"\bmpa\b"],
}

def _parse_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^\d,\.\-eE]", "", regex=True)
    comma_decimal = s.str.contains(",") & ~s.str.contains(r"\.")
    s = np.where(comma_decimal, s.str.replace(",", ".", regex=False), s.str.replace(",", "", regex=False))
    return pd.to_numeric(pd.Series(s, index=series.index), errors="coerce")

def _score_match(col_norm: str, raw_col: str, target: str) -> int:
    score = 0
    for pat in _PATTERNS[target]:
        if re.search(pat, col_norm):
            score += 10
    raw_low = str(raw_col).lower()
    for upat in _UNIT_HINTS.get(target, []):
        if re.search(upat, raw_low):
            score += 3
    return score

def standardize_well_df(df: pd.DataFrame, add_only: bool = True) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    cols = list(df.columns)
    norm_map = {c: _norm_colname(c) for c in cols}

    chosen = {}
    if "Column1" in df.columns: chosen.setdefault("Time", "Column1")
    if "Column2" in df.columns: chosen.setdefault("Flow", "Column2")
    if "Column4" in df.columns: chosen.setdefault("Current", "Column4")
    if "Column6" in df.columns: chosen.setdefault("Pressure", "Column6")

    for target in ["Time", "Current", "Flow", "Pressure"]:
        if target not in chosen:
            ranked = sorted(cols, key=lambda c: _score_match(norm_map[c], c, target), reverse=True)
            if ranked and _score_match(norm_map[ranked[0]], ranked[0], target) > 0:
                chosen[target] = ranked[0]

    out = df.copy()
    if "Time" not in out.columns and "Time" in chosen:
        out["Time"] = pd.to_datetime(out[chosen["Time"]], errors="coerce")
    for t in ["Current", "Flow", "Pressure"]:
        if t not in out.columns and t in chosen:
            series = out[chosen[t]]
            out[t] = series if pd.api.types.is_numeric_dtype(series) else _parse_numeric(series)

    if not add_only:
        rename_map = {chosen.get("Time", None): "Time",
                      chosen.get("Current", None): "Current",
                      chosen.get("Flow", None): "Flow",
                      chosen.get("Pressure", None): "Pressure"}
        rename_map = {k: v for k, v in rename_map.items() if k is not None}
        out = out.rename(columns=rename_map)

    return out

def _read_csv_std(*args, **kwargs):
    df = _ORIG_READ_CSV(*args, **kwargs)
    try:
        df = standardize_well_df(df, add_only=True)
    except Exception:
        pass
    return df

pd.read_csv = _read_csv_std

TIME_COL_STD = "Time"
FEATURE_COLS_STD = ["Current", "Flow", "Pressure"]

# -----------------------
# Core pipeline
# -----------------------
def resample_and_fill(df, ts_col, step_s, feature_cols):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.set_index(ts_col).sort_index()
    df = df.resample(f"{step_s}s").mean().interpolate(limit=6, limit_direction='both').dropna()
    return df

def zfit(df):
    return {'mu': df.mean().to_dict(),
            'sigma': df.std(ddof=0).replace(0, 1e-6).to_dict()}

def zapply(df, sc, clip=5.0):
    z = (df - pd.Series(sc['mu'])) / pd.Series(sc['sigma'])
    return z.clip(-clip, clip) if clip else z

def make_windows(X, win, stride):
    W=[]
    for s in range(0, len(X)-win+1, stride):
        W.append(X[s:s+win])
    return np.stack(W) if W else np.empty((0,win,X.shape[1]))

class LSTMAE(nn.Module):
    def __init__(self, n, latent=16, h=(64,32)):
        super().__init__()
        self.e1 = nn.LSTM(n, h[0], batch_first=True)
        self.e2 = nn.LSTM(h[0], h[1], batch_first=True)
        self.h2z = nn.Linear(h[1], latent)
        self.z2h = nn.Linear(latent, h[1])
        self.d1 = nn.LSTM(h[1], h[0], batch_first=True)
        self.d2 = nn.LSTM(h[0], n, batch_first=True)
    def forward(self, x):
        e,_ = self.e1(x); e,_ = self.e2(e); z = self.h2z(e[:,-1,:])
        h0 = self.z2h(z).unsqueeze(1).repeat(1, x.size(1), 1)
        d,_ = self.d1(h0); y,_ = self.d2(d); return y

# -----------------------
# App state
# -----------------------
if "STATE" not in st.session_state:
    st.session_state.STATE = {
        "train_df": None,
        "live_df": None,
        "train_ready": False,
        "scaler": None,
        "model_state": None,
        "threshold": None,
        "per_feat_thr": None,
        "win": None,
        "stride": None,
        "loss_hist": [],
        "human_messages": [],
        "alias": {'Column2':'flow','Column4':'current','Column6':'pressure'},
        "train_proc_df": None,   # parity
        "train_proc_X":  None,   # parity
    }
S = st.session_state.STATE

st.title("💧 Well LSTM Autoencoder — Parity-Safe Anomaly Detection")

tabs = st.tabs(["1) Data", "2) Train", "3) Detect", "4) Export"])

# ======================
# Tab 1: Data
# ======================
with tabs[0]:
    st.subheader("Upload Data")
    cL, cR = st.columns(2)

    with cL:
        train_file = st.file_uploader("Training CSV (historical)", type=["csv"], key="train_csv")
        if train_file is not None:
            try:
                S["train_df"] = pd.read_csv(train_file)
                S["train_df"] = standardize_well_df(S["train_df"], add_only=True)
                st.success("Training CSV loaded.")
            except Exception as e:
                st.error(f"Failed to read training CSV: {e}")

    with cR:
        live_file = st.file_uploader("Live CSV (to detect anomalies)", type=["csv"], key="live_csv")
        LIVE_SKIPROWS = st.number_input("LIVE CSV skiprows", min_value=0, value=0, step=1, key="live_skiprows")
        if live_file is not None:
            try:
                content = live_file.getvalue()
                buf = io.BytesIO(content)
                tmp_df = pd.read_csv(buf, skiprows=LIVE_SKIPROWS)
                S["live_df"] = standardize_well_df(tmp_df, add_only=True)
                st.success("Live CSV loaded.")
            except Exception as e:
                st.error(f"Failed to read live CSV: {e}")

    st.markdown("**Auto standardization active** → app tries to add: `Time`, `Current`, `Flow`, `Pressure`.")

    st.divider()
    st.subheader("Preview")
    p1, p2 = st.columns(2)
    if S["train_df"] is not None:
        with p1:
            st.caption("Training head()")
            st.dataframe(S["train_df"].head(10), use_container_width=True)
    if S["live_df"] is not None:
        with p2:
            st.caption("Live head()")
            st.dataframe(S["live_df"].head(10), use_container_width=True)

    st.divider()
    st.subheader("Configure Columns & Sampling")
    TIMESTAMP_COL = st.selectbox("Timestamp column", options=([TIME_COL_STD] + (list(S["train_df"].columns) if S["train_df"] is not None else [])), index=0, key="ts_col")
    available_cols = list(S["train_df"].columns) if S["train_df"] is not None else []
    suggested = [c for c in FEATURE_COLS_STD if S["train_df"] is not None and c in available_cols]
    default_feats = suggested if suggested else [c for c in ["Column2","Column4","Column6"] if c in available_cols] or ["Column2","Column4","Column6"]
    FEATURE_COLS = st.multiselect("Feature columns (e.g., Current, Flow, Pressure)", options=available_cols, default=default_feats, key="feat_cols")

    c1, c2, c3 = st.columns(3)
    with c1:
        SAMPLE_SECONDS = st.number_input("Sampling step (seconds)", min_value=1, value=3600, step=60, key="samp_secs")
    with c2:
        WINDOW_MIN    = st.number_input("Window length (minutes)", min_value=1, value=360, step=1, key="win_min")
    with c3:
        STRIDE_MIN    = st.number_input("Stride (minutes)", min_value=1, value=60, step=1, key="stride_min")

    # User control for classic top-feature factor K
    TOPFEAT_K = st.slider("Top-feature factor K (× global thr)", min_value=0.05, max_value=2.0, value=0.35, step=0.05, key="topfeat_k")

    ready = (S["train_df"] is not None) and (S["live_df"] is not None) and (len(FEATURE_COLS) > 0) and (TIMESTAMP_COL in (S["train_df"].columns if S["train_df"] is not None else []))
    S["train_ready"] = bool(ready)
    if ready:
        st.success("Ready to train (go to Tab 2).")
    else:
        st.info("Upload both CSVs and set columns to proceed.")

# ======================
# Tab 2: Train
# ======================
with tabs[1]:
    st.subheader("Train LSTM Autoencoder")
    if not S["train_ready"]:
        st.warning("Please complete Tab 1 (Data) first.")
    else:
        with st.expander("Hyperparameters", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                EPOCHS = st.number_input("Epochs", min_value=1, value=20, step=1, key="epochs")
                LATENT = st.number_input("Latent size", min_value=2, value=16, step=1, key="latent")
            with c2:
                H1 = st.number_input("Hidden-1", min_value=8, value=64, step=8, key="h1")
                H2 = st.number_input("Hidden-2", min_value=8, value=32, step=8, key="h2")
            with c3:
                LR = st.number_input("Learning rate", min_value=1e-5, value=1e-3, step=1e-5, format="%.5f", key="lr")
                ZCLIP = st.number_input("Z-score clip (σ)", min_value=0.0, value=5.0, step=0.5, key="zclip")

        start = st.button("▶ Train", type="primary", use_container_width=False, key="btn_train")

        if start:
            try:
                df_train = S["train_df"]
                df_train_p = resample_and_fill(
                    df_train[[st.session_state["ts_col"]] + st.session_state["feat_cols"]].copy(),
                    st.session_state["ts_col"],
                    st.session_state["samp_secs"],
                    st.session_state["feat_cols"]
                )
                scaler = zfit(df_train_p[st.session_state["feat_cols"]])
                Z = zapply(df_train_p[st.session_state["feat_cols"]], scaler, ZCLIP).to_numpy()
                win = max(1, int(st.session_state["win_min"]*60 / st.session_state["samp_secs"]))
                stride = max(1, int(st.session_state["stride_min"]*60 / st.session_state["samp_secs"]))
                X = make_windows(Z, win, stride)
            except Exception as e:
                st.error(f"Preprocessing error: {e}")
                X = np.empty((0,1,len(st.session_state["feat_cols"]) if "feat_cols" in st.session_state else 3))

            if X.shape[0] == 0:
                st.error("Not enough training windows; reduce WINDOW_MIN or provide more rows.")
            else:
                st.write(f"Training windows: **{X.shape[0]}** | Steps/window: **{X.shape[1]}** | Features: **{X.shape[2]}**")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                m = LSTMAE(n=X.shape[2], latent=int(LATENT), h=(int(H1), int(H2))).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=float(LR))
                lossf = nn.MSELoss()
                Xt = torch.tensor(X, dtype=torch.float32, device=device)

                loss_hist = []
                prog = st.progress(0, text="Training...")
                msg = st.empty()
                for ep in range(int(EPOCHS)):
                    m.train()
                    opt.zero_grad()
                    y = m(Xt)
                    loss = lossf(y, Xt)
                    loss.backward()
                    opt.step()
                    loss_hist.append(float(loss.item()))
                    prog.progress(int((ep+1)/EPOCHS*100), text=f"Training... {ep+1}/{EPOCHS}  loss={loss.item():.6f}")
                    if (ep+1) % max(1, EPOCHS//5) == 0:
                        msg.write(f"epoch {ep+1}/{EPOCHS}  loss={loss.item():.6f}")

                m.eval()
                with torch.no_grad():
                    recon = m(Xt).cpu().numpy()

                rmse = np.sqrt(((recon - X)**2).mean(axis=(1,2)))
                thr_global = float(rmse.mean() + 3.0*rmse.std())

                err_train = (recon - X)**2
                rmse_feat_train = np.sqrt(err_train.mean(axis=1))
                mu_feat  = rmse_feat_train.mean(axis=0)
                std_feat = rmse_feat_train.std(axis=0, ddof=0)
                per_feat_thr = (mu_feat + 3.0*std_feat).astype(float)

                S["model_state"] = m.state_dict()
                S["scaler"] = scaler
                S["threshold"] = thr_global
                S["per_feat_thr"] = per_feat_thr.tolist()
                S["win"] = win
                S["stride"] = stride
                S["loss_hist"] = loss_hist

                S["train_proc_df"] = df_train_p.copy()
                S["train_proc_X"]  = X.copy()

                st.success(f"Training done. Global threshold (RMSE) = **{thr_global:.6f}**")

                lcol, rcol = st.columns(2)
                with lcol:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=loss_hist, mode="lines", name="Train loss"))
                    fig.update_layout(title="Training loss", xaxis_title="Epoch", yaxis_title="MSE loss", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                with rcol:
                    hist = np.histogram(rmse, bins=30)
                    centers = (hist[1][1:] + hist[1][:-1]) / 2
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=centers, y=hist[0], name="RMSE"))
                    fig2.add_vline(x=thr_global, line_dash="dash",
                                   annotation_text=f"thr={thr_global:.4f}", annotation_position="top right")
                    fig2.update_layout(title="RMSE distribution (TRAIN)", xaxis_title="RMSE", yaxis_title="Count", height=350)
                    st.plotly_chart(fig2, use_container_width=True)

                thr_tbl = pd.DataFrame({"feature": st.session_state["feat_cols"], "per_feature_thr": per_feat_thr})
                st.caption("Per-feature thresholds (μ+3σ from TRAIN)")
                st.dataframe(thr_tbl, use_container_width=True)

# ======================
# Tab 3: Detect
# ======================
with tabs[2]:
    st.subheader("Detect on Live CSV (Parity-Safe)")
    if S["model_state"] is None or S["scaler"] is None:
        st.warning("Please train the model in Tab 2 first.")
    else:
        try:
            df_live = S["live_df"].copy()
            df_live_p = resample_and_fill(df_live[[st.session_state["ts_col"]] + st.session_state["feat_cols"]],
                                          st.session_state["ts_col"],
                                          st.session_state["samp_secs"],
                                          st.session_state["feat_cols"])
            Z_live = zapply(df_live_p[st.session_state["feat_cols"]], S["scaler"], st.session_state["zclip"]).to_numpy()
            win = S["win"]; stride = S["stride"]
            X_live = make_windows(Z_live, win, stride)

            same_feed = False
            try:
                same_feed = (
                    S.get("train_proc_df") is not None and
                    df_live_p.index.equals(S["train_proc_df"].index) and
                    np.array_equal(
                        df_live_p[st.session_state["feat_cols"]].to_numpy(),
                        S["train_proc_df"][st.session_state["feat_cols"]].to_numpy()
                    )
                )
            except Exception:
                same_feed = False

            if same_feed:
                st.info("🔁 Same-file parity detected — anomalies suppressed by design.")
                ts = str(pd.to_datetime(df_live_p.index).sort_values()[-1]) if len(df_live_p) > 0 else 'latest-window'
                st.markdown(_message_card(
                    status="NORMAL",
                    ts=ts,
                    top_feature="—",
                    rmse=0.0,
                    per_thr=0.0,
                    global_thr=float(S['threshold']),
                    k_thr=float(st.session_state["topfeat_k"]) * float(S['threshold']),
                    action="Parity run (Train == Live)."
                ), unsafe_allow_html=True)
                S["human_messages"].append({
                    "timestamp": ts,
                    "top_tag": "—",
                    "feature_rmse": "0.0",
                    "global_threshold": f"{float(S['threshold']):.6f}",
                    "message": f"{ts} | NORMAL — parity run (Train == Live)."
                })
                st.stop()

            if len(X_live) == 0:
                st.error("Not enough live windows; reduce WINDOW_MIN or provide more rows.")
            else:
                m = LSTMAE(n=Z_live.shape[1], latent=int(st.session_state["latent"]), h=(int(st.session_state["h1"]), int(st.session_state["h2"])))
                m.load_state_dict(S["model_state"]); m.eval()
                with torch.no_grad():
                    recon_live = m(torch.tensor(X_live, dtype=torch.float32)).numpy()
                rmse_live = np.sqrt(((recon_live - X_live)**2).mean(axis=(1,2)))

                thr = float(S["threshold"])
                flag = bool(len(rmse_live) >= 3 and np.all(rmse_live[-3:] > thr))
                st.write(f"Global anomaly flag (last 3 windows > thr): **{flag}**")

                err = (recon_live - X_live)**2
                rmse_feat = np.sqrt(err.mean(axis=1))
                last_vec = rmse_feat[-1]
                per_thr_vec = np.array(S.get("per_feat_thr", [thr]*len(last_vec)), float)
                k_thr = float(st.session_state["topfeat_k"]) * thr
                per_thr_used = np.maximum(per_thr_vec, k_thr)

                top_idx = int(np.argsort(last_vec)[::-1][0])
                top_tag = st.session_state["feat_cols"][top_idx]
                alias = S["alias"]
                top_tag_display = alias.get(top_tag, top_tag)
                top_val = float(last_vec[top_idx])

                status = 'ANOMALY' if top_val > float(per_thr_used[top_idx]) else 'NORMAL'
                ts = str(pd.to_datetime(df_live_p.index).sort_values()[-1]) if len(df_live_p) > 0 else 'latest-window'

                suggestions = {
                    "flow":     "Check inlet blockage/valve position; verify pump NPSH.",
                    "current":  "Inspect pump load: impeller fouling, suction blockage, mechanical drag.",
                    "pressure": "Check discharge valve/line restrictions; verify pressure transmitter.",
                    "Flow":     "Check inlet blockage/valve position; verify pump NPSH.",
                    "Current":  "Inspect pump load: impeller fouling, suction blockage, mechanical drag.",
                    "Pressure": "Check discharge valve/line restrictions; verify pressure transmitter.",
                }
                action = suggestions.get(top_tag_display, "Inspect related instrumentation and operating conditions.")

                st.markdown(_message_card(
                    status=status,
                    ts=ts,
                    top_feature=str(top_tag_display),
                    rmse=top_val,
                    per_thr=float(per_thr_vec[top_idx]),
                    global_thr=thr,
                    k_thr=k_thr,
                    action=action if status == "ANOMALY" else "Operating normally."
                ), unsafe_allow_html=True)

                msg = (f"{ts} | {status} — highest deviation in [{top_tag_display}] "
                       f"(feature RMSE={top_val:.4f}, per-feature thr_used={float(per_thr_used[top_idx]):.4f}, "
                       f"per-feature thr(μ+3σ)={float(per_thr_vec[top_idx]):.4f}, "
                       f"K×global={k_thr:.4f}, global thr={thr:.4f}).")
                if status == "ANOMALY":
                    msg += f" Suggested action: {action}"

                S["human_messages"].append({
                    "timestamp": ts,
                    "top_tag": top_tag_display,
                    "feature_rmse": f"{top_val:.6f}",
                    "global_threshold": f"{thr:.6f}",
                    "message": msg
                })

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=rmse_live, mode="lines", name="Live RMSE"))
                fig.add_hline(y=thr, line_dash="dash", annotation_text=f"global thr={thr:.4f}")
                fig.update_layout(title="Live RMSE by window", xaxis_title="Window index", yaxis_title="RMSE", height=350)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                feats_x = [str(c) for c in st.session_state["feat_cols"]]
                fig2.add_trace(go.Bar(x=feats_x, y=last_vec, name="RMSE (last window)"))
                fig2.add_trace(go.Scatter(x=feats_x, y=per_thr_vec, mode="lines+markers", name="Per-feature thr (μ+3σ)"))
                fig2.add_hline(y=k_thr, line_dash="dot", annotation_text=f"K×global = {k_thr:.4f}")
                fig2.update_layout(title="Per-feature RMSE vs Thresholds (last window)", xaxis_title="Feature", yaxis_title="RMSE", height=350)
                st.plotly_chart(fig2, use_container_width=True)

                st.caption("Session messages")
                st.dataframe(pd.DataFrame(S["human_messages"]), use_container_width=True)

        except Exception as e:
            st.error(f"Detection error: {e}")

# ======================
# Tab 4: Export
# ======================
with tabs[3]:
    st.subheader("Download Artifacts & Logs")
    artifacts = {}

    if S["scaler"] is not None:
        artifacts["scaler.json"] = json.dumps(S["scaler"], indent=2).encode("utf-8")

    if S["threshold"] is not None:
        artifacts["threshold.json"] = json.dumps({"threshold_rmse": float(S["threshold"])}, indent=2).encode("utf-8")

    if S.get("per_feat_thr") is not None:
        per_feat = {
            "features": st.session_state.get("feat_cols", []),
            "per_feature_thresholds": S["per_feat_thr"]
        }
        artifacts["per_feature_thresholds.json"] = json.dumps(per_feat, indent=2).encode("utf-8")

    config = {
        "timestamp_col": st.session_state.get("ts_col"),
        "feature_cols": st.session_state.get("feat_cols"),
        "sample_seconds": st.session_state.get("samp_secs"),
        "window_min": st.session_state.get("win_min"),
        "stride_min": st.session_state.get("stride_min"),
        "z_clip": st.session_state.get("zclip"),
        "topfeat_k": st.session_state.get("topfeat_k")
    }
    artifacts["config.json"] = json.dumps(config, indent=2).encode("utf-8")

    if S["model_state"] is not None:
        feats = len(st.session_state.get("feat_cols", FEATURE_COLS_STD))
        mdl = LSTMAE(n=feats, latent=int(st.session_state.get("latent", 16)),
                     h=(int(st.session_state.get("h1", 64)), int(st.session_state.get("h2", 32))))
        mdl.load_state_dict(S["model_state"])
        tmp_path = "well_lstm.pt"
        torch.save(mdl.state_dict(), tmp_path)
        with open(tmp_path, "rb") as f:
            artifacts["well_lstm.pt"] = f.read()
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    msgs = pd.DataFrame(S["human_messages"])
    if not msgs.empty:
        buf = io.StringIO(); msgs.to_csv(buf, index=False)
        artifacts["human_messages.csv"] = buf.getvalue().encode("utf-8")

    if not artifacts:
        st.info("Train & Detect first to generate artifacts.")
    else:
        cols = st.columns(2)
        i = 0
        for name, payload in artifacts.items():
            with cols[i % 2]:
                st.download_button(f"Download {name}", data=payload, file_name=name, mime="application/octet-stream", key=f"dl_{name}")
            i += 1

    st.caption("Keep scaler/threshold/per-feature thresholds/model/config together for future scoring.")
