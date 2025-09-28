# app.py
# Water Well LSTM Autoencoder — Anomaly Events with Per-Feature Thresholds
# -----------------------------------------------------------------------
# Tabs:
#  1) Data: upload & preview (XLSX for training, CSV for live), feature selection, lock
#  2) Train: LSTM-AE training, threshold selection (global + per-feature), loss chart
#  3) Detect: window anomalies + merged EVENTS (timestamps, top feature, action), CSV export
#  4) Settings/Export: threshold defaults, download model/scaler/config/profile; load artifacts
#
# Features: typically select these five → Current, Pressure, Flow rate, Water level, Frequency
# Event merge rule: gap_tolerance=1, min_event_windows=2 (debounces flicker but keeps short faults)

import io, os, json, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# ------------------------- Page config & styles -------------------------
st.set_page_config(page_title="Water Well — AE Anomaly Detector", layout="wide")
st.markdown("""
<style>
  .small-note { color:#6b7280; font-size:0.9rem; }
  .ok  { color:#16a34a; } .warn { color:#b45309; } .err { color:#dc2626; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Session State Helpers -------------------------
def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# Initialize session slots
ss_get("train_df", None)
ss_get("live_df", None)
ss_get("selected_cols", None)
ss_get("feature_mode", "Manual")   # Manual so you can pick your 5 columns
ss_get("feature_profile", None)
ss_get("scaler", None)
ss_get("model", None)
ss_get("train_time_values", None)
ss_get("live_time_values", None)
ss_get("config", {
    "window": 60,
    "hidden": 64,
    "latent": 32,
    "epochs": 15,
    "batch": 64,
    "lr": 1e-3,
    "threshold_policy": "Mu+K*Sigma",
    "thresh_K": 2.0,
    "percentile": 97.5,
    "iqr_alpha": 1.5,
    "fixed_threshold": 0.01
})
ss_get("train_err_stats", None)   # global: mu, sigma, thr, errors (list)
ss_get("feat_thr_stats", None)    # per-feature: {col: {"mu":..,"sigma":..,"thr":..,"errors":[...]}}
ss_get("live_errors", None)       # detect cache

# ------------------------- Data Utilities -------------------------
def _numeric_quality(df0: pd.DataFrame):
    df = df0.apply(pd.to_numeric, errors='coerce')
    nnr = 1.0 - df.isna().mean(axis=0)             # non-null ratio
    var = df.var(axis=0, ddof=0).fillna(0)
    score = nnr + (var > 0).astype(float)          # prefer columns with values & variance
    return df, score.sort_values(ascending=False)

def _select_columns(df_raw: pd.DataFrame, feat_cols=None, top_k=None):
    if feat_cols is not None:
        df = df_raw[feat_cols].copy()
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
        df = df.loc[:, df.var(axis=0, ddof=0) > 0]
        if df.shape[1] == 0:
            raise ValueError("Selected columns have no usable numeric data.")
        return df
    df_num, score = _numeric_quality(df_raw)
    keep = list(score.index[: (top_k or df_num.shape[1])])
    df = df_num[keep].dropna(axis=0, how='any')
    df = df.loc[:, df.var(axis=0, ddof=0) > 0]
    if df.shape[1] == 0:
        raise ValueError("No usable numeric columns after auto-selection.")
    return df

def read_training_xlsx(file, sheet=0, skip_top=0):
    header = None if skip_top > 0 else 0
    df = pd.read_excel(file, sheet_name=sheet, header=header, skiprows=skip_top)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    if header is None:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df

def read_live_csv(file, skip_top=0):
    header = "infer" if skip_top == 0 else None
    df = pd.read_csv(file, header=header, skiprows=skip_top)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    if header is None:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df

def make_windows(arr: np.ndarray, win: int):
    if arr.shape[0] < win:
        raise ValueError(f"Dataset too short for WINDOW={win}. Got length={arr.shape[0]}")
    X = []
    for i in range(len(arr) - win + 1):
        X.append(arr[i:i+win])
    return np.stack(X)

# ------------------------- Model -------------------------
class LSTMAE(nn.Module):
    def __init__(self, input_size: int, hidden=64, latent=32):
        super().__init__()
        self.enc = nn.LSTM(input_size, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.dec = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, input_size)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_z(h[-1])
        z_seq = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.dec(z_seq)
        return self.out(dec_out)

def train_model(model, loader, epochs=15, lr=1e-3, device="cpu", progress_cb=None):
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device); model.train()
    n = len(loader.dataset)
    for ep in range(1, epochs + 1):
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            yb = model(xb)
            loss = crit(yb, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        if progress_cb:
            progress_cb(ep, epochs, total / n)
    return model

# ------------------------- Threshold Policies -------------------------
def threshold_from_policy(train_errs: np.ndarray, policy: str, cfg):
    mu = float(train_errs.mean())
    sigma = float(train_errs.std())
    if policy == "Mu+K*Sigma":
        thr = mu + float(cfg["thresh_K"]) * sigma
    elif policy == "Percentile":
        thr = float(np.percentile(train_errs, float(cfg["percentile"])))
    elif policy == "IQR":
        q1 = np.percentile(train_errs, 25)
        q3 = np.percentile(train_errs, 75)
        thr = float(q3 + float(cfg["iqr_alpha"]) * (q3 - q1))
    elif policy == "Fixed":
        thr = float(cfg.get("fixed_threshold", mu + 2.0 * sigma))
    else:
        thr = mu + 2.0 * sigma
    return {"mu": mu, "sigma": sigma, "thr": thr}

# ------------------------- Plots -------------------------
def plot_error_with_threshold(errors: np.ndarray, thr: float, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=errors, mode="lines", name="Reconstruction MSE"))
    fig.add_hline(y=thr, line_dash="dash", annotation_text=f"threshold={thr:.4g}", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Window Index", yaxis_title="MSE")
    return fig

def plot_error_histogram(train_errs: np.ndarray, thr: float, title: str):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train_errs, nbinsx=50, name="Train MSE", opacity=0.8))
    fig.add_vline(x=thr, line_dash="dash", annotation_text=f"threshold={thr:.4g}", annotation_position="top")
    fig.update_layout(title=title, xaxis_title="MSE", yaxis_title="Count", barmode="overlay")
    return fig

# ------------------------- Tab 1: Data -------------------------
def tab_data():
    st.header("1) Data — Upload & Feature Selection")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Training Excel file (.xlsx)")
        xlsx_file = st.file_uploader("Upload training Excel file", type=["xlsx"], key="xlsx_up")
        train_sheet = st.number_input("Excel sheet index", min_value=0, step=1, value=0, key="sheet_idx")
        skip_excel = st.number_input("Skip top rows (Excel)", min_value=0, step=1, value=0, key="skip_excel")
    with colB:
        st.subheader("Data to be checked (.csv)")
        csv_file = st.file_uploader("Upload data file", type=["csv"], key="csv_up")
        skip_csv = st.number_input("Skip top rows (CSV)", min_value=0, step=1, value=0, key="skip_csv")

    if xlsx_file is not None:
        try:
            df_tr_raw = read_training_xlsx(xlsx_file, sheet=train_sheet, skip_top=skip_excel)
            st.write("**Train preview**", df_tr_raw.head())
            st.caption(f"Train shape: {df_tr_raw.shape}")
        except Exception as e:
            st.error(f"Error reading training Excel: {e}")
            df_tr_raw = None
    else:
        df_tr_raw = None

    if csv_file is not None:
        try:
            df_lv_raw = read_live_csv(csv_file, skip_top=skip_csv)
            st.write("**Live preview**", df_lv_raw.head())
            st.caption(f"Live shape: {df_lv_raw.shape}")
        except Exception as e:
            st.error(f"Error reading live CSV: {e}")
            df_lv_raw = None
    else:
        df_lv_raw = None

    st.markdown("---")
    st.subheader("Feature Selection")

    feature_mode = st.radio("Mode", ["Manual", "Auto Top-K", "Profile"], horizontal=True, key="feat_mode_radio")
    st.session_state["feature_mode"] = feature_mode

    selected_cols = None
    if feature_mode == "Auto Top-K":
        k = st.slider("Top-K numeric columns", min_value=1, max_value=16, value=5, step=1, key="topk_auto")
        if df_tr_raw is not None:
            try:
                df_auto = _select_columns(df_tr_raw, feat_cols=None, top_k=k)
                selected_cols = list(df_auto.columns)
                st.success(f"Auto-selected columns from training: {selected_cols}")
            except Exception as e:
                st.error(f"Auto-selection failed: {e}")
    elif feature_mode == "Manual":
        cols_union = []
        if df_tr_raw is not None:
            cols_union = list(df_tr_raw.columns)
        if df_lv_raw is not None:
            cols_union = sorted(list(set(cols_union) | set(df_lv_raw.columns)))
        default5 = [c for c in ["Current","Pressure","Flow rate","Water level","Frequency"] if c in cols_union][:5]
        picked = st.multiselect("Choose feature columns (select your 5)", options=cols_union, default=default5, key="manual_cols")
        if picked:
            selected_cols = list(picked)
            st.info(f"Selected: {selected_cols}")
    else:  # Profile JSON
        profile_json = st.text_area("Paste feature profile JSON",
                                    value=st.session_state["feature_profile"] and json.dumps(st.session_state["feature_profile"], indent=2) or "",
                                    key="profile_text")
        if st.button("Load profile", key="btn_load_profile"):
            try:
                prof = json.loads(profile_json)
                if not isinstance(prof, dict) or "columns" not in prof:
                    raise ValueError("Profile must be a dict with key 'columns'.")
                selected_cols = list(prof["columns"])
                st.session_state["feature_profile"] = prof
                st.success(f"Loaded profile. Columns = {selected_cols}")
            except Exception as e:
                st.error(f"Invalid profile: {e}")

    if st.button("Lock data & features", type="primary",
                 disabled=(df_tr_raw is None or df_lv_raw is None or not selected_cols),
                 key="btn_lock"):
        try:
            df_tr = _select_columns(df_tr_raw, feat_cols=selected_cols, top_k=None)
            df_lv = df_lv_raw[selected_cols].copy()
            df_lv = df_lv.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
            df_lv = df_lv.loc[:, df_lv.var(axis=0, ddof=0) > 0]
            missing = [c for c in selected_cols if c not in df_lv.columns]
            if missing:
                raise ValueError(f"Live CSV missing selected columns: {missing}")

            st.session_state["train_df"] = df_tr
            st.session_state["live_df"]  = df_lv
            st.session_state["selected_cols"] = selected_cols

            # Capture Time columns if present
            if "Time" in (df_tr_raw.columns if df_tr_raw is not None else []):
                try:
                    st.session_state["train_time_values"] = pd.to_datetime(df_tr_raw["Time"]).reset_index(drop=True)
                except Exception:
                    st.warning("Could not parse 'Time' from training file; training time will be row-based.")
            if "Time" in (df_lv_raw.columns if df_lv_raw is not None else []):
                try:
                    st.session_state["live_time_values"] = pd.to_datetime(df_lv_raw["Time"]).reset_index(drop=True)
                except Exception:
                    st.warning("Could not parse 'Time' from live CSV; events will be row-indexed.")
            else:
                st.info("Tip: Add a 'Time' column to your live CSV to get timestamped events.")

            want5 = {"Current","Pressure","Flow rate","Water level","Frequency"}
            have = set(selected_cols)
            if want5.issubset(have):
                st.caption("Training & detection will use 5 features: Current, Pressure, Flow rate, Water level, Frequency.")
            else:
                st.warning(f"You selected {selected_cols}. For your standard setup, include exactly: {sorted(want5)}")

            st.success(f"Locked. Train shape={df_tr.shape}, Live shape={df_lv.shape}. Columns={selected_cols}")
        except Exception as e:
            st.error(f"Failed to lock: {e}")

# ------------------------- Tab 2: Train -------------------------
def tab_train():
    st.header("2) Train — LSTM Autoencoder")
    cfg = st.session_state["config"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        cfg["window"] = st.number_input("Window", min_value=8, max_value=2048, value=int(cfg["window"]), step=1, key="window_input")
    with c2:
        cfg["hidden"] = st.number_input("Hidden", min_value=8, max_value=1024, value=int(cfg["hidden"]), step=8, key="hidden_units")
    with c3:
        cfg["latent"] = st.number_input("Latent", min_value=4, max_value=512, value=int(cfg["latent"]), step=4, key="latent_units")
    with c4:
        cfg["epochs"] = st.number_input("Epochs", min_value=1, max_value=500, value=int(cfg["epochs"]), step=1, key="epochs_input")
    with c5:
        cfg["batch"] = st.number_input("Batch", min_value=1, max_value=4096, value=int(cfg["batch"]), step=1, key="batch_input")
    with c6:
        cfg["lr"] = float(st.text_input("LR", value=str(cfg["lr"]), key="lr_input"))

    st.session_state["config"] = cfg

    df_tr = st.session_state["train_df"]
    df_lv = st.session_state["live_df"]
    cols = st.session_state["selected_cols"]

    if df_tr is None or df_lv is None or not cols:
        st.warning("Please finish the Data tab (upload + lock) first.")
        return

    # Adjust window if needed
    min_len = min(len(df_tr), len(df_lv))
    win = int(cfg["window"])
    if min_len <= win:
        new_w = max(8, min_len - 1)
        st.warning(f"Reducing WINDOW from {win} to {new_w} due to short data.")
        win = new_w

    # Fit scaler on training
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(df_tr.values)

    # Make windows
    try:
        Xtr_w = make_windows(Xtr, win)   # (nW, win, F)
    except Exception as e:
        st.error(f"Windowing error: {e}")
        return

    st.write(f"Train windows: {Xtr_w.shape}  |  Features: {Xtr_w.shape[-1]} ({cols})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Using device: `{device}`")

    model = LSTMAE(input_size=Xtr_w.shape[-1], hidden=int(cfg["hidden"]), latent=int(cfg["latent"]))
    tr_ds = TensorDataset(torch.tensor(Xtr_w, dtype=torch.float32))
    tr_dl = DataLoader(tr_ds, batch_size=int(cfg["batch"]), shuffle=True)

    prog = st.progress(0.0)
    loss_chart = st.empty()
    tracked = []

    def on_progress(ep, total, loss_val):
        tracked.append((ep, loss_val))
        prog.progress(ep / total)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[e for e, _ in tracked], y=[l for _, l in tracked],
                                 mode="lines+markers", name="train_loss"))
        fig.update_layout(title="Training loss (MSE)", xaxis_title="Epoch", yaxis_title="Loss")
        loss_chart.plotly_chart(fig, use_container_width=True)

    if st.button("Start Training", type="primary", key="btn_start_training"):
        model = train_model(model, tr_dl, epochs=int(cfg["epochs"]),
                            lr=float(cfg["lr"]), device=device, progress_cb=on_progress)
        model.eval()
        with torch.no_grad():
            Rtr = model(torch.tensor(Xtr_w, dtype=torch.float32).to(device)).cpu().numpy()

        # Global training errors (per window)
        err_tr = ((Rtr - Xtr_w) ** 2).mean(axis=(1, 2))

        # Per-feature training errors (per window), averaged across time steps
        per_feat_err_tr = ((Rtr - Xtr_w) ** 2).mean(axis=1)  # (nW, F)

        # ---- Threshold policy selection (from training errors) ----
        st.subheader("Threshold Policy (from training errors)")
        policy = st.selectbox(
            "Policy",
            ["Mu+K*Sigma", "Percentile", "IQR", "Fixed"],
            index=["Mu+K*Sigma","Percentile","IQR","Fixed"].index(cfg["threshold_policy"]),
            key="policy_train"
        )
        if policy == "Mu+K*Sigma":
            cfg["thresh_K"] = st.slider("K (σ)", 0.5, 5.0, float(cfg["thresh_K"]), 0.1, key="thresh_k_train")
        elif policy == "Percentile":
            cfg["percentile"] = st.slider("Percentile", 90.0, 99.9, float(cfg["percentile"]), 0.1, key="percentile_train")
        elif policy == "IQR":
            cfg["iqr_alpha"] = st.slider("α (IQR multiplier)", 0.5, 5.0, float(cfg["iqr_alpha"]), 0.1, key="iqr_train")
        else:
            default_mu2s = float(err_tr.mean() + 2.0 * err_tr.std())
            cfg["fixed_threshold"] = float(
                st.text_input("Fixed threshold (MSE)", value=f"{default_mu2s:.6g}", key="fixed_thr_train")
            )
        st.session_state["config"]["threshold_policy"] = policy

        # Global threshold stats
        stats_global = threshold_from_policy(err_tr, policy, cfg)
        mu, sigma, thr = stats_global["mu"], stats_global["sigma"], stats_global["thr"]
        st.success(f"Global train errors → μ={mu:.6g}, σ={sigma:.6g}, threshold={thr:.6g}")
        st.plotly_chart(plot_error_histogram(err_tr, thr, "Training errors distribution"), use_container_width=True)

        # Per-feature thresholds
        feat_thr_stats = {}
        rows = []
        for j, c in enumerate(cols):
            s = threshold_from_policy(per_feat_err_tr[:, j], policy, cfg)
            feat_thr_stats[c] = {"mu": s["mu"], "sigma": s["sigma"], "thr": s["thr"], "errors": per_feat_err_tr[:, j].tolist()}
            rows.append({"Feature": c, "mu": s["mu"], "sigma": s["sigma"], "threshold": s["thr"]})
        st.markdown("#### Per-feature thresholds (from training)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Save artifacts to session
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["config"]["window"] = win
        st.session_state["train_err_stats"] = {"mu": mu, "sigma": sigma, "thr": thr, "errors": err_tr.tolist()}
        st.session_state["feat_thr_stats"] = feat_thr_stats

# ------------------------- Tab 3 helpers (Events) -------------------------
def _merge_anomalous_windows(end_idxs, win_size, gap_tolerance=1, min_event_windows=2):
    """
    Group anomalous window end indices into events.
    Returns list of dicts: {start_idx, end_idx, idx_list}.
    """
    end_idxs = [int(x) for x in sorted(end_idxs)]
    events = []
    if not end_idxs:
        return events

    run = [end_idxs[0]]
    for i in range(1, len(end_idxs)):
        # Adjacent anomalous windows have end indices differing by 1; allow 1 normal gap -> diff <= 2
        if (end_idxs[i] - end_idxs[i-1]) <= (gap_tolerance + 1):
            run.append(end_idxs[i])
        else:
            if len(run) >= min_event_windows:
                events.append({
                    "start_idx": run[0] - (win_size - 1),
                    "end_idx":   run[-1],
                    "idx_list":  run.copy()
                })
            run = [end_idxs[i]]

    if len(run) >= min_event_windows:
        events.append({
            "start_idx": run[0] - (win_size - 1),
            "end_idx":   run[-1],
            "idx_list":  run.copy()
        })
    return events

def _recommended_action_for_feature(feat_name: str) -> str:
    name = (feat_name or "").strip().lower()
    actions = {
        "flow rate": (
            "• Check suction/discharge valves & strainers for blockage/leaks\n"
            "• Verify setpoint & sensor calibration\n"
            "• Inspect pipeline restrictions or air ingress"
        ),
        "pressure": (
            "• Inspect for suction leaks, cavitation, or discharge blockage\n"
            "• Verify pressure transmitter health & impulse lines\n"
            "• Check pump operating point vs system curve"
        ),
        "current": (
            "• Check for mechanical friction, bearing wear, or misalignment\n"
            "• Inspect impeller blockage or high load conditions\n"
            "• Review VFD limits/alarms and phase imbalance"
        ),
        "water level": (
            "• Verify well static/dynamic level; consider low-level protection\n"
            "• Check pump NPSH margin; risk of cavitation\n"
            "• Review drawdown vs pump curve and throttling"
        ),
        "frequency": (
            "• Verify VFD setpoint/commands and PID tuning\n"
            "• Check power quality & drive alarms\n"
            "• Confirm ramps/limits aren’t constraining control"
        ),
    }
    for key in actions:
        if key in name:
            return actions[key]
    return "• Inspect process & instrumentation around this feature\n• Verify sensor health and recent setpoint changes"

# ------------------------- Tab 3: Detect (Events + Actions) -------------------------
def tab_detect():
    st.header("3) Detect — Live CSV Inference (Events + Actions)")
    try:
        df_lv = st.session_state["live_df"]
        df_tr = st.session_state["train_df"]
        cols   = st.session_state["selected_cols"]
        model  = st.session_state["model"]
        scaler = st.session_state["scaler"]
        cfg    = st.session_state["config"]
        stats_g  = st.session_state["train_err_stats"]
        stats_pf = st.session_state.get("feat_thr_stats", None)
    except KeyError as e:
        st.error(f"Missing session key: {e}. Please complete Data and Train tabs first.")
        return

    if any(x is None for x in [df_lv, df_tr, cols, model, scaler, stats_g]):
        st.warning("Please complete Data and Train tabs first.")
        return

    # Timestamps
    time_values = st.session_state.get("live_time_values", None)
    have_time = (time_values is not None) and (len(time_values) == len(df_lv))

    # Scale & window
    win = int(cfg["window"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        Xlv = scaler.transform(df_lv.values)
        Xlv_w = make_windows(Xlv, win)  # (nW, win, F)
    except Exception as e:
        st.error(f"Live scaling/windowing error: {e}")
        return

    model.eval()
    with torch.no_grad():
        Rlv = model(torch.tensor(Xlv_w, dtype=torch.float32).to(device)).cpu().numpy()

    # Errors
    err_lv = ((Rlv - Xlv_w) ** 2).mean(axis=(1, 2))      # global per-window
    per_feat_err = ((Rlv - Xlv_w) ** 2).mean(axis=1)     # (nW, F) mean across time

    # Thresholds: global + per-feature
    policy = cfg["threshold_policy"]
    global_thr = float(threshold_from_policy(
        np.array(st.session_state["train_err_stats"]["errors"]), policy, cfg)["thr"]
    )
    is_anom_global = (err_lv > global_thr)

    # Build per-feature threshold vector aligned to 'cols'
    feat_thr_vec = None
    if stats_pf is not None:
        try:
            feat_thr_vec = np.array([stats_pf[c]["thr"] for c in cols], dtype=float)  # (F,)
        except Exception:
            feat_thr_vec = None

    if feat_thr_vec is not None:
        is_anom_feat_mat = (per_feat_err > feat_thr_vec[None, :])  # (nW, F)
        is_anom_any_feat = is_anom_feat_mat.any(axis=1)
    else:
        is_anom_feat_mat = np.zeros_like(per_feat_err, dtype=bool)
        is_anom_any_feat = np.zeros(err_lv.shape, dtype=bool)
        st.warning("Per-feature thresholds not available (train Tab 2 after adding). Using global threshold only.")

    # Decision rule selector
    decision = st.selectbox(
        "Decision rule",
        ["Global OR Per-feature (recommended)", "Per-feature only", "Global only"],
        index=0,
        key="decision_rule_select"
    )
    if decision == "Per-feature only":
        is_anom = is_anom_any_feat
    elif decision == "Global only":
        is_anom = is_anom_global
    else:  # OR
        is_anom = np.logical_or(is_anom_global, is_anom_any_feat)

    # Map window -> ending row index
    end_idx = np.arange(win - 1, win - 1 + len(err_lv))

    # Window-level dataframe
    out = pd.DataFrame({
        "row_idx_end": end_idx,
        "recon_mse": err_lv,
        "thr_global": global_thr,
        "is_anom_global": is_anom_global,
        "is_anomaly": is_anom
    })
    # live values at window end
    for c in cols:
        out[c] = df_lv.iloc[end_idx][c].values
    # per-feature errors, shares, thresholds and flags
    sum_err = per_feat_err.sum(axis=1) + 1e-12
    for j, c in enumerate(cols):
        out[f"err_{c}"] = per_feat_err[:, j]
        out[f"share_{c}"] = per_feat_err[:, j] / sum_err
        thr_c = float(feat_thr_vec[j]) if feat_thr_vec is not None else np.nan
        out[f"thr_{c}"] = thr_c
        out[f"is_anom_{c}"] = (per_feat_err[:, j] > thr_c) if np.isfinite(thr_c) else False
    if have_time:
        out["Time_end"] = time_values.iloc[end_idx].values

    st.session_state["live_errors"] = {
        "err_lv": err_lv.tolist(),
        "thr": float(global_thr),
        "is_anom": is_anom.tolist(),
        "end_idx": end_idx.tolist(),
        "out_df": out
    }

    # Plots + metrics
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(
            plot_error_with_threshold(err_lv, global_thr, "Live reconstruction error vs GLOBAL threshold"),
            use_container_width=True
        )
    with c2:
        st.metric("Anomalous windows", int(is_anom.sum()))
        st.caption(f"Policy: {policy} | Rule: {decision}")

    st.markdown("#### Window-level anomalies")
    st.dataframe(out, use_container_width=True)
    st.download_button(
        "Download anomaly_windows.csv",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="anomaly_windows.csv",
        mime="text/csv",
        key="dl_windows_csv"
    )

    # Merge anomalous windows → events (optimum rule: gap=1, min 2 windows)
    anom_end_rows = end_idx[np.where(is_anom)[0]]
    if len(anom_end_rows) == 0:
        st.info("No anomalies to merge at current threshold/rule.")
        return

    events_runs = _merge_anomalous_windows(
        end_idxs=anom_end_rows,
        win_size=win,
        gap_tolerance=1,
        min_event_windows=2
    )

    # Build events table
    event_rows = []
    row_to_widx = {int(r): int(i) for i, r in enumerate(end_idx)}
    for eid, ev in enumerate(events_runs, start=1):
        idx_list = [int(r) for r in ev["idx_list"]]
        w_inds = [row_to_widx[r] for r in idx_list if r in row_to_widx]
        if not w_inds:
            continue

        mse_vals = err_lv[w_inds]
        max_mse = float(np.max(mse_vals))
        mean_mse = float(np.mean(mse_vals))

        ev_feat_err = per_feat_err[w_inds, :]
        mean_feat_err = ev_feat_err.mean(axis=0)
        shares = mean_feat_err / (mean_feat_err.sum() + 1e-12)

        j_top = int(np.argmax(mean_feat_err))
        top_feat = cols[j_top]
        top_share = float(shares[j_top])

        start_idx = int(ev["start_idx"])
        end_idx_ev = int(ev["end_idx"])
        duration_rows = end_idx_ev - start_idx + 1

        if have_time and 0 <= start_idx < len(time_values) and 0 <= end_idx_ev < len(time_values):
            start_time = pd.to_datetime(time_values.iloc[start_idx])
            end_time   = pd.to_datetime(time_values.iloc[end_idx_ev])
            duration_s = (end_time - start_time).total_seconds()
        else:
            start_time = None; end_time = None; duration_s = None

        action_text = _recommended_action_for_feature(top_feat)

        event_rows.append({
            "event_id": eid,
            "start_idx": start_idx,
            "end_idx": end_idx_ev,
            "start_time": start_time,
            "end_time": end_time,
            "duration_rows": duration_rows,
            "duration_seconds": duration_s,
            "count_windows": len(w_inds),
            "max_mse": max_mse,
            "mean_mse": mean_mse,
            "top_feature": top_feat,
            "top_feature_share": top_share,
            "top3_features": ", ".join([f"{cols[j]} ({shares[j]:.0%})" for j in np.argsort(-shares)[:3]]),
            "recommended_action": action_text
        })

    df_events = pd.DataFrame(event_rows)

    # Metrics & table
    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("Events merged", len(df_events))
    with c4:
        st.metric("First event time", str(df_events["start_time"].min()) if have_time and len(df_events) else "—")
    with c5:
        st.metric("Last event time", str(df_events["end_time"].max()) if have_time and len(df_events) else "—")

    st.markdown("#### Event-level summary (merged)")
    st.dataframe(df_events, use_container_width=True)
    st.download_button(
        "Download anomaly_events.csv",
        data=df_events.to_csv(index=False).encode("utf-8"),
        file_name="anomaly_events.csv",
        mime="text/csv",
        key="dl_events_csv"
    )

    # Inspect a specific event
    st.markdown("### Inspect an event")
    if len(df_events) == 0:
        st.info("No events to inspect.")
        return

    eid_pick = int(st.number_input("Choose event_id", min_value=1, max_value=int(df_events["event_id"].max()), value=1))
    ev_row = df_events[df_events["event_id"] == eid_pick].iloc[0]
    start = int(ev_row["start_idx"]); stop  = int(ev_row["end_idx"]) + 1
    start = max(0, start); stop = min(len(df_lv), stop)

    x_axis = list(range(start, stop))
    if have_time:
        x_axis = time_values.iloc[start:stop].astype(str).tolist()

    fig_tr = go.Figure()
    for c in cols:
        fig_tr.add_trace(go.Scatter(x=x_axis, y=df_lv[c].iloc[start:stop].values, mode="lines", name=c))
    fig_tr.update_layout(
        title=f"Feature traces during Event #{eid_pick} (rows {start}:{stop-1})",
        xaxis_title=("Time" if have_time else "Row"),
        yaxis_title="Value"
    )
    st.plotly_chart(fig_tr, use_container_width=True)

    # Per-feature error share bar chart for the chosen event
    idx_list = list(range(start + (win - 1), stop))  # end rows inside event
    w_inds = [row_to_widx[r] for r in idx_list if r in row_to_widx]
    if w_inds:
        ev_feat_err = per_feat_err[w_inds, :]
        mean_feat_err = ev_feat_err.mean(axis=0)
        shares = mean_feat_err / (mean_feat_err.sum() + 1e-12)
        fig_sh = go.Figure()
        fig_sh.add_trace(go.Bar(x=cols, y=shares))
        fig_sh.update_layout(title=f"Per-feature error share — Event #{eid_pick}",
                             xaxis_title="Feature", yaxis_title="Share")
        st.plotly_chart(fig_sh, use_container_width=True)

    st.markdown("**Recommended action for this event**")
    st.code(str(ev_row["recommended_action"]), language="markdown")

# ------------------------- Tab 4: Settings / Export -------------------------
def tab_settings_export():
    st.header("4) Settings / Export")

    cfg = st.session_state["config"]
    st.subheader("Threshold Defaults")
    policy = st.selectbox(
        "Default policy",
        ["Mu+K*Sigma", "Percentile", "IQR", "Fixed"],
        index=["Mu+K*Sigma","Percentile","IQR","Fixed"].index(cfg["threshold_policy"]),
        key="policy_settings"
    )
    cfg["threshold_policy"] = policy
    if policy == "Mu+K*Sigma":
        cfg["thresh_K"] = st.slider("K (σ)", 0.5, 5.0, float(cfg["thresh_K"]), 0.1, key="thresh_k_settings")
    elif policy == "Percentile":
        cfg["percentile"] = st.slider("Percentile", 90.0, 99.9, float(cfg["percentile"]), 0.1, key="percentile_settings")
    elif policy == "IQR":
        cfg["iqr_alpha"] = st.slider("α (IQR)", 0.5, 5.0, float(cfg["iqr_alpha"]), 0.1, key="iqr_settings")
    else:  # Fixed
        cfg["fixed_threshold"] = float(
            st.text_input("Fixed threshold (MSE)", value=str(cfg.get("fixed_threshold", 0.01)), key="fixed_thr_settings")
        )
    st.session_state["config"] = cfg

    # Save/Load artifacts
    st.subheader("Artifacts")
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    selected_cols = st.session_state["selected_cols"]
    profile = {"columns": selected_cols} if selected_cols else None

    c1, c2, c3 = st.columns(3)
    with c1:
        if model is not None:
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            st.download_button("Download model.pt", data=buffer.getvalue(), file_name="model.pt", key="dl_model")
        else:
            st.caption("Model not trained yet.")
    with c2:
        if scaler is not None:
            st.download_button("Download scaler.pkl", data=pickle.dumps(scaler), file_name="scaler.pkl", key="dl_scaler")
        else:
            st.caption("Scaler not ready.")
    with c3:
        st.download_button("Download config.json", data=json.dumps(cfg, indent=2).encode("utf-8"),
                           file_name="config.json", key="dl_config")

    st.subheader("Feature Profile")
    if profile:
        st.download_button("Download feature_profile.json",
                           data=json.dumps(profile, indent=2).encode("utf-8"),
                           file_name="feature_profile.json", key="dl_profile")
    else:
        st.caption("No feature profile yet (lock data in Tab 1).")

    st.markdown("---")
    st.subheader("Load Existing Artifacts")
    up_model = st.file_uploader("Upload model.pt", type=["pt"], key="up_model")
    up_scaler = st.file_uploader("Upload scaler.pkl", type=["pkl"], key="up_scaler")
    up_config = st.file_uploader("Upload config.json", type=["json"], key="up_config")
    if st.button("Load artifacts", key="btn_load_artifacts"):
        try:
            if up_config:
                st.session_state["config"] = json.load(io.TextIOWrapper(up_config, encoding="utf-8"))
            if up_scaler:
                st.session_state["scaler"] = pickle.load(up_scaler)
            if up_model:
                df_ref = st.session_state["train_df"] or st.session_state["live_df"]
                if df_ref is None:
                    st.warning("Cannot infer model input size (no data locked). Please lock data first, then load model.")
                else:
                    input_size = df_ref.shape[1]
                    cfg = st.session_state["config"]
                    model = LSTMAE(input_size=input_size, hidden=int(cfg["hidden"]), latent=int(cfg["latent"]))
                    model.load_state_dict(torch.load(up_model, map_location="cpu"))
                    st.session_state["model"] = model
            st.success("Artifacts loaded.")
        except Exception as e:
            st.error(f"Failed to load artifacts: {e}")

# ------------------------- Main -------------------------
def main():
    st.title("Water Wells — LSTM Autoencoder Anomaly Events (Per-Feature Thresholds)")
    st.caption("Upload training (XLSX) & live (CSV), pick 5 features, train, detect events with timestamps and actions. Per-feature thresholds catch single-feature spikes.")

    tabs = st.tabs(["Data", "Train", "Detect", "Settings/Export"])
    with tabs[0]:
        tab_data()
    with tabs[1]:
        tab_train()
    with tabs[2]:
        tab_detect()
    with tabs[3]:
        tab_settings_export()

if __name__ == "__main__":
    main()
