"""
app.py  –  MLB Win Predictor (Streamlit)
Run: streamlit run app.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

st.set_page_config(page_title="MLB Win Predictor", page_icon="⚾",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.main-header{font-size:2.2rem;font-weight:700;color:#003087;text-align:center;padding:.4rem 0 .1rem}
.sub-header{font-size:.95rem;color:#666;text-align:center;margin-bottom:1.5rem}
.metric-card{background:#f0f4ff;border-radius:10px;padding:1rem;text-align:center;border:1px solid #ccd9ff}
.win-label{color:#1a7a1a;font-weight:700;font-size:1.4rem}
.loss-label{color:#b00020;font-weight:700;font-size:1.4rem}
.section-title{font-size:1.05rem;font-weight:600;color:#003087;
  border-bottom:2px solid #003087;margin:1rem 0 .5rem;padding-bottom:.2rem}
.game-card{background:#fff;border:1px solid #dde4f5;border-radius:8px;
  padding:.8rem 1rem;margin-bottom:.5rem}
.favored{color:#003087;font-weight:700}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"
DATA_DIR   = "data"

DS1_COLS = [f"B{i}" for i in range(1,14)] + [f"SP{i}" for i in range(1,17)] + ["X1"]
DS2_COLS = [f"B{i}" for i in range(1,14)] + [f"P{i}"  for i in range(1,19)] + ["X1"]

BATTING_FIELDS = {
    "B1":("At Bats",0,50,30),"B2":("Hits",0,25,8),"B3":("Walks",0,15,3),
    "B4":("Strikeouts",0,25,9),"B5":("Plate Appearances",0,55,36),
    "B6":("Batting Avg",0.0,.5,.25),"B7":("OBP",0.0,.6,.32),
    "B8":("SLG",0.0,.9,.40),"B9":("OPS",0.0,1.5,.72),
    "B10":("Pitches/PA",0,180,135),"B11":("Strikes",0,130,88),
    "B12":("Putouts",0,55,27),"B13":("Assists",0,20,6),
}
SP_FIELDS = {
    "SP1":("IP",0.0,9.0,5.2),"SP2":("H Allowed",0,20,6),"SP3":("BB",0,10,2),
    "SP4":("K",0,20,6),"SP5":("HR Allowed",0,6,1),"SP6":("ERA",0.0,20.0,3.8),
    "SP7":("BF",0,45,23),"SP8":("Pitches",0,140,90),"SP9":("Strikes",0,100,60),
    "SP10":("Contact K",0,60,35),"SP11":("Swinging K",0,30,10),
    "SP12":("Looking K",0,30,15),"SP13":("GB",0,25,7),
    "SP14":("FB",0,25,7),"SP15":("LD",0,20,5),"SP16":("Game Score",-10,100,45),
}
ALL_P_FIELDS = {
    "P1":("IP (All P)",0.0,9.0,9.0),"P2":("H Allowed (All P)",0,25,8),
    "P3":("BB (All P)",0,15,3),"P4":("K (All P)",0,25,9),
    "P5":("HR (All P)",0,6,1),"P6":("ERA (All P)",0.0,20.0,4.0),
    "P7":("BF (All P)",0,45,36),"P8":("Pitches (All P)",0,200,145),
    "P9":("Strikes (All P)",0,150,95),"P10":("Contact K (All P)",0,80,55),
    "P11":("Swinging K (All P)",0,40,14),"P12":("Looking K (All P)",0,40,26),
    "P13":("GB (All P)",0,30,10),"P14":("FB (All P)",0,30,10),
    "P15":("LD (All P)",0,25,7),"P16":("Game Score (All P)",-10,100,40),
    "P17":("Inherited Runners",0,10,0),"P18":("Inherited Score",0,10,0),
}

TEAM_NAMES = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles",
    "BOS":"Boston Red Sox","CHC":"Chicago Cubs","CHW":"Chicago White Sox",
    "CIN":"Cincinnati Reds","CLE":"Cleveland Guardians","COL":"Colorado Rockies",
    "DET":"Detroit Tigers","HOU":"Houston Astros","KCR":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins",
    "MIL":"Milwaukee Brewers","MIN":"Minnesota Twins","NYM":"New York Mets",
    "NYY":"New York Yankees","OAK":"Athletics","PHI":"Philadelphia Phillies",
    "PIT":"Pittsburgh Pirates","SDP":"San Diego Padres","SEA":"Seattle Mariners",
    "SFG":"San Francisco Giants","STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays",
    "TEX":"Texas Rangers","TOR":"Toronto Blue Jays","WSN":"Washington Nationals",
}


# ── Resource loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models_cached(ds_num, use_fs):
    suffix = "fs" if use_fs else "no_fs"
    loaded, missing = {}, []
    for tag, disp in [("svm","SVM"),("ann","ANN"),("1dcnn","1DCNN")]:
        if use_fs and tag == "1dcnn":
            continue
        key  = f"ds{ds_num}_{suffix}_{tag}"
        pkl  = os.path.join(MODELS_DIR, f"{key}.pkl")
        kp   = os.path.join(MODELS_DIR, f"{key}.keras")
        if os.path.exists(pkl):
            loaded[disp] = ("sklearn", joblib.load(pkl))
        elif os.path.exists(kp):
            try:
                import tensorflow as tf
                loaded[disp] = ("keras", tf.keras.models.load_model(kp))
            except Exception as e:
                missing.append(f"{disp}: {e}")
        else:
            missing.append(disp)
    return loaded, missing


@st.cache_data
def load_scaler(ds_num, use_fs):
    suffix = "_selected" if use_fs else ""
    path   = os.path.join(DATA_DIR, f"scaler_ds{ds_num}{suffix}.csv")
    if not os.path.exists(path):
        return None, None
    sc = pd.read_csv(path)
    return sc["min"].values.astype("float32"), sc["max"].values.astype("float32")


@st.cache_data(ttl=3600)
def load_season_avgs(year):
    path = os.path.join(DATA_DIR, f"team_season_avgs_{year}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_split_summary():
    p = os.path.join(DATA_DIR, "split_summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


@st.cache_data
def load_training_summary():
    p = os.path.join(MODELS_DIR, "training_summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ── Prediction helpers ────────────────────────────────────────────────────────
def minmax_norm(x, mn, mx):
    rng = mx - mn; rng[rng == 0] = 1.0
    return ((x - mn) / rng).clip(0, 1).astype("float32")


def run_predict(models, x_norm):
    from models import prepare_1dcnn_input
    probs = {}
    for name, (kind, model) in models.items():
        try:
            if kind == "sklearn":
                svm, sc = model
                p = svm.predict_proba(sc.transform(x_norm.reshape(1,-1)))[0][1]
            elif name == "1DCNN":
                p = float(model.predict(prepare_1dcnn_input(x_norm.reshape(1,-1,1)),
                                        verbose=0)[0][0])
            else:
                p = float(model.predict(x_norm.reshape(1,-1), verbose=0)[0][0])
            probs[name] = round(float(p), 4)
        except Exception as e:
            st.warning(f"{name}: {e}")
            probs[name] = None
    return probs


# ── Today's games logic ───────────────────────────────────────────────────────
BR_ABBR_MAP = {
    "SD":"SDP","SF":"SFG","KC":"KCR","TB":"TBR","WSH":"WSN",
    "CHW":"CHW","AZ":"ARI","ATH":"OAK","CWS":"CHW",
}

def resolve(abbr):
    return BR_ABBR_MAP.get(abbr.upper(), abbr.upper())


def get_team_vec(team_abbr, avgs_df, feat_cols, is_home):
    row = avgs_df[avgs_df["team_abbr"] == team_abbr]
    if len(row) == 0:
        can = BR_ABBR_MAP.get(team_abbr)
        if can:
            row = avgs_df[avgs_df["team_abbr"] == can]
    if len(row) == 0:
        return None
    data_cols = [c for c in feat_cols if c != "X1"]
    vec = [float(row.iloc[0].get(c, 0)) for c in data_cols]
    vec.append(1.0 if is_home else 0.0)
    return np.array(vec, dtype="float32")


def predict_game(home, away, avgs_df, models, scaler, feat_cols):
    col_min, col_max = scaler
    x_h = get_team_vec(home, avgs_df, feat_cols, True)
    x_a = get_team_vec(away, avgs_df, feat_cols, False)
    if x_h is None or x_a is None:
        return None, None, "Missing season data"

    ph = run_predict(models, minmax_norm(x_h, col_min, col_max))
    pa = run_predict(models, minmax_norm(x_a, col_min, col_max))

    combined = {}
    for m in ph:
        if ph[m] is not None and pa[m] is not None:
            combined[m] = round((ph[m] + (1 - pa[m])) / 2, 4)

    if not combined:
        return None, None, "Prediction failed"

    ens   = round(float(np.mean(list(combined.values()))), 4)
    fav   = home if ens >= 0.5 else away
    prob  = ens if ens >= 0.5 else 1 - ens
    label = f"{TEAM_NAMES.get(fav, fav)}  ({prob*100:.1f}%)"
    return ens, combined, label


def render_today_tab(ds_num, use_fs, season=2026):
    st.markdown(f"### ⚾ Today's Games — {date.today().strftime('%A, %B %d, %Y')}")

    avgs_df = load_season_avgs(season)
    if avgs_df is None:
        st.warning(f"No {season} season averages found. "
                   f"Run `python data_collection.py --years {season}` first.")
        return

    models, missing = load_models_cached(ds_num, use_fs)
    if not models:
        st.error("No models loaded. Run `python train.py` first.")
        return

    scaler = load_scaler(ds_num, use_fs)
    if scaler[0] is None:
        st.error("No scaler file found. Run `python preprocessing.py` first.")
        return

    feat_cols = DS1_COLS if ds_num == 1 else DS2_COLS

    # Today's scheduled games (hardcoded from live data, refreshed on reload)
    # In production this would call today_games.fetch_todays_schedule()
    from today_games import fetch_todays_schedule
    today_str = date.today().strftime("%Y-%m-%d")
    with st.spinner("Fetching today's schedule from Baseball-Reference…"):
        games = fetch_todays_schedule(today_str)

    if not games:
        st.info("No games scheduled today, or schedule could not be fetched.")
        return

    st.info(
        f"Using **{season} season-to-date averages** as pre-game feature proxies.  "
        f"Predictions improve as more games are played."
    )

    model_names = list(models.keys())
    rows = []
    for game in games:
        home = resolve(game["home_team"])
        away = resolve(game["away_team"])
        ens, per_model, label = predict_game(
            home, away, avgs_df, models, scaler, feat_cols)
        rows.append({
            "away": away, "home": home,
            "ens": ens, "label": label,
            "status": game["status"],
            **(per_model or {}),
        })

    # Render game cards
    for r in rows:
        away_name = TEAM_NAMES.get(r["away"], r["away"])
        home_name = TEAM_NAMES.get(r["home"], r["home"])
        ens       = r["ens"]

        with st.container():
            c1, c2, c3 = st.columns([3, 4, 2])
            with c1:
                st.markdown(f"**{away_name}**  *@*  **{home_name}**")
                st.caption(f"Status: {r['status']}")
            with c2:
                if ens is not None:
                    home_pct = int(ens * 100)
                    away_pct = 100 - home_pct
                    bar_html = (
                        f'<div style="display:flex;height:22px;border-radius:4px;overflow:hidden;'
                        f'font-size:.75rem;font-weight:600">'
                        f'<div style="width:{away_pct}%;background:#c8102e;color:white;'
                        f'display:flex;align-items:center;justify-content:center">'
                        f'{away_name.split()[-1]} {away_pct}%</div>'
                        f'<div style="width:{home_pct}%;background:#003087;color:white;'
                        f'display:flex;align-items:center;justify-content:center">'
                        f'{home_name.split()[-1]} {home_pct}%</div>'
                        f'</div>'
                    )
                    st.markdown(bar_html, unsafe_allow_html=True)
                    # Per-model breakdown
                    per = "  ·  ".join(
                        f"{m}: {int(r[m]*100)}%" for m in model_names if r.get(m) is not None)
                    st.caption(per)
                else:
                    st.caption(r["label"])
            with c3:
                if ens is not None:
                    fav  = r["home"] if ens >= 0.5 else r["away"]
                    prob = ens if ens >= 0.5 else 1 - ens
                    cls  = "win-label"
                    st.markdown(
                        f'<div class="{cls}">{TEAM_NAMES.get(fav, fav).split()[-1]}'
                        f'<br><span style="font-size:.9rem">{prob*100:.1f}%</span></div>',
                        unsafe_allow_html=True)
            st.divider()

    # Download button
    df_out = pd.DataFrame([{
        "Away": TEAM_NAMES.get(r["away"], r["away"]),
        "Home": TEAM_NAMES.get(r["home"], r["home"]),
        "Predicted Winner": r["label"],
        "Home Win Prob (Ensemble)": f"{r['ens']*100:.1f}%" if r["ens"] else "N/A",
    } for r in rows])
    st.download_button("⬇️ Download predictions CSV",
                       df_out.to_csv(index=False).encode(),
                       file_name=f"mlb_predictions_{today_str}.csv",
                       mime="text/csv")


# ── Manual predict form ───────────────────────────────────────────────────────
def gauge(prob, label):
    c = "#1a7a1a" if prob >= 0.5 else "#b00020"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        number={"suffix":"%","font":{"size":28,"color":c}},
        title={"text":label,"font":{"size":14,"color":"#003087"}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":c,"thickness":.35},
               "steps":[{"range":[0,50],"color":"#ffe5e5"},{"range":[50,100],"color":"#e5f5e5"}],
               "threshold":{"line":{"color":"#003087","width":3},"thickness":.8,"value":50}},
    ))
    fig.update_layout(height=210, margin=dict(t=40,b=10,l=20,r=20))
    return fig


def input_form(ds_num):
    with st.form("pred_form"):
        st.markdown('<div class="section-title">⚾ Batting Stats</div>',unsafe_allow_html=True)
        bat_vals = {}
        for i,(k,(lbl,lo,hi,d)) in enumerate(BATTING_FIELDS.items()):
            col = st.columns(4)[i%4]; isf = isinstance(lo,float)
            bat_vals[k] = col.number_input(lbl, min_value=type(lo)(lo),
                max_value=type(hi)(hi), value=type(d)(d),
                step=0.001 if isf else 1, format="%.3f" if isf else "%d", key=k)
        fields = SP_FIELDS if ds_num == 1 else ALL_P_FIELDS
        lbl_h  = "🎯 Starting Pitcher Stats" if ds_num == 1 else "🎯 All Pitcher Stats"
        st.markdown(f'<div class="section-title">{lbl_h}</div>', unsafe_allow_html=True)
        p_vals = {}
        for i,(k,(lbl,lo,hi,d)) in enumerate(fields.items()):
            col = st.columns(4)[i%4]; isf = isinstance(lo,float)
            p_vals[k] = col.number_input(lbl, min_value=type(lo)(lo),
                max_value=type(hi)(hi), value=type(d)(d),
                step=0.1 if isf else 1, format="%.1f" if isf else "%d", key=k)
        st.markdown('<div class="section-title">🏟️ Context</div>', unsafe_allow_html=True)
        loc = st.radio("Team", ["Home","Away"], horizontal=True)
        all_vals = {**bat_vals, **p_vals, "X1": 1 if loc=="Home" else 0}
        return st.form_submit_button("⚾ Predict", use_container_width=True), all_vals


def display_results(probs):
    st.markdown("---")
    st.markdown("### 📊 Results")
    cols = st.columns(len(probs))
    for col,(name,prob) in zip(cols, probs.items()):
        if prob is None: col.warning(f"{name}: failed"); continue
        col.plotly_chart(gauge(prob, name), use_container_width=True, key=f"g_{name}")
    for col,(name,prob) in zip(cols, probs.items()):
        if prob is None: continue
        v   = "WIN ✅" if prob>=.5 else "LOSS ❌"
        cls = "win-label" if prob>=.5 else "loss-label"
        col.markdown(
            f'<div class="metric-card"><div class="{cls}">{v}</div>'
            f'<div style="color:#555;margin-top:6px">{prob*100:.1f}% win prob</div>'
            f'<div style="color:#999;font-size:.8rem">Conf: {abs(prob-.5)*200:.0f}%</div></div>',
            unsafe_allow_html=True)
    valid = [p for p in probs.values() if p]
    if valid:
        avg = np.mean(valid)
        st.info(f"**Ensemble:** {'WIN' if avg>=.5 else 'LOSS'} — {avg*100:.1f}%")


# ── Results tab ───────────────────────────────────────────────────────────────
def show_results_tab(summary, split_info):
    if split_info:
        sr, ps, trs = split_info.get("split_ratio",.5), split_info.get("partial_season",2026), split_info.get("train_seasons",[])
        st.info(
            f"**Training:** {trs} (full) + first {sr*100:.0f}% of {ps}  |  "
            f"**Holdout test:** last {(1-sr)*100:.0f}% of {ps}  |  "
            f"Train: {split_info.get('n_train_rows','?')} rows  •  "
            f"Test: {split_info.get('n_test_rows','?')} rows"
        )
    if not summary:
        st.info("No results yet. Run `python train.py`.")
        return
    rows = []
    for ds_key, ds_vals in summary.items():
        for exp_key, exp_val in ds_vals.items():
            model = exp_key.rsplit("_",1)[0].upper()
            fs    = "Yes" if exp_key.endswith("_fs") else "No"
            rows.append({"Dataset":ds_key.upper(),"Model":model,
                "Feature Selection":fs,
                "CV Acc (%)":round(exp_val.get("cv_avg_accuracy",0)*100,2),
                "Holdout Acc (%)":round(exp_val.get("test_accuracy",0)*100,2)
                                  if exp_val.get("test_accuracy") else None})
    df_r = pd.DataFrame(rows)
    fig  = px.bar(df_r.melt(id_vars=["Dataset","Model","Feature Selection"],
                   value_vars=["CV Acc (%)","Holdout Acc (%)"],
                   var_name="Metric",value_name="Accuracy"),
        x="Model",y="Accuracy",color="Metric",barmode="group",facet_col="Dataset",
        color_discrete_map={"CV Acc (%)":"#003087","Holdout Acc (%)":"#c8102e"},text_auto=".2f")
    fig.update_layout(height=360, yaxis_range=[80,100])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_r, use_container_width=True, hide_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/"
        "Major_League_Baseball_logo.svg/200px-Major_League_Baseball_logo.svg.png", width=140)
    st.sidebar.title("⚙️ Settings")
    ds_num = st.sidebar.radio("Dataset",[1,2],
        format_func=lambda x: f"DS{x} – {'SP only' if x==1 else 'All Pitchers'}")
    use_fs = st.sidebar.checkbox("Feature-selected models (ANN + SVM)")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Training:** 2023 + 2024 + 2025 (full)  \n"
        "**+ partial:** 2026 early season  \n"
        "**Holdout test:** 2026 late season")

    st.markdown('<div class="main-header">⚾ MLB Win Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Trained on 2023–2026 · 1DCNN · ANN · SVM · '
        'Pre-game predictions using season averages</div>', unsafe_allow_html=True)

    models, missing = load_models_cached(ds_num, use_fs)
    if models:
        st.success(f"✅ Loaded: {', '.join(models.keys())}")
    if missing:
        st.warning(f"⚠️ Not found – run `python train.py`: {', '.join(missing)}")

    tab_today, tab_manual, tab_results, tab_fi, tab_guide = st.tabs(
        ["📅 Today's Games", "🎯 Manual Predict", "📈 Model Results",
         "📊 Feature Importance", "📖 How to Use"])

    with tab_today:
        render_today_tab(ds_num, use_fs, season=2026)

    with tab_manual:
        submitted, all_vals = input_form(ds_num)
        if submitted:
            if not models:
                st.error("No models loaded.")
            else:
                cols  = DS1_COLS if ds_num == 1 else DS2_COLS
                x_raw = np.array([[all_vals.get(c,0) for c in cols]], dtype="float32")
                mn, mx = load_scaler(ds_num, use_fs)
                x_n   = minmax_norm(x_raw, mn, mx) if mn is not None else x_raw
                display_results(run_predict(models, x_n.squeeze()))

    with tab_results:
        show_results_tab(load_training_summary(), load_split_summary())

    with tab_fi:
        path = os.path.join(DATA_DIR, f"feature_importance_ds{ds_num}.csv")
        if os.path.exists(path):
            fi  = pd.read_csv(path)
            fig = px.bar(fi.head(20), x="weight", y="label", orientation="h",
                color="selected",color_discrete_map={True:"#003087",False:"#aab4cc"},
                title=f"Dataset {ds_num} – ReliefF weights (top 20)")
            fig.update_layout(height=480, yaxis={"autorange":"reversed"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `python preprocessing.py` to generate feature importances.")

    with tab_guide:
        st.markdown("""
## How to Use

### 1. Collect data (2023 – 2026)
```bash
python data_collection.py              # scrapes all four seasons
python data_collection.py --years 2026 --resume  # update 2026 only
```

### 2. Preprocess + split
```bash
python preprocessing.py
```
- **Training:** 2023 + 2024 + 2025 (full) + first 50% of 2026 (chronological)
- **Holdout test:** last 50% of 2026

### 3. Train
```bash
python train.py
```

### 4. Predict today's games from the terminal
```bash
python today_games.py
python today_games.py --date 2026-04-13
```

### 5. Launch this app
```bash
streamlit run app.py
```

---
### Pre-game prediction note
Because box-score stats don't exist before first pitch, the **Today's Games** tab uses
each team's **season-to-date averages** as feature proxies. This is standard practice
for pre-game win-probability models. Predictions become more reliable as the season progresses.
        """)


if __name__ == "__main__":
    main()
