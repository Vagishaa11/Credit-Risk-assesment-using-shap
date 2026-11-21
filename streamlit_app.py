import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from streamlit_option_menu import option_menu
from pathlib import Path
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss, precision_score, recall_score, f1_score)

# ---------------------------------------------------------------- setup
st.set_page_config(page_title="Credit Risk Studio", page_icon="🏦", layout="wide")

INK, PANEL, LINE = "#0D1B2A", "#14263A", "#24405C"
TEXT, MUTED = "#E6EDF5", "#8FA3B8"
TEAL, AMBER, CORAL, INDIGO = "#2EC4B6", "#FFB547", "#FF6B6B", "#7C8CFF"
MODEL_COLORS = [INDIGO, TEAL, AMBER]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Manrope',sans-serif; }}
.stApp {{ background:{INK}; color:{TEXT}; }}
header[data-testid="stHeader"], #MainMenu, footer {{ visibility:hidden; height:0; }}
.block-container {{ padding-top:1.2rem; max-width:1250px; }}
h1,h2,h3 {{ letter-spacing:-0.02em; color:{TEXT}; }}

.hero {{ background:{PANEL}; border:1px solid {LINE}; border-left:6px solid {TEAL};
         border-radius:16px; padding:2rem 2.2rem; margin:1rem 0 1.4rem; }}
.hero h1 {{ font-size:2.3rem; margin:0 0 .4rem; font-weight:800; }}
.hero p {{ color:{MUTED}; font-size:1.05rem; max-width:760px; margin:0; line-height:1.6; }}

.kpi {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:1rem 1.2rem; height:100%; }}
.kpi .l {{ color:{MUTED}; font-size:.85rem; font-weight:600; }}
.kpi .v {{ font-size:1.9rem; font-weight:800; margin:.15rem 0; }}
.kpi .s {{ color:{MUTED}; font-size:.78rem; }}

.panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:1.2rem 1.4rem; }}
.panel h4 {{ margin:0 0 .5rem; }}
.panel p, .panel li {{ color:{MUTED}; line-height:1.65; font-size:.95rem; }}
.step {{ display:flex; gap:.9rem; padding:.7rem 0; border-bottom:1px solid {LINE}; }}
.step:last-child {{ border-bottom:none; }}
.step b {{ color:{TEAL}; min-width:1.4rem; }}

.badge {{ display:inline-block; padding:.5rem 1.2rem; border-radius:999px; font-weight:800; font-size:1.2rem; }}
.champ {{ background:rgba(46,196,182,.12); border:1px solid {TEAL}; color:{TEAL};
          padding:.7rem 1rem; border-radius:12px; font-weight:600; margin:.6rem 0 1rem; }}

/* radio buttons become pill buttons */
div[role="radiogroup"] {{ gap:.5rem; flex-wrap:wrap; }}
div[role="radiogroup"] > label {{ background:{PANEL}; border:1px solid {LINE}; padding:.4rem 1.1rem;
                                 border-radius:10px; cursor:pointer; }}
div[role="radiogroup"] > label > div:first-child {{ display:none; }}
div[role="radiogroup"] > label:has(input:checked) {{ background:{TEAL}; border-color:{TEAL}; }}
div[role="radiogroup"] > label:has(input:checked) * {{ color:{INK} !important; font-weight:700; }}

/* buttons */
.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
    background:{TEAL}; color:{INK}; font-weight:800; border:none; border-radius:10px;
    padding:.6rem 1.4rem; width:100%; }}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{ background:#5ad8cc; color:{INK}; }}
div[data-testid="stForm"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:1.2rem; }}
button[data-baseweb="tab"] {{ font-weight:600; color:{MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{TEAL}; }}

/* make every piece of text readable on the dark background */
.stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stWidgetLabel"] *, [data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"] *, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] * {{ color:{TEXT}; }}
[data-testid="stCaptionContainer"] *, .panel p, .panel li, .kpi .l, .kpi .s {{ color:{MUTED}; }}

/* inputs, dropdowns and their menus */
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {{
    background-color:{PANEL} !important; color:{TEXT} !important; border-color:{LINE} !important; }}
[data-baseweb="select"] *, [data-baseweb="input"] * {{ color:{TEXT} !important; }}
[data-baseweb="popover"] *, ul[role="listbox"] li {{ background-color:{PANEL}; color:{TEXT} !important; }}
ul[role="listbox"] li:hover {{ background-color:{LINE}; }}
[data-testid="stNumberInput"] button {{ background-color:{PANEL}; color:{TEXT}; }}
[data-testid="stAlert"] * {{ color:{TEXT}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def class1(sv):
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def dense(a):
    return a.toarray() if hasattr(a, "toarray") else a


def style(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=45, b=10),
                      font=dict(family="Manrope", color=TEXT), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


def kpi(col, label, value, sub="", color=TEXT):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div>'
                 f'<div class="v" style="color:{color}">{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def contrib_chart(values, names, title, top=10):
    s = pd.Series(values, index=names)
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[CORAL if v > 0 else TEAL for v in s.values]))
    fig.update_layout(title=title)
    fig.update_xaxes(title="Pushes toward low risk  ◂  SHAP value  ▸  pushes toward default")
    return style(fig, 420)


BASE = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "XGBoost (calibrated)": "xgb_model.pkl",
    "Random Forest (calibrated)": "rf_model.pkl",
}
REQUIRED = ["credit_risk_dataset.csv", "credit_risk_model.pkl", "best_threshold.pkl",
            "thresholds.pkl", *MODEL_FILES.values()]


@st.cache_data(show_spinner="Preparing data...")
def prepare_data():
    """Same cleaning and split as the notebook, so the test set is identical."""
    df = pd.read_csv(BASE / "credit_risk_dataset.csv").drop_duplicates()
    df = df[df["person_age"].between(18, 100)]
    df = df[df["person_emp_length"].isna() |
            ((df["person_emp_length"] <= df["person_age"]) & (df["person_emp_length"] <= 60))]
    df = df[(df["person_income"] > 0) & (df["loan_amnt"] > 0) & (df["person_income"] <= 1_000_000)]
    df = df[df["loan_int_rate"].isna() | df["loan_int_rate"].between(0, 40)]
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]
    X, y = df.drop("loan_status", axis=1), df["loan_status"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    eda = dict(n_rows=len(df), default_rate=float(y.mean()),
               by_grade=df.groupby("loan_grade")["loan_status"].mean().to_dict(),
               by_intent=df.groupby("loan_intent")["loan_status"].mean().to_dict(),
               by_home=df.groupby("person_home_ownership")["loan_status"].mean().to_dict())
    return X_test, y_test, eda


def fitted_pipeline(model):
    """Return the fitted preprocessing + tree pipeline inside a calibrated model."""
    folds = getattr(model, "calibrated_classifiers_", None)
    if not folds:
        return None
    est = getattr(folds[0], "estimator", None)
    return est if est is not None else getattr(folds[0], "base_estimator", None)


@st.cache_resource(show_spinner="Loading models and computing metrics...")
def load_everything():
    X_test, y_test, eda = prepare_data()
    thresholds = joblib.load(BASE / "thresholds.pkl")
    models = {n: joblib.load(BASE / f) for n, f in MODEL_FILES.items()}

    perf, rows = {}, []
    for n, m in models.items():
        thr = float(thresholds[n])
        prob = m.predict_proba(X_test)[:, 1]
        pred = (prob >= thr).astype(int)
        fpr, tpr, _ = roc_curve(y_test, prob)
        prec, rec, _ = precision_recall_curve(y_test, prob)
        frac, mean = calibration_curve(y_test, prob, n_bins=10)
        perf[n] = dict(threshold=thr, fpr=fpr, tpr=tpr, precision=prec, recall=rec,
                       frac_pos=frac, mean_pred=mean, cm=confusion_matrix(y_test, pred))
        rows.append({"Model": n, "ROC-AUC": roc_auc_score(y_test, prob),
                     "PR-AUC": average_precision_score(y_test, prob),
                     "Brier (lower=better)": brier_score_loss(y_test, prob),
                     "Precision": precision_score(y_test, pred, zero_division=0),
                     "Recall": recall_score(y_test, pred, zero_division=0),
                     "F1": f1_score(y_test, pred, zero_division=0), "Threshold": thr})
    comp = pd.DataFrame(rows).set_index("Model").round(4)
    champ = comp["PR-AUC"].idxmax()                      # same rule as the notebook

    # Production model = exactly what your API loads
    prod_model = joblib.load(BASE / "credit_risk_model.pkl")
    prod_thr = float(joblib.load(BASE / "best_threshold.pkl"))

    # SHAP needs a tree model: use the champion if it is tree-based, otherwise XGBoost
    pipe = fitted_pipeline(models[champ])
    shap_name = champ
    if pipe is None:
        shap_name = "XGBoost (calibrated)"
        pipe = fitted_pipeline(models[shap_name])
    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifer" if "classifer" in pipe.named_steps else "classifier"]

    X_s = X_test.sample(min(800, len(X_test)), random_state=42)
    Xt = pd.DataFrame(dense(pre.transform(X_s)),
                      columns=[c.split("__")[-1] for c in pre.get_feature_names_out()], index=X_s.index)
    explainer = shap.TreeExplainer(clf)
    sv = class1(explainer.shap_values(Xt, check_additivity=False))

    data = dict(comparison=comp, champion_name=champ, perf=perf, eda=eda,
                shap_values=sv, shap_X=Xt, shap_model=type(clf).__name__ + f" ({shap_name})")
    return data, prod_model, prod_thr, pipe, explainer


_missing = [f for f in REQUIRED if not (BASE / f).exists()]
if _missing:
    st.error(f"Missing files in {BASE}: {', '.join(_missing)}")
    st.info("Run the 'save thresholds' cell from the notebook, then copy all .pkl files and "
            "credit_risk_dataset.csv next to streamlit_app.py.")
    st.stop()

data, champion_model, champion_thr, shap_pipe, explainer = load_everything()
comp = data["comparison"]
champ = data["champion_name"]
perf = data["perf"]
eda = data["eda"]
names = list(perf.keys())
color_of = {n: MODEL_COLORS[i % 3] for i, n in enumerate(names)}

# ---------------------------------------------------------------- navigation
page = option_menu(
    None, ["Overview", "Model performance", "Comparison", "Explainability", "Predict"],
    icons=["house", "speedometer2", "bar-chart-line", "lightbulb", "calculator"],
    orientation="horizontal",
    styles={
        "container": {"background-color": PANEL, "border": f"1px solid {LINE}", "border-radius": "14px", "padding": "6px"},
        "icon": {"color": MUTED, "font-size": "16px"},
        "nav-link": {"font-family": "Manrope", "font-size": "15px", "font-weight": "600", "color": MUTED,
                     "border-radius": "10px", "--hover-color": LINE},
        "nav-link-selected": {"background-color": TEAL, "color": INK, "font-weight": "800"},
    },
)

# ================================================================ OVERVIEW
if page == "Overview":
    st.markdown("""<div class="hero"><h1>Credit Risk Studio</h1>
    <p>A machine learning system that estimates how likely a loan applicant is to default,
    so lenders can approve safer loans, price risk fairly and explain every decision.</p></div>""",
                unsafe_allow_html=True)

    c = st.columns(4)
    kpi(c[0], "Loans analysed", f"{eda['n_rows']:,}", "after cleaning and validation")
    kpi(c[1], "Historical default rate", f"{eda['default_rate']:.1%}", "share of loans that defaulted", CORAL)
    kpi(c[2], "Models compared", "3", "Logistic Regression, XGBoost, Random Forest")
    kpi(c[3], "Production model", champ.replace(" (calibrated)", ""), f"decision threshold {champion_thr:.3f}", TEAL)

    st.write("")
    a, b = st.columns([1.1, 1])
    with a:
        st.markdown("""<div class="panel"><h4>The business problem</h4>
        <p>Every loan carries the risk that the borrower will not repay. Approving too many risky
        applicants causes losses. Rejecting too many safe applicants loses revenue and customers.</p>
        <p>This project predicts the probability of default from the applicant's income, job history,
        credit history, home ownership, loan size, loan grade and interest rate. The probability is
        converted into a <b>High Risk</b> or <b>Low Risk</b> decision using a threshold tuned for the best
        balance between precision and recall.</p></div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""<div class="panel"><h4>How the system works</h4>
        <div class="step"><b>1</b><span>Clean the data: remove duplicates, impossible ages and income outliers</span></div>
        <div class="step"><b>2</b><span>Train three models with class weights for the imbalanced target</span></div>
        <div class="step"><b>3</b><span>Tune, calibrate probabilities and choose the decision threshold</span></div>
        <div class="step"><b>4</b><span>Explain predictions with SHAP and serve the best model</span></div></div>""",
                    unsafe_allow_html=True)

    st.write("")
    st.subheader("Where defaults happen")
    group = st.radio("Group loans by", ["Loan grade", "Loan intent", "Home ownership"], horizontal=True)
    src = {"Loan grade": eda["by_grade"], "Loan intent": eda["by_intent"], "Home ownership": eda["by_home"]}[group]
    s = pd.Series(src).sort_index()
    fig = px.bar(x=s.index, y=s.values * 100, labels={"x": group, "y": "Default rate (%)"},
                 color=s.values, color_continuous_scale=[TEAL, AMBER, CORAL])
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style(fig, 340))

# ================================================================ PERFORMANCE
elif page == "Model performance":
    st.subheader("Model performance")
    st.caption("Evaluated on the held-out test set, using each model's tuned threshold.")
    m = st.radio("Choose a model", names, horizontal=True)
    p, row = perf[m], comp.loc[m]
    if m == champ:
        st.markdown('<div class="champ">Selected as the production model</div>', unsafe_allow_html=True)

    c = st.columns(6)
    for col, k in zip(c, ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Brier (lower=better)"]):
        kpi(col, k.replace(" (lower=better)", " (lower is better)"), f"{row[k]:.3f}",
            color=color_of[m] if k != "Brier (lower=better)" else TEXT)
    st.write("")

    t1, t2, t3 = st.tabs(["ROC and precision-recall", "Confusion matrix", "Calibration"])
    with t1:
        a, b = st.columns(2)
        f = go.Figure(go.Scatter(x=p["fpr"], y=p["tpr"], fill="tozeroy", line=dict(color=color_of[m], width=3)))
        f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=MUTED, dash="dash"))
        f.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        a.plotly_chart(style(f))
        f = go.Figure(go.Scatter(x=p["recall"], y=p["precision"], fill="tozeroy", line=dict(color=color_of[m], width=3)))
        f.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
        b.plotly_chart(style(f))
    with t2:
        cm = p["cm"]
        norm = st.toggle("Show as percentage of each actual class", value=False)
        z = cm / cm.sum(axis=1, keepdims=True) * 100 if norm else cm
        f = px.imshow(z, text_auto=".1f" if norm else "d", aspect="auto",
                      x=["Predicted low risk", "Predicted high risk"], y=["Actually safe", "Actually defaulted"],
                      color_continuous_scale=[[0, PANEL], [1, color_of[m]]])
        f.update_coloraxes(showscale=False)
        a, b = st.columns([1.3, 1])
        a.plotly_chart(style(f, 400))
        tn, fp, fn, tp = cm.ravel()
        b.markdown(f"""<div class="panel"><h4>Reading the matrix</h4>
        <p><b style="color:{TEAL}">{tp:,}</b> defaulters caught correctly<br>
        <b style="color:{CORAL}">{fn:,}</b> defaulters missed (costly losses)<br>
        <b style="color:{AMBER}">{fp:,}</b> safe applicants wrongly flagged (lost business)<br>
        <b>{tn:,}</b> safe applicants approved correctly</p></div>""", unsafe_allow_html=True)
    with t3:
        f = go.Figure()
        f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=MUTED, dash="dash"))
        f.add_trace(go.Scatter(x=p["mean_pred"], y=p["frac_pos"], mode="lines+markers",
                               line=dict(color=color_of[m], width=3), name=m))
        f.update_layout(title="Calibration: a 30% score should mean about 30% defaults",
                        xaxis_title="Predicted probability", yaxis_title="Actual default rate")
        st.plotly_chart(style(f, 420))

# ================================================================ COMPARISON
elif page == "Comparison":
    st.subheader("Model comparison")
    st.markdown(f'<div class="champ">Winner by PR-AUC: {champ}</div>', unsafe_allow_html=True)

    higher = [c for c in comp.columns if c not in ("Brier (lower=better)", "Threshold")]
    st.dataframe(comp.style.set_properties(**{"color": TEXT, "background-color": PANEL}).format("{:.4f}").highlight_max(subset=higher, color="#155e55")
                 .highlight_min(subset=["Brier (lower=better)"], color="#155e55"))
    st.caption("Highlighted cells are the best value in each column. For Brier score, lower is better.")

    chosen = st.multiselect("Metrics to compare", higher, default=["PR-AUC", "Precision", "Recall", "F1"])
    if chosen:
        long = comp[chosen].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        f = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                   color_discrete_map=color_of)
        st.plotly_chart(style(f, 400))

    a, b = st.columns(2)
    radar_keys = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
    f = go.Figure()
    for n in names:
        v = [comp.loc[n, k] for k in radar_keys]
        f.add_trace(go.Scatterpolar(r=v + v[:1], theta=radar_keys + radar_keys[:1], name=n,
                                    line=dict(color=color_of[n], width=3)))
    f.update_layout(title="Overall profile", polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(gridcolor=LINE), angularaxis=dict(gridcolor=LINE)))
    a.plotly_chart(style(f, 420))

    f = go.Figure()
    for n in names:
        f.add_trace(go.Scatter(x=perf[n]["fpr"], y=perf[n]["tpr"], name=n, line=dict(color=color_of[n], width=3)))
    f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=MUTED, dash="dash"))
    f.update_layout(title="ROC curves together", xaxis_title="False positive rate", yaxis_title="True positive rate")
    b.plotly_chart(style(f, 420))

# ================================================================ EXPLAINABILITY
elif page == "Explainability":
    st.subheader("Explainability with SHAP")
    st.caption(f"Computed on 800 test applicants using the tuned {data['shap_model']}. "
               "Red pushes toward default, teal pushes toward safe.")
    sv, X = data["shap_values"], data["shap_X"]
    view = st.radio("View", ["Global: what matters overall", "Local: why one applicant got their score"],
                    horizontal=True)

    if view.startswith("Global"):
        top = st.slider("Number of features", 5, 20, 12)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values().tail(top)
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=TEAL))
        f.update_layout(title="Average impact of each feature", xaxis_title="Mean |SHAP value|")
        a, b = st.columns(2)
        a.plotly_chart(style(f, 480))

        plt.close("all")
        shap.summary_plot(sv, X, show=False, max_display=top, plot_size=(6, 5.2))
        fig = plt.gcf()
        fig.patch.set_facecolor(INK)
        for ax in fig.axes:
            ax.set_facecolor(INK)
            ax.tick_params(colors=TEXT)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            for sp in ax.spines.values():
                sp.set_color(LINE)
        b.pyplot(fig)
        st.markdown("""<div class="panel"><p>Each dot is one applicant. Dots on the right increased the
        default risk, dots on the left lowered it. Red dots are high feature values, blue dots are low ones.</p></div>""",
                    unsafe_allow_html=True)
    else:
        if "row" not in st.session_state:
            st.session_state.row = 5
        c = st.columns(4)
        if c[0].button("Highest risk"):
            st.session_state.row = int(sv.sum(axis=1).argmax())
        if c[1].button("Lowest risk"):
            st.session_state.row = int(sv.sum(axis=1).argmin())
        if c[2].button("Random applicant"):
            st.session_state.row = int(np.random.randint(0, len(X)))
        i = st.session_state.row
        a, b = st.columns([1.5, 1])
        a.plotly_chart(contrib_chart(sv[i], X.columns, f"Why applicant #{i} scored this way"))
        raw = X.iloc[i]
        numeric = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
                   "loan_percent_income", "cb_person_cred_hist_length"]
        shown = raw[[c for c in X.columns if c in numeric or raw[c] == 1]]   # hide one-hot columns that are 0
        b.markdown('<div class="panel"><h4>Applicant profile</h4></div>', unsafe_allow_html=True)
        b.dataframe(shown.rename("value").to_frame().style.set_properties(**{"color": TEXT, "background-color": PANEL}), height=360)

# ================================================================ PREDICT
else:
    st.subheader("Loan risk prediction")
    st.markdown(f'<div class="champ">Scoring with {champ} · decision threshold {champion_thr:.3f}</div>',
                unsafe_allow_html=True)

    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 100, 28)
        income = c1.number_input("Annual income", 1, 1_000_000, 55_000, step=1000)
        home = c1.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        emp = c2.number_input("Years employed", 0.0, 60.0, 4.0, step=0.5)
        intent = c2.selectbox("Loan purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                                               "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        grade = c2.selectbox("Loan grade", list("ABCDEFG"), index=1)
        amnt = c3.number_input("Loan amount", 1, 1_000_000, 10_000, step=500)
        rate = c3.number_input("Interest rate (%)", 0.1, 40.0, 11.0, step=0.1)
        dflt = c3.selectbox("Previous default on file", ["N", "Y"])
        hist = c3.number_input("Credit history (years)", 0, 80, 4)
        go_btn = st.form_submit_button("Assess risk")

    if go_btn:
        if emp > age or hist > age:
            st.error("Years employed and credit history cannot be greater than the applicant's age.")
            st.stop()
        row = dict(person_age=age, person_income=float(income), person_home_ownership=home,
                   person_emp_length=emp, loan_intent=intent, loan_grade=grade, loan_amnt=float(amnt),
                   loan_int_rate=rate, cb_person_default_on_file=dflt, cb_person_cred_hist_length=hist)
        row["loan_percent_income"] = row["loan_amnt"] / row["person_income"]   # same formula as the API
        df_in = pd.DataFrame([row])

        prob = float(champion_model.predict_proba(df_in)[:, 1][0])
        high = prob >= champion_thr

        pre = shap_pipe.named_steps["preprocessor"]
        Xr = pd.DataFrame(dense(pre.transform(df_in)),
                          columns=[c.split("__")[-1] for c in pre.get_feature_names_out()])
        contrib = class1(explainer.shap_values(Xr))[0]

        a, b = st.columns([1, 1.2])
        with a:
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=44)),
                title=dict(text="Probability of default"),
                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=CORAL if high else TEAL),
                           bgcolor=PANEL, borderwidth=0,
                           steps=[dict(range=[0, champion_thr * 100], color="#17384a"),
                                  dict(range=[champion_thr * 100, 100], color="#3b2a3a")],
                           threshold=dict(line=dict(color="white", width=4), value=champion_thr * 100))))
            st.plotly_chart(style(g, 320))
            colr, txt = (CORAL, "High Risk") if high else (TEAL, "Low Risk")
            st.markdown(f'<div style="text-align:center"><span class="badge" style="background:{colr}22;'
                        f'border:1px solid {colr};color:{colr}">{txt}</span></div>', unsafe_allow_html=True)
            st.caption(f"The white line marks the decision threshold ({champion_thr:.1%}). "
                       f"Loan is {row['loan_percent_income']:.1%} of annual income.")
        with b:
            st.plotly_chart(contrib_chart(contrib, Xr.columns, "What drove this applicant's score", top=8))
        if champ.startswith("Logistic"):
            st.caption("Drivers are explained with the tuned XGBoost model, since the logistic model is not tree-based.")