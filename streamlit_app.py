"""Judge the Model: a Streamlit app for a clinical lead deciding whether any classifier is fit for a job.

Run locally:  bash run_app.sh
Every number is computed from results/ through m11.py; nothing is typed in or retrained here.
Every table and plot is labelled "Training" or "Test".
"""

import pandas as pd
import streamlit as st

import m11

TRAIN = "Training"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]

# Plain-language name of the positive class for each job.
POSITIVE = {
    "readmission": "readmitted within 30 days",
    "diabetes": "has diabetes (doctor diagnosis or HbA1c ≥ 6.5%)",
}
JOB = {
    "readmission": "Predict, at discharge, who will be readmitted within 30 days.",
    "diabetes": "Flag adults for follow-up testing; finding people who have diabetes matters most.",
}

st.set_page_config(page_title="Judge the Model", layout="wide")


@st.cache_data
def _load(name: str, modified: float):
    return m11.load(name)


def load(name: str):
    """Read results/<name>.json, re-reading it whenever the file changes (so new results and decisions show up)."""
    path = m11.RESULTS / f"{name}.json"
    return _load(name, path.stat().st_mtime if path.exists() else 0.0)


def recommendations() -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.values()):
            st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run "
                       "`python -m analysis`, then commit and push.")
        for k in m11.DATASETS:
            if any(recs.values()):
                st.markdown(f"**{m11.DATASET_LABEL[k]}:** {recs.get(k) or '_not written yet_'}")


def matrix_frame(c: dict, key: str) -> pd.DataFrame:
    """Confusion matrix: actual outcomes as rows, predicted outcomes as columns."""
    rows = {"readmission": ("Readmitted within 30 days", "Not readmitted within 30 days"),
            "diabetes": ("Has diabetes", "Does not have diabetes")}[key]
    return pd.DataFrame(
        [[c["tp"], c["fn"]], [c["fp"], c["tn"]]],
        index=[f"Actually: {rows[0]}", f"Actually: {rows[1]}"],
        columns=["Flagged (predicted positive)", "Not flagged (predicted negative)"],
    )


def pct(v) -> str:
    return m11.fmt_pct(v)


def per100(v) -> str:
    return "undefined" if v is None else f"{v:.1f}"


def metric_rows(m: dict) -> pd.DataFrame:
    """The three headline metrics, each with its denominator in words."""
    return pd.DataFrame([
        {"Metric": "Sensitivity (recall)", "Value": pct(m["sensitivity"]),
         "Plain words": f"Of the {m['positives']:,} patients who {{pos}}, the share the model flagged."},
        {"Metric": "Specificity", "Value": pct(m["specificity"]),
         "Plain words": f"Of the {m['negatives']:,} patients who did not, the share the model did not flag."},
        {"Metric": "Precision", "Value": pct(m["precision"]),
         "Plain words": f"Of the {m['flagged']:,} patients the model flagged, the share who really {{pos}}."},
    ])


def show_metrics(m: dict, key: str) -> None:
    pos_verb = {"readmission": "were readmitted", "diabetes": "have diabetes"}[key]
    df = metric_rows(m)
    df["Plain words"] = df["Plain words"].str.replace("{pos}", pos_verb, regex=False)
    st.table(df.set_index("Metric"))


# ---- 1. Beyond Accuracy ------------------------------------------------------
def beyond_accuracy(key: str, res: dict, model: str) -> None:
    p, y = m11.preds(res, model)
    c = m11.confusion(p, y, 0.5)
    m = m11.metrics(c)
    base = m11.majority_accuracy(y)

    st.markdown(f"**Job:** {JOB[key]}  \n**Positive class:** patients who are **{POSITIVE[key]}**.")
    st.caption(f"{TEST} results · all numbers below are on the {m['n']:,} test patients, which the model never learned from.")

    a, b, c3 = st.columns(3)
    a.metric("Prevalence (Test)", pct(m["prevalence"]),
             help=f"Share of the {m['n']:,} test patients who are {POSITIVE[key]}.")
    b.metric('Accuracy of "always no" (Test)', pct(base),
             help="A baseline that predicts the most common class for every patient.")
    c3.metric(f"Accuracy of {m11.MODEL_LABEL[model]} (Test)", pct(m["accuracy"]))
    st.caption('"Always no" never flags anyone, so it is right for every patient who does not have the outcome '
               "and wrong for every patient who does. A model has to beat it, and accuracy alone cannot show whether it does so by catching positives.")

    st.subheader(f"{m11.MODEL_LABEL[model]} at threshold 0.50")
    st.caption(f"{TEST} · rows are what actually happened, columns are what the model predicted.")
    st.table(matrix_frame(c, key))
    st.caption(f"Four groups of patients: {c['tp']:,} caught (true positives), {c['fn']:,} missed (false negatives), "
               f"{c['fp']:,} false alarms (false positives), {c['tn']:,} correctly left alone (true negatives).")
    show_metrics(m, key)

    st.subheader("All five models at threshold 0.50")
    st.caption(f"{TEST} · same {m['n']:,} test patients for every model.")
    fitted = [k for k in m11.MODELS if k in res["models"]]
    cols = st.columns(len(fitted))
    rows = []
    for col, k in zip(cols, fitted):
        pk, yk = m11.preds(res, k)
        ck = m11.confusion(pk, yk, 0.5)
        mk = m11.metrics(ck)
        with col:
            st.markdown(f"**{m11.MODEL_LABEL[k]}**")
            st.caption(f"{TEST}")
            st.table(pd.DataFrame([[ck["tp"], ck["fn"]], [ck["fp"], ck["tn"]]],
                                  index=["Actual +", "Actual −"], columns=["Flagged", "Not flagged"]))
        rows.append({"Model": m11.MODEL_LABEL[k], "Accuracy": pct(mk["accuracy"]),
                     "Sensitivity": pct(mk["sensitivity"]), "Specificity": pct(mk["specificity"]),
                     "FPR": pct(mk["fpr"]), "Precision": pct(mk["precision"]), "F1": pct(mk["f1"])})
    st.caption(f"{TEST} metrics at 0.50. Undefined means the denominator is zero (for example, precision when no one is flagged).")
    st.table(pd.DataFrame(rows).set_index("Model"))
    missing = [m11.MODEL_LABEL[k] for k in m11.MODELS if k not in res["models"]]
    if missing:
        st.warning("Not trained yet: " + ", ".join(missing) + ". Run `python -m analysis`, then commit and push `results/`.")


# ---- 2. A Threshold for the Job ------------------------------------------------
def threshold(key: str, res: dict, model: str) -> None:
    p, y = m11.preds(res, model)
    st.markdown(f"**Job:** {JOB[key]}  \n**Positive class:** patients who are **{POSITIVE[key]}**.")
    t = st.slider("Decision threshold: flag a patient when the model's probability is at least this value",
                  min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"thr-{key}-{model}")
    c0, c1 = m11.confusion(p, y, 0.5), m11.confusion(p, y, t)
    m0, m1 = m11.metrics(c0), m11.metrics(c1)
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]} on {m1['n']:,} test patients. The threshold changes who is flagged, not the model.")

    left, right = st.columns(2)
    for col, label, c, m in [(left, "Threshold 0.50", c0, m0), (right, f"Slider threshold {t:.2f}", c1, m1)]:
        with col:
            st.subheader(label)
            st.caption(TEST)
            st.table(matrix_frame(c, key))
            st.metric("Flagged per 100 patients (Test)", per100(m["flagged_per_100"]))

    def delta(a, b):
        return "undefined" if a is None or b is None else f"{(b - a) * 100:+.1f} pts"

    comp = pd.DataFrame([
        {"Measure": "Sensitivity", "0.50": pct(m0["sensitivity"]), f"{t:.2f}": pct(m1["sensitivity"]), "Change": delta(m0["sensitivity"], m1["sensitivity"])},
        {"Measure": "Specificity", "0.50": pct(m0["specificity"]), f"{t:.2f}": pct(m1["specificity"]), "Change": delta(m0["specificity"], m1["specificity"])},
        {"Measure": "Precision", "0.50": pct(m0["precision"]), f"{t:.2f}": pct(m1["precision"]), "Change": delta(m0["precision"], m1["precision"])},
        {"Measure": "Missed positives (count)", "0.50": f"{c0['fn']:,}", f"{t:.2f}": f"{c1['fn']:,}", "Change": f"{c1['fn'] - c0['fn']:+,}"},
        {"Measure": "False alarms (count)", "0.50": f"{c0['fp']:,}", f"{t:.2f}": f"{c1['fp']:,}", "Change": f"{c1['fp'] - c0['fp']:+,}"},
        {"Measure": "Flagged per 100", "0.50": per100(m0["flagged_per_100"]), f"{t:.2f}": per100(m1["flagged_per_100"]),
         "Change": "undefined" if m0["flagged_per_100"] is None or m1["flagged_per_100"] is None else f"{m1['flagged_per_100'] - m0['flagged_per_100']:+.1f}"},
        {"Measure": "False alarms per 100 patients without the outcome (FPR × 100)",
         "0.50": "undefined" if m0["fpr"] is None else f"{m0['fpr'] * 100:.1f}",
         f"{t:.2f}": "undefined" if m1["fpr"] is None else f"{m1['fpr'] * 100:.1f}", "Change": delta(m0["fpr"], m1["fpr"])},
    ]).set_index("Measure")
    st.subheader("0.50 compared with the slider")
    st.caption(f"{TEST} · 0.50 is the default cut-off, not a clinical choice.")
    st.table(comp)
    show_metrics(m1, key)


# ---- 3. Training vs. Test ------------------------------------------------------
def train_vs_test(key: str, res: dict, model: str) -> None:
    n_tr, n_te = res["train"]["n"], res["test"]["n"]
    st.markdown(f"**Job:** {JOB[key]}")
    a, b = st.columns(2)
    a.metric(f"{TRAIN} patients", f"{n_tr:,}", help=f"{res['train']['n_pos']:,} are {POSITIVE[key]}. Models learn from these.")
    b.metric(f"{TEST} patients", f"{n_te:,}", help=f"{res['test']['n_pos']:,} are {POSITIVE[key]}. Models never saw these.")
    st.caption(f"Each {TRAIN} AUROC rests on {n_tr:,} patients the model learned from; each {TEST} AUROC rests on "
               f"{n_te:,} patients it has never seen ({res['test']['n_pos']:,} of them {POSITIVE[key].split(' (')[0]}).")

    s = m11.auroc_summary(res)
    rows = [{"Model": m11.MODEL_LABEL[k], f"{TRAIN} AUROC": m11.fmt_num(v["train"]), f"{TEST} AUROC": m11.fmt_num(v["test"]),
             f"Gap ({TRAIN} − {TEST})": m11.fmt_num(v["gap"])} for k, v in s.items()]
    st.subheader("AUROC on the training patients and on the test patients")
    st.caption(f"{TRAIN} and {TEST} · AUROC is the chance that a random patient with the outcome gets a higher probability than "
               "one without. It describes ranking, not accuracy, and it does not choose a threshold.")
    st.table(pd.DataFrame(rows).set_index("Model"))
    st.caption(f"A large gap means the model fits the {TRAIN} patients much better than {TEST} patients: it may have memorised them. "
               f"Expect the {TEST} AUROC on new patients, never the {TRAIN} one.")

    chart = pd.DataFrame({
        f"{TRAIN} AUROC": {m11.MODEL_LABEL[k]: v["train"] for k, v in s.items() if v["train"] is not None},
        f"{TEST} AUROC": {m11.MODEL_LABEL[k]: v["test"] for k, v in s.items() if v["test"] is not None},
    })
    st.bar_chart(chart)
    st.caption(f"{TRAIN} and {TEST} AUROC by model.")
    missing = [m11.MODEL_LABEL[k] for k in m11.MODELS if k not in res["models"]]
    if missing:
        st.warning("Not trained yet: " + ", ".join(missing) + ". Run `python -m analysis`, then commit and push `results/`.")


def priority_panel(key: str) -> None:
    """Design choice: which error matters most for this job, in the author's own words from DECISIONS.md."""
    d = load("decisions") or {}
    why = (d.get("sections", {}).get("before_results", {}) or {}).get(key, "")
    miss = {"readmission": "a patient who will be readmitted is not flagged, so gets no extra follow-up",
            "diabetes": "an adult who has diabetes is not flagged, so gets no follow-up test"}[key]
    alarm = {"readmission": "a patient who will not be readmitted is flagged, so uses extra staff time",
             "diabetes": "an adult without diabetes is flagged, so gets an extra test they did not need"}[key]
    with st.container(border=True):
        st.subheader("Which mistake matters most for this job?")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Missed positive (false negative):** {miss}.")
        c2.markdown(f"**False alarm (false positive):** {alarm}.")
        if why:
            st.markdown(f"**Our reasoning, written before seeing any results:** {why}")
        else:
            st.caption("No reasoning yet: fill in `Before results` in DECISIONS.md and run `python -m analysis`.")


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

st.title("Judge the Model")
st.caption("Which classifier would you trust for this job, at what threshold, and with what limits?")
recommendations()
section = st.sidebar.radio("Section", SECTIONS)
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
res = load(key)
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
st.header(section)
st.caption(f"{m11.DATASET_LABEL[key]}: {res['train']['n']:,} training rows ({res['train']['n_pos']:,} positive), "
           f"{res['test']['n']:,} test rows ({res['test']['n_pos']:,} positive).")
priority_panel(key)
if section == SECTIONS[2] or model in res["models"]:
    RENDER[section](key, res, model)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
