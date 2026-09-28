"""Solar Review — reviewer workbench (Streamlit).

CRM-style ops tool for triaging multi-source solar project records:
dashboard, review queue with saved views, project evidence,
AI review notes, reviewer decisions, audit log, model performance.

Light/Dark toggle in the top bar. Every custom color comes from the
active palette below, so text contrast is guaranteed in both modes
(dark ink on light surfaces, light ink on dark surfaces).

Data is a documented mix of synthetic canonical records and public
benchmarks (NAB, DeepMatcher, HoloClean, FEBRL4). Not production data.
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "models.yaml")
DECISIONS_PATH = os.path.join(OUTPUT_DIR, "review_decisions.json")

DEFAULT_THRESHOLD_AUTO = 0.90
DEFAULT_THRESHOLD_LOW = 0.30
MODEL_VERSION = "v1.0"

PALETTES = {
    "Light": {
        "page_bg": "#f4f5f6",
        "card_bg": "#ffffff",
        "sidebar_bg": "#ffffff",
        "ink": "#16191d",
        "muted": "#5f6368",
        "border": "#e1e4e8",
        "accent": "#1a7f5a",
        "accent_ink": "#ffffff",
        "input_bg": "#ffffff",
        "pill": {
            "green": ("#e6f4ea", "#137333"),
            "amber": ("#fef7e0", "#965500"),
            "gray": ("#eef0f2", "#4b5563"),
            "red": ("#fce8e6", "#b3261e"),
            "blue": ("#e8f0fe", "#175cd3"),
        },
    },
    "Dark": {
        "page_bg": "#0f1417",
        "card_bg": "#171e22",
        "sidebar_bg": "#12181c",
        "ink": "#f1f3f4",
        "muted": "#9aa0a6",
        "border": "#2e353b",
        "accent": "#35b779",
        "accent_ink": "#06281c",
        "input_bg": "#1d2429",
        "pill": {
            "green": ("#12361f", "#7bdea6"),
            "amber": ("#3d2c00", "#ffd666"),
            "gray": ("#262c31", "#c4c9cd"),
            "red": ("#431407", "#ff9d97"),
            "blue": ("#10294f", "#9ec1ff"),
        },
    },
}


def load_config():
    cfg = {
        "thresholds": {"auto_match": DEFAULT_THRESHOLD_AUTO, "manual_review_low": DEFAULT_THRESHOLD_LOW},
        "model": {"version": MODEL_VERSION},
        "features": [
            "customer_similarity", "installer_similarity", "address_similarity",
            "city_match", "state_match", "capacity_diff", "status_match",
        ],
    }
    try:
        import yaml  # optional; falls back to defaults above
        with open(CONFIG_PATH) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg["thresholds"].update(file_cfg.get("thresholds", {}))
        cfg["model"].update(file_cfg.get("model", {}))
        if file_cfg.get("features"):
            cfg["features"] = file_cfg["features"]
    except Exception:
        pass
    return cfg


CFG = load_config()

st.set_page_config(
    page_title="Board",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- theme state: sync native Streamlit chrome with our palette ----
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Light"
try:
    st._config.set_option("theme.base", st.session_state.ui_theme.lower())
except Exception:
    pass
P = PALETTES[st.session_state.ui_theme]

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 0.8rem; max-width: 1320px; }}
html, body, [data-testid="stAppViewContainer"] {{
    background: {P["page_bg"]}; color: {P["ink"]};
}}
section[data-testid="stSidebar"] {{ background: {P["sidebar_bg"]}; border-right: 1px solid {P["border"]}; }}
h1 {{ font-size: 1.25rem !important; font-weight: 650 !important; color: {P["ink"]} !important; }}
h2 {{ font-size: 1.0rem !important; font-weight: 620 !important; margin-top: 1.1rem; color: {P["ink"]} !important; }}
h3 {{ font-size: 0.92rem !important; font-weight: 620 !important; color: {P["ink"]} !important; }}
p, li, span, div {{ color: {P["ink"]}; }}
.small-note {{ color: {P["muted"]} !important; font-size: 0.82rem; }}
.topbar-anchor {{ display: none; }}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .topbar-anchor),
div[data-testid="stVerticalBlock"]:has(> .topbar-anchor) {{
    background: {P["card_bg"]}; border: 1px solid {P["border"]}; border-radius: 10px;
    padding: 10px 16px; margin-bottom: 14px;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .topbar-anchor) div[data-testid="column"],
div[data-testid="stVerticalBlock"]:has(> .topbar-anchor) div[data-testid="column"] {{
    align-self: center;
}}
.brand {{ font-weight: 700; font-size: 1.02rem; color: {P["ink"]}; white-space: nowrap; }}
.envbadge {{ display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
             background: {P["accent"]}; color: {P["accent_ink"]}; border-radius: 4px;
             padding: 2px 8px; margin-left: 8px; vertical-align: middle; }}
.metric-card {{ background: {P["card_bg"]}; border: 1px solid {P["border"]}; border-radius: 8px;
                padding: 10px 14px; }}
.metric-card .lbl {{ color: {P["muted"]}; font-size: 0.78rem; }}
.metric-card .val {{ font-size: 1.35rem; font-weight: 650; color: {P["ink"]}; }}
.metric-card .sub {{ color: {P["muted"]}; font-size: 0.78rem; }}
.pill {{ display: inline-block; padding: 1px 9px; border-radius: 999px;
         font-size: 0.75rem; font-weight: 600; white-space: nowrap; }}
.crm-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
.crm-table th {{ text-align: left; color: {P["muted"]}; font-weight: 600; font-size: 0.76rem;
                 text-transform: uppercase; letter-spacing: 0.03em;
                 border-bottom: 1px solid {P["border"]}; padding: 7px 10px; }}
.crm-table td {{ border-bottom: 1px solid {P["border"]}; padding: 7px 10px; color: {P["ink"]}; }}
.crm-table tr:hover td {{ background: {P["page_bg"]}; }}
.panel {{ background: {P["card_bg"]}; border: 1px solid {P["border"]}; border-radius: 10px; padding: 14px 16px; }}
</style>
""",
    unsafe_allow_html=True,
)

PILL_CSS = "".join(
    f".pill-{k} {{ background: {bg}; color: {fg}; border: 1px solid {bg}; }} "
    for k, (bg, fg) in P["pill"].items()
)
st.markdown(f"<style>{PILL_CSS}</style>", unsafe_allow_html=True)

# Native Streamlit widgets follow theme.base, which can lag behind our
# toggle — so pin them to the active palette directly. Every rule below
# uses palette ink on palette surfaces: readable in both modes.
WIDGET_CSS = f"""
button[data-testid="stBaseButton-secondary"] {{
    background: {P["card_bg"]} !important; color: {P["ink"]} !important;
    border: 1px solid {P["border"]} !important;
}}
button[data-testid="stBaseButton-secondary"]:hover {{
    border-color: {P["accent"]} !important; color: {P["accent"]} !important;
}}
button[data-testid="stBaseButton-primary"] {{
    background: {P["accent"]} !important; color: {P["accent_ink"]} !important;
    border: 1px solid {P["accent"]} !important;
}}
div[data-testid="stSegmentedControl"] button {{
    color: {P["ink"]} !important; background: transparent !important;
}}
div[data-testid="stSegmentedControl"] button:hover {{
    background: {P["card_bg"]} !important;
}}
div[data-testid="stTextInput"] input {{
    background: {P["input_bg"]} !important; color: {P["ink"]} !important;
    border: 1px solid {P["border"]} !important; caret-color: {P["ink"]} !important;
}}
div[data-testid="stTextInput"] input::placeholder {{ color: {P["muted"]} !important; }}
div[data-testid="stTextArea"] textarea {{
    background: {P["input_bg"]} !important; color: {P["ink"]} !important;
    border: 1px solid {P["border"]} !important; caret-color: {P["ink"]} !important;
}}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
    background-color: {P["input_bg"]} !important; border-color: {P["border"]} !important;
}}
div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] span {{
    color: {P["ink"]} !important;
}}
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg,
div[data-testid="stMultiSelect"] div[data-baseweb="select"] svg {{
    fill: {P["muted"]} !important;
}}
ul[data-baseweb="menu"] {{ background-color: {P["card_bg"]} !important; }}
ul[data-baseweb="menu"] li span {{ color: {P["ink"]} !important; }}
span[data-baseweb="tag"] {{
    background: {P["card_bg"]} !important; border: 1px solid {P["border"]} !important;
}}
span[data-baseweb="tag"] span {{ color: {P["ink"]} !important; }}
"""
st.markdown(f"<style>{WIDGET_CSS}</style>", unsafe_allow_html=True)


def pill(text, kind="gray"):
    bg_fg = P["pill"].get(kind, P["pill"]["gray"])
    return (
        f'<span class="pill" style="background:{bg_fg[0]};color:{bg_fg[1]};'
        f'border:1px solid {bg_fg[0]}">{text}</span>'
    )


ROUTE_PILL = {"AUTO_MATCH": "green", "MANUAL_REVIEW": "amber", "NON_MATCH": "gray"}
CLASS_PILL = {
    "ENTITY_MISMATCH": "red",
    "NUMERIC_CONFLICT": "amber",
    "STATUS_CONFLICT": "amber",
    "FORMATTING_DIFFERENCE": "blue",
    "POTENTIAL_DUPLICATE": "blue",
    "DOCUMENT_MISMATCH": "amber",
    "MISSING_DATA": "amber",
    "OTHER": "gray",
}


def route_of(prob):
    hi = CFG["thresholds"]["auto_match"]
    lo = CFG["thresholds"]["manual_review_low"]
    if prob >= hi:
        return "AUTO_MATCH"
    if prob < lo:
        return "NON_MATCH"
    return "MANUAL_REVIEW"


@st.cache_data
def load_tables():
    master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_projects.csv"))
    pairs = pd.read_csv(os.path.join(OUTPUT_DIR, "entity_pairs.csv"))
    features = pd.read_csv(os.path.join(OUTPUT_DIR, "pairwise_features.csv")).fillna(0)
    ai = pd.read_csv(os.path.join(OUTPUT_DIR, "ai_reviewer_output.csv"))
    canonical = pd.read_csv(os.path.join(OUTPUT_DIR, "canonical_source_records.csv"))
    sources = {}
    try:
        sources["crm"] = pd.read_csv(os.path.join(OUTPUT_DIR, "crm_export_corrupted.csv"))
        sources["erp"] = pd.read_csv(os.path.join(OUTPUT_DIR, "erp_export_corrupted.csv"))
        sources["document"] = pd.read_csv(os.path.join(OUTPUT_DIR, "document_metadata_corrupted.csv"))
    except Exception:
        pass
    try:
        sources["partner"] = pd.read_excel(os.path.join(OUTPUT_DIR, "partner_export_corrupted.xlsx"))
    except Exception:
        pass
    return master, pairs, features, ai, canonical, sources


@st.cache_resource
def load_match_model(_features):
    cols = CFG["features"]
    X = _features[cols].fillna(0)
    y = _features["label"].values
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model, cols


def read_decisions():
    if not os.path.exists(DECISIONS_PATH):
        return []
    with open(DECISIONS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def queue_frame(pairs, features, ai, model, cols):
    merged = pairs.merge(features, on="pair_id", how="left")
    merged = merged.merge(ai[["pair_id", "classification", "summary", "clarification"]], on="pair_id", how="left")
    merged["match_probability"] = model.predict_proba(merged[cols].fillna(0))[:, 1].round(4)
    merged["route"] = merged["match_probability"].apply(route_of)
    merged["uncertainty"] = (0.5 - (merged["match_probability"] - 0.5).abs()).round(4)
    return merged


def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def crm_table(df):
    st.markdown(df.to_html(escape=False, index=False, classes="crm-table"), unsafe_allow_html=True)


def goto(page):
    st.session_state.nav = page
    st.rerun()


# ---------------- pages ----------------

def page_dashboard(master, queue, canonical):
    st.markdown("## Pipeline overview")
    st.markdown('<p class="small-note">Live figures from the current data snapshot. Select a stage to open it filtered in the queue.</p>', unsafe_allow_html=True)

    route_counts = queue["route"].value_counts()
    auto = int(route_counts.get("AUTO_MATCH", 0))
    manual = int(route_counts.get("MANUAL_REVIEW", 0))
    non = int(route_counts.get("NON_MATCH", 0))
    total = len(queue)
    decisions = read_decisions()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Projects in scope", f"{len(master):,}", f"{len(canonical):,} source records")
    with c2:
        metric_card("Needs review", f"{manual:,}", f"{manual / max(total, 1):.1%} of pairs")
    with c3:
        metric_card("Auto-matchable", f"{auto:,}", f"{auto / max(total, 1):.1%} coverage")
    with c4:
        metric_card("Decisions recorded", f"{len(decisions):,}", f"{non:,} closed as non-match")

    st.markdown("### By stage")
    rc = queue["route"].value_counts().rename_axis("stage").reset_index(name="pairs")
    rc["share"] = (rc["pairs"] / max(total, 1)).map(lambda x: f"{x:.1%}")
    crm_table(rc)

    st.markdown("### By issue type")
    cc = queue["classification"].value_counts().rename_axis("issue_type").reset_index(name="pairs")
    cc["issue_type"] = cc["issue_type"].apply(lambda c: pill(c, CLASS_PILL.get(c, "gray")))
    crm_table(cc)

    st.markdown("### Records per source")
    sc = canonical["source_system"].value_counts().rename_axis("source").reset_index(name="records")
    crm_table(sc)


VIEWS = {
    "Needs review": {"route": ["MANUAL_REVIEW"], "classification": []},
    "High risk": {"route": ["MANUAL_REVIEW"], "classification": ["ENTITY_MISMATCH"]},
    "Auto-matchable": {"route": ["AUTO_MATCH"], "classification": []},
    "Non-match": {"route": ["NON_MATCH"], "classification": []},
    "All pairs": {"route": [], "classification": []},
}


def page_queue(queue):
    st.markdown("## Review queue")
    st.markdown('<p class="small-note">Ground-truth labels are hidden here by design — route on model evidence, then decide.</p>', unsafe_allow_html=True)

    view = st.segmented_control("Saved view", list(VIEWS.keys()), default="Needs review")
    spec = VIEWS[view]

    f1, f2, f3 = st.columns(3)
    with f1:
        routes = st.multiselect("Routing", ["AUTO_MATCH", "MANUAL_REVIEW", "NON_MATCH"],
                                default=spec["route"])
    with f2:
        classes = st.multiselect("Issue type", sorted(queue["classification"].dropna().unique()),
                                 default=spec["classification"])
    with f3:
        sort = st.selectbox("Sort by", ["Most uncertain first", "Highest P(match)", "Lowest P(match)"])

    q = queue if not routes else queue[queue["route"].isin(routes)]
    if classes:
        q = q[q["classification"].isin(classes)]
    if sort == "Most uncertain first":
        q = q.sort_values("uncertainty", ascending=False)
    elif sort == "Highest P(match)":
        q = q.sort_values("match_probability", ascending=False)
    else:
        q = q.sort_values("match_probability")

    limit = st.slider("Rows shown", 25, 300, 75)
    view_df = q.head(limit).copy()
    view_df["route"] = view_df["route"].apply(lambda r: pill(r, ROUTE_PILL.get(r, "gray")))
    view_df["classification"] = view_df["classification"].apply(lambda c: pill(c, CLASS_PILL.get(c, "gray")))
    st.write(f"Showing {len(view_df)} of {len(q)} matching pairs")
    crm_table(view_df[["pair_id", "source_a", "source_b", "route", "classification",
                        "match_probability", "customer_similarity",
                        "installer_similarity", "address_similarity"]])

    st.download_button("Export filtered view (CSV)",
                       q.to_csv(index=False).encode(),
                       f"review_queue_{view.lower().replace(' ', '_')}.csv", "text/csv")


def source_rows_for(sources, project_id):
    rows = {}

    def pick(df, id_col, pid):
        hit = df[df[id_col].astype(str) == str(pid)]
        return hit.iloc[0] if len(hit) else None

    crm = sources.get("crm")
    if crm is not None:
        r = pick(crm, "project_id", project_id)
        if r is not None:
            rows["CRM"] = {"Customer": r.get("customer_name"), "Installer": r.get("installer"),
                           "Capacity kW": r.get("system_size_kw"), "Address": r.get("address"),
                           "Status": r.get("status")}
    erp = sources.get("erp")
    if erp is not None:
        r = pick(erp, "project_reference", project_id)
        if r is not None:
            rows["ERP"] = {"Customer": r.get("customer"), "Installer": "—",
                           "Capacity kW": r.get("installed_capacity"), "Address": "—",
                           "Status": r.get("record_status")}
    partner = sources.get("partner")
    if partner is not None:
        r = pick(partner, "external_reference", project_id)
        if r is not None:
            rows["Partner"] = {"Customer": "—", "Installer": r.get("installer_name"),
                               "Capacity kW": r.get("capacity"), "Address": r.get("site_address"),
                               "Status": r.get("completion_status")}
    doc = sources.get("document")
    if doc is not None:
        r = pick(doc, "project_reference", project_id)
        if r is not None:
            rows["Document"] = {"Customer": "—", "Installer": "—",
                                "Capacity kW": "—", "Address": "—",
                                "Status": f"{r.get('signing_status')} · {r.get('document_name')}"}
    return rows


def page_project(master, canonical, sources, queue):
    st.markdown("## Project evidence")
    default_pid = st.session_state.get("selected_project") or master["project_id"].iloc[0]
    pid = st.selectbox("Project", master["project_id"].tolist(),
                       index=master["project_id"].tolist().index(default_pid)
                       if default_pid in master["project_id"].tolist() else 0)
    st.session_state.selected_project = pid
    proj = master[master["project_id"] == pid].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Customer", str(proj["customer_name"]), f"{proj['city']}, {proj['state']}")
    with c2:
        metric_card("Installer", str(proj["installer"]), str(proj["address"]))
    with c3:
        metric_card("System size", f"{proj['system_size_kw']} kW", f"Installed {proj['install_date']}")
    with c4:
        metric_card("Status", str(proj["status"]), f"Updated {proj['updated_at']}")

    st.markdown("### Source comparison")
    rows = source_rows_for(sources, pid)
    if rows:
        comp = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "source"})
        crm_table(comp)
    else:
        st.info("No source rows found for this project.")

    st.markdown("### Pairs involving this project")
    rel = queue[(queue["record_a_id"].astype(str) == str(pid)) | (queue["record_b_id"].astype(str) == str(pid))]
    if len(rel):
        show = rel[["pair_id", "source_a", "source_b", "route", "classification", "match_probability"]].copy()
        show["route"] = show["route"].apply(lambda r: pill(r, ROUTE_PILL.get(r, "gray")))
        show["classification"] = show["classification"].apply(lambda c: pill(c, CLASS_PILL.get(c, "gray")))
        crm_table(show)
    else:
        st.info("No pairs reference this project.")


def page_ai_review(queue):
    st.markdown("## AI review notes")
    st.markdown('<p class="small-note">Rule-based explainer over model evidence. It organises facts for the reviewer — it does not decide truth.</p>', unsafe_allow_html=True)

    default_pid = st.session_state.get("selected_pair") or queue["pair_id"].iloc[0]
    pid = st.selectbox("Pair", queue["pair_id"].tolist(),
                       index=queue["pair_id"].tolist().index(default_pid)
                       if default_pid in queue["pair_id"].tolist() else 0)
    st.session_state.selected_pair = pid
    row = queue[queue["pair_id"] == pid].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### {pill(row['classification'], CLASS_PILL.get(row['classification'], 'gray'))} &nbsp; {pill(row['route'], ROUTE_PILL.get(row['route'], 'gray'))}",
                    unsafe_allow_html=True)
        st.markdown("**Summary**")
        st.write(row["summary"])
        st.markdown("**Field evidence**")
        ev = pd.DataFrame([
            {"field": "Customer", "similarity": round(float(row["customer_similarity"]), 3)},
            {"field": "Installer", "similarity": round(float(row["installer_similarity"]), 3)},
            {"field": "Address", "similarity": round(float(row["address_similarity"]), 3)},
            {"field": "City match", "similarity": int(row["city_match"])},
            {"field": "State match", "similarity": int(row["state_match"])},
            {"field": "Capacity diff (kW)", "similarity": round(float(row["capacity_diff"]), 2)},
            {"field": "Status match", "similarity": int(row["status_match"])},
        ])
        crm_table(ev)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Model**")
        st.write(f"P(match) = {row['match_probability']:.4f}")
        st.progress(min(max(float(row["match_probability"]), 0.0), 1.0))
        st.write(f"Routing: {row['route']}")
        st.markdown("**Suggested next step**")
        st.write(row["clarification"])
        st.markdown("**Uncertainty**")
        st.write("Explainer output is deterministic given the same evidence. "
                 "Low field similarities with a mid-range probability mean: verify manually, do not auto-approve.")
        if st.button("Decide on this pair"):
            st.session_state.selected_pair = pid
            goto("Record a decision")
        st.markdown("</div>", unsafe_allow_html=True)


def page_decision(queue):
    st.markdown("## Record a decision")
    shortlist = queue[queue["route"] == "MANUAL_REVIEW"]["pair_id"].tolist() or queue["pair_id"].tolist()
    default_pid = st.session_state.get("selected_pair")
    pid = st.selectbox("Pair", shortlist,
                       index=shortlist.index(default_pid) if default_pid in shortlist else 0)
    row = queue[queue["pair_id"] == pid].iloc[0]
    st.markdown(f"{pill(row['route'], ROUTE_PILL.get(row['route'], 'gray'))} &nbsp; "
                f"{pill(row['classification'], CLASS_PILL.get(row['classification'], 'gray'))} &nbsp; "
                f"P(match) = {row['match_probability']:.4f}", unsafe_allow_html=True)
    st.write(row["summary"])

    reviewer = st.text_input("Reviewer ID", value="reviewer_001")
    decision = st.radio("Decision", ["MATCH", "NON_MATCH", "CLARIFICATION"], horizontal=True)
    reason = st.text_area("Reason (required)", placeholder="e.g. capacity differs 2.0 kW across CRM and Partner; requested installer confirmation")
    comment = st.text_area("Comment (optional)")

    if st.button("Submit decision", type="primary"):
        if not reviewer.strip():
            st.error("Reviewer ID is required.")
        elif not reason.strip():
            st.error("A reason is required — decisions without reasons are not auditable.")
        else:
            record = {
                "pair_id": pid,
                "reviewer_id": reviewer.strip(),
                "decision": decision,
                "reason": reason.strip(),
                "comment": comment.strip(),
                "model_version": CFG["model"]["version"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with open(DECISIONS_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            st.success(f"Decision recorded for {pid}.")


def page_audit():
    st.markdown("## Audit log")
    logs = read_decisions()
    if not logs:
        st.info("No decisions recorded yet. Use “Record a decision” — every entry lands here with reviewer, reason, and model version.")
        return
    df = pd.DataFrame(logs)
    order = ["timestamp", "pair_id", "reviewer_id", "decision", "reason", "comment", "model_version"]
    df = df[[c for c in order if c in df.columns]]
    st.write(f"{len(df)} decisions on record")
    crm_table(df)
    st.download_button("Download audit CSV", df.to_csv(index=False), "audit_log.csv", "text/csv")


def page_model_performance():
    st.markdown("## Model performance")
    st.markdown('<p class="small-note">Persisted evaluation outputs. Metrics are reproduced from the notebooks, not typed by hand.</p>', unsafe_allow_html=True)

    st.markdown("### Entity matching — real dirty data (DeepMatcher)")
    p = os.path.join(OUTPUT_DIR, "dirty_er_ml.csv")
    if os.path.exists(p):
        crm_table(pd.read_csv(p))
        st.markdown('<p class="small-note">dirty_test vs clean_test on the same pair IDs: dirt degrades recall, TF-IDF cosine carries most of the signal.</p>', unsafe_allow_html=True)
    else:
        st.info("dirty_er_ml.csv not found — run notebook 06.")

    st.markdown("### Anomaly detection — labeled NAB series")
    p = os.path.join(OUTPUT_DIR, "nab_labeled_eval.csv")
    if os.path.exists(p):
        crm_table(pd.read_csv(p))
        st.markdown('<p class="small-note">Point-level precision/recall against hand-labeled windows. Real labeled anomalies are hard — rolling MAD underperforms, Isolation Forest is steadier.</p>', unsafe_allow_html=True)
    else:
        st.info("nab_labeled_eval.csv not found — run notebook 07.")

    st.markdown("### Limitations")
    p = os.path.join(REPORT_DIR, "phase9_final_evaluation.json")
    try:
        with open(p) as f:
            lims = json.load(f).get("limitations", [])
        for item in lims:
            st.write(f"• {item}")
    except Exception:
        st.info("Final evaluation report not found — run notebook 10.")


PAGES = ["Dashboard", "Review queue", "Project evidence", "AI review notes",
         "Record a decision", "Audit log", "Model performance"]
MODULES = {
    "Workspace": ["Dashboard", "Review queue"],
    "Records": ["Project evidence", "AI review notes"],
    "Activity": ["Record a decision", "Audit log"],
    "Insights": ["Model performance"],
}


def main():
    master, pairs, features, ai, canonical, sources = load_tables()
    model, cols = load_match_model(features)
    queue = queue_frame(pairs, features, ai, model, cols)

    # ---- top bar: brand, global search, theme toggle (one bar) ----
    with st.container():
        st.markdown('<div class="topbar-anchor"></div>', unsafe_allow_html=True)
        tb1, tb2, tb3 = st.columns([2.5, 4.2, 2.3])
        with tb1:
            st.markdown('<span class="brand">◧ Board</span>', unsafe_allow_html=True)
        with tb2:
            q = st.text_input("Search", placeholder="Cari pair ID (PAIR-…) atau project ID (PRJ-…)…",
                              label_visibility="collapsed")
        with tb3:
            theme = st.segmented_control("Tampilan", ["Light", "Dark"],
                                         default=st.session_state.ui_theme,
                                         label_visibility="collapsed")
    if theme != st.session_state.ui_theme:
        st.session_state.ui_theme = theme
        try:
            st._config.set_option("theme.base", theme.lower())
        except Exception:
            pass
        st.rerun()

    if q:
        q = q.strip()
        if q in pairs["pair_id"].tolist():
            st.session_state.selected_pair = q
            goto("AI review notes")
        elif q in master["project_id"].tolist():
            st.session_state.selected_project = q
            goto("Project evidence")
        else:
            st.warning(f"Tidak ditemukan: {q}")

    # ---- sidebar module navigation ----
    if "nav" not in st.session_state:
        st.session_state.nav = "Dashboard"
    with st.sidebar:
        st.markdown("### Board")
        st.markdown('<p class="small-note">Reviewer workbench</p>', unsafe_allow_html=True)
        st.markdown("Modules")
        for module, items in MODULES.items():
            st.markdown(f'<p class="small-note">{module}</p>', unsafe_allow_html=True)
            for item in items:
                if st.button(item, key=f"nav_{item}",
                             type="primary" if st.session_state.nav == item else "secondary",
                             use_container_width=True):
                    goto(item)
        st.markdown("---")
        st.markdown(f'<p class="small-note">Model {CFG["model"]["version"]} · '
                    f'auto ≥ {CFG["thresholds"]["auto_match"]} · '
                    f'review ≥ {CFG["thresholds"]["manual_review_low"]}</p>', unsafe_allow_html=True)

    page = st.session_state.nav
    if page == "Dashboard":
        page_dashboard(master, queue, canonical)
    elif page == "Review queue":
        page_queue(queue)
    elif page == "Project evidence":
        page_project(master, canonical, sources, queue)
    elif page == "AI review notes":
        page_ai_review(queue)
    elif page == "Record a decision":
        page_decision(queue)
    elif page == "Audit log":
        page_audit()
    elif page == "Model performance":
        page_model_performance()

    st.markdown("---")
    st.markdown('<p class="small-note">Independent portfolio project. Synthetic canonical records plus public benchmarks (Solar Power Generation, FEBRL4, NAB, DeepMatcher, HoloClean, UCI Donation). Thresholds: config/models.yaml.</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
