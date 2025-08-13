import pandas as pd
import streamlit as st

"""
Suggestions module with ranking:
- Ranks suggestions by (coverage gap boost + tactic weight + log-source fit).
- Keeps UI simple and compatible with existing design.
"""

TACTIC_WEIGHT = {
    "initial-access": 0.95, "execution": 0.9, "persistence": 0.85, "privilege-escalation": 0.9,
    "defense-evasion": 0.8, "credential-access": 0.9, "discovery": 0.6, "lateral-movement": 0.9,
    "collection": 0.7, "command-and-control": 0.8, "exfiltration": 0.85, "impact": 0.9
}

def split_csv_cell(val: str) -> list:
    return [p.strip() for p in str(val or "").split(",") if str(p).strip()]

def tactic_weight_from_csv(tactics_csv: str) -> float:
    if not tactics_csv:
        return 0.5
    vals = []
    for t in str(tactics_csv).split(","):
        vals.append(TACTIC_WEIGHT.get(t.strip().lower(), 0.6))
    return max(vals) if vals else 0.6

def render_suggestions(mapped_csv_file, alerts_library_csv):
    st.set_page_config(page_title="Suggestions", layout="wide")
    st.title("Suggestions (Ranked)")

    if not mapped_csv_file or not alerts_library_csv:
        st.info("Upload both mapped_usecases.csv and an Alerts Library CSV.")
        return

    mapped = pd.read_csv(mapped_csv_file)
    lib = pd.read_csv(alerts_library_csv)

    # Derive user's log sources from mapped CSV
    user_sources = set()
    col = "Log Source" if "Log Source" in mapped.columns else None
    if col:
        for v in mapped[col].fillna("").astype(str):
            for s in split_csv_cell(v):
                if s and s.upper() != "N/A":
                    user_sources.add(s)

    st.caption(f"Detected log sources: {', '.join(sorted(user_sources)) or 'None'}")

    # Coverage gap computation
    seen_ids = set()
    if "TechniqueIDs" in mapped.columns:
        for val in mapped["TechniqueIDs"].fillna("").astype(str):
            for tid in split_csv_cell(val):
                seen_ids.add(tid)

    # Expect library to have Use Case, Technique, Log Source, (optional) Tactics
    use_col = "Use Case" if "Use Case" in lib.columns else lib.columns[0]
    tech_col = "Technique" if "Technique" in lib.columns else None
    log_col  = "Log Source" if "Log Source" in lib.columns else None
    tac_col  = "Tactics" if "Tactics" in lib.columns else None

    rows = []
    for _, r in lib.iterrows():
        log_fit = 1.0 if not user_sources else (1.0 if (set(split_csv_cell(r.get(log_col,""))) & user_sources) else 0.4)
        technique = str(r.get(tech_col, ""))
        tid = technique.split(" ")[0].strip() if technique.startswith("T") else ""
        gap = 1.0 if (tid and tid not in seen_ids) else 0.5
        tac_w = tactic_weight_from_csv(str(r.get(tac_col, "")))
        score = round(100 * (0.5*gap + 0.3*tac_w + 0.2*log_fit), 2)

        rows.append({
            "Use Case": r.get(use_col, ""),
            "Technique": technique,
            "Log Source": r.get(log_col, ""),
            "Tactics": r.get(tac_col, ""),
            "Relevance Score": score
        })

    out = pd.DataFrame(rows).drop_duplicates().sort_values("Relevance Score", ascending=False).reset_index(drop=True)
    if out.empty:
        st.info("No suggestions found for the detected log sources.")
        return

    st.subheader("Suggested Use Cases (sorted by priority)")
    st.dataframe(out, use_container_width=True)
    st.download_button("Download Suggestions CSV", data=out.to_csv(index=False).encode("utf-8"),
                       file_name="suggested_alerts.csv", mime="text/csv")

if __name__ == "__main__":
    st.write("Run render_suggestions() from a parent app.")
