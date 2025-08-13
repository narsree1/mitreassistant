import pandas as pd
import streamlit as st

TOTAL_TECHNIQUES = 211  # per user instruction; update when ATT&CK version changes

def compute_coverage(mapped_df: pd.DataFrame) -> float:
    """
    Coverage % = (unique TechniqueIDs present / TOTAL_TECHNIQUES) * 100
    TechniqueIDs expected as comma-separated string column 'TechniqueIDs'.
    """
    tech_ids = set()
    if "TechniqueIDs" in mapped_df.columns:
        for val in mapped_df["TechniqueIDs"].fillna("").astype(str):
            for tid in val.split(","):
                tid = tid.strip()
                if tid:
                    tech_ids.add(tid)

    covered = len(tech_ids)
    coverage = round((covered / TOTAL_TECHNIQUES) * 100, 2) if TOTAL_TECHNIQUES else 0.0
    return coverage

def render_analytics(mapped_csv_file):
    st.title("Analytics")
    if not mapped_csv_file:
        st.info("Upload the mapped_usecases.csv produced by the app.")
        return

    df = pd.read_csv(mapped_csv_file)

    # Metrics
    total_usecases = len(df)
    coverage_pct = compute_coverage(df)
    mapped_rows = (df["Match Source"] != "N/A").sum() if "Match Source" in df.columns else 0
    lib_rows = (df["Match Source"] == "Library").sum() if "Match Source" in df.columns else 0
    model_rows = (df["Match Source"] == "Model").sum() if "Match Source" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Security Use Cases", total_usecases)
    col2.metric("Framework Coverage", f"{coverage_pct}%")
    col3.metric("Library Matches", lib_rows)
    col4.metric("Model Matches", model_rows)

    # Show distinct techniques
    st.subheader("Mapped Techniques (unique)")
    if "TechniqueIDs" in df.columns and "Techniques" in df.columns:
        exploded = []
        for _, r in df.iterrows():
            ids = [t.strip() for t in str(r["TechniqueIDs"]).split(",") if str(t).strip()]
            disps = [t.strip() for t in str(r["Techniques"]).split(",") if str(t).strip()]
            for tid, disp in zip(ids, disps if len(disps) == len(ids) else ids):
                exploded.append({"TechniqueID": tid, "Technique": disp})
        uniq = pd.DataFrame(exploded).drop_duplicates().sort_values("TechniqueID")
        st.dataframe(uniq, use_container_width=True)

def main():
    st.set_page_config(page_title="ATT&CK Analytics", layout="wide")
    uploaded = st.file_uploader("Upload mapped_usecases.csv", type=["csv"])
    render_analytics(uploaded)

if __name__ == "__main__":
    main()
