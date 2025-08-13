import pandas as pd
import streamlit as st

"""
Suggestions module:
- Uses the mapped CSV and the user's log sources to suggest additional library alerts/use cases.
- Robust splitting and normalization of comma-separated values.
"""

def split_csv_cell(val: str) -> list:
    return [p.strip() for p in str(val or "").split(",") if str(p).strip()]

def render_suggestions(mapped_csv_file, alerts_library_csv):
    st.set_page_config(page_title="Suggestions", layout="wide")
    st.title("Suggestions (Log-Source Aware)")

    if not mapped_csv_file or not alerts_library_csv:
        st.info("Upload both mapped_usecases.csv and an Alerts Library CSV.")
        return

    mapped = pd.read_csv(mapped_csv_file)
    lib = pd.read_csv(alerts_library_csv)

    # Derive user's log sources from mapped CSV (or the raw input if available)
    user_sources = set()
    col = "Log Source" if "Log Source" in mapped.columns else None
    if col:
        for v in mapped[col].fillna("").astype(str):
            for s in split_csv_cell(v):
                if s and s.upper() != "N/A":
                    user_sources.add(s)

    st.caption(f"Detected log sources: {', '.join(sorted(user_sources)) or 'None'}")

    # Filter library by user's log sources
    use_col = "Log Source" if "Log Source" in lib.columns else None
    tech_col = "Technique"  if "Technique"  in lib.columns else None
    title_col= "Use Case"   if "Use Case"   in lib.columns else lib.columns[0]

    rows = []
    if use_col and tech_col:
        for _, r in lib.iterrows():
            lib_sources = set(split_csv_cell(r[use_col]))
            if not user_sources or (lib_sources & user_sources):
                rows.append({
                    "Use Case": r.get(title_col, ""),
                    "Technique": r.get(tech_col, ""),
                    "Log Source": ", ".join(sorted(lib_sources)) if lib_sources else ""
                })

    out = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    if out.empty:
        st.info("No suggestions found for the detected log sources.")
        return

    st.subheader("Suggested Alerts / Use Cases")
    st.dataframe(out, use_container_width=True)
    st.download_button("Download Suggestions CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="suggested_alerts.csv", mime="text/csv")

if __name__ == "__main__":
    st.write("Run render_suggestions() from a parent app.")
