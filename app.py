import os
import json
import uuid
import time
import math
import datetime as dt
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Embedding + Reranker (non-LLM)
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_NAME    = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # classical reranker (not an LLM)

TOPK_CANDIDATES  = 15        # stage-1 candidates
TOPN_LABELS      = 3         # allow up to N labels if confident
TAU_ACCEPT       = 0.42      # accept threshold (tune on eval set)
DELTA_MARGIN     = 0.06      # winner margin over next best to accept as single-label

# Priority scoring weights (editable)
PRIORITY_WEIGHTS = {
    "confidence": 0.45,      # mapping confidence (reranker score 0..1 approx)
    "tactic": 0.20,          # tactic criticality weight
    "coverage_gap": 0.20,    # boost for techniques that are rare/uncovered
    "readiness": 0.10,       # ease signals (library, has log source)
    "recency": 0.05          # placeholder for future (e.g., trending threats)
}

# Tactic criticality (editable). Higher = implement sooner.
TACTIC_WEIGHT = {
    "initial-access": 0.95,
    "execution": 0.9,
    "persistence": 0.85,
    "privilege-escalation": 0.9,
    "defense-evasion": 0.8,
    "credential-access": 0.9,
    "discovery": 0.6,
    "lateral-movement": 0.9,
    "collection": 0.7,
    "command-and-control": 0.8,
    "exfiltration": 0.85,
    "impact": 0.9
}

@st.cache_resource(show_spinner=False)
def load_models():
    emb = SentenceTransformer(EMBED_MODEL_NAME)
    rer = CrossEncoder(RERANKER_NAME)  # lightweight cross-encoder
    return emb, rer

# -----------------------------
# MITRE ATT&CK Loading
# -----------------------------
def normalize_technique_row(tid: str, name: str) -> Tuple[str, str, str]:
    """Returns: (technique_id, technique_name, display_string) where display = 'T####[.###] - Name'"""
    tid = tid.strip()
    name = name.strip()
    display = f"{tid} - {name}"
    return tid, name, display

def load_mitre_json(mitre_json_file) -> pd.DataFrame:
    """Load techniques + sub-techniques from a MITRE JSON (Navigator export or bundle)."""
    data = json.load(mitre_json_file)

    rows = []
    for obj in data.get("techniques", data.get("objects", [])):
        tid = obj.get("technique_id") or obj.get("external_id") or obj.get("id") or ""
        name = obj.get("name") or obj.get("technique", "")
        desc = obj.get("description", "")
        raw_tac = obj.get("tactic", obj.get("tactics", []))
        if isinstance(raw_tac, list):
            tactic = ", ".join([str(t).strip().lower() for t in raw_tac])
        else:
            tactic = str(raw_tac).strip().lower()

        if tid and name:
            tid, name, display = normalize_technique_row(tid, name)
            rows.append({
                "TechniqueID": tid,
                "TechniqueName": name,
                "TechniqueDisplay": display,
                "Description": desc,
                "Tactics": tactic
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["TechniqueID"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def embed_techniques(mitre_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    emb_model, _ = load_models()
    texts = (mitre_df["TechniqueName"] + " :: " + mitre_df["Description"].fillna("")).tolist()
    mat = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    return mitre_df.copy(), mat

# -----------------------------
# Candidate Generation + Rerank
# -----------------------------
def candidate_generation(query: str, tech_emb: np.ndarray, emb_model: SentenceTransformer, topk: int = TOPK_CANDIDATES) -> List[int]:
    q = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = (tech_emb @ q)
    idx = np.argpartition(sims, -topk)[-topk:]
    return idx[np.argsort(-sims[idx])].tolist()  # sorted desc

def rerank_candidates(query: str, candidates_df: pd.DataFrame, reranker: CrossEncoder) -> List[Tuple[int, float]]:
    pairs = [(query, f"{row.TechniqueDisplay}: {row.Description}") for _, row in candidates_df.iterrows()]
    scores = reranker.predict(pairs).tolist()  # higher is better
    s_arr = np.array(scores, dtype=float)
    if len(s_arr) > 1:
        min_s, max_s = float(s_arr.min()), float(s_arr.max())
        if max_s > min_s:
            s_arr = (s_arr - min_s) / (max_s - min_s)  # normalize 0..1
    order = np.argsort(s_arr)[::-1]
    return [(candidates_df.index[i], float(s_arr[i])) for i in order]

def choose_labels(reranked: List[Tuple[int, float]], mitre_df: pd.DataFrame) -> List[Tuple[str, str, float, str]]:
    """Return list of (TechniqueID, TechniqueDisplay, conf_score, tactics_csv)."""
    if not reranked:
        return []
    results: List[Tuple[str, str, float, str]] = []
    for idx, score in reranked[:TOPN_LABELS]:
        if score >= TAU_ACCEPT:
            tid = mitre_df.at[idx, "TechniqueID"]
            disp = mitre_df.at[idx, "TechniqueDisplay"]
            tactics = mitre_df.at[idx, "Tactics"]
            results.append((tid, disp, score, tactics))
    if not results:
        return []
    if len(results) == 1 and len(reranked) > 1:
        best = results[0][2]
        next_best = reranked[1][1]
        if best - next_best < DELTA_MARGIN:
            idx2, score2 = reranked[1]
            if score2 >= TAU_ACCEPT and len(results) < TOPN_LABELS:
                tid2 = mitre_df.at[idx2, "TechniqueID"]
                disp2 = mitre_df.at[idx2, "TechniqueDisplay"]
                tactics2 = mitre_df.at[idx2, "Tactics"]
                results.append((tid2, disp2, score2, tactics2))
    return results

# -----------------------------
# Library Matching (exact/heuristic)
# -----------------------------
def library_fast_match(usecase: str, library_df: pd.DataFrame) -> List[str]:
    """Simple exact/substring match over 'Use Case' → returns list of 'T#### - Name' or names."""
    if library_df is None or library_df.empty:
        return []
    hits: List[str] = []
    title_col = "Use Case" if "Use Case" in library_df.columns else library_df.columns[0]
    tech_col  = "Technique" if "Technique" in library_df.columns else None
    for _, row in library_df.iterrows():
        title = str(row.get(title_col, "")).strip().lower()
        if title and title in usecase.lower():
            label = str(row.get(tech_col, "")).strip()
            if label:
                hits.append(label)
    hits = [h for h in dict.fromkeys(hits) if h]  # de-dup
    return hits

# -----------------------------
# Priority scoring
# -----------------------------
def tactic_weight_from_csv(tactics_csv: str) -> float:
    if not tactics_csv:
        return 0.5
    vals = []
    for t in str(tactics_csv).split(","):
        vals.append(TACTIC_WEIGHT.get(t.strip().lower(), 0.6))
    return max(vals) if vals else 0.6

def compute_row_priority(conf_scores: List[float],
                         tactics_list: List[str],
                         technique_ids: List[str],
                         freq_counts: Dict[str, int],
                         match_source: str,
                         has_log_source: bool) -> float:
    conf = max(conf_scores) if conf_scores else 0.0
    tac_w = max([tactic_weight_from_csv(t) for t in tactics_list] or [0.6])
    invs = [1.0 / float(1 + freq_counts.get(tid, 0)) for tid in technique_ids] or [1.0]
    gap = min(float(np.mean(invs)), 1.0)
    readiness = (0.6 if match_source == "Library" else 0.0) + (0.4 if has_log_source else 0.0)
    w = PRIORITY_WEIGHTS
    score = (
        w["confidence"]   * conf +
        w["tactic"]       * tac_w +
        w["coverage_gap"] * gap +
        w["readiness"]    * readiness +
        w["recency"]      * 0.5
    )
    return round(100.0 * score / sum(w.values()), 2)

# -----------------------------
# Mapping Pipeline
# -----------------------------
def map_row_to_mitre(description: str,
                     mitre_df: pd.DataFrame,
                     tech_emb: np.ndarray,
                     emb_model: SentenceTransformer,
                     reranker: CrossEncoder,
                     library_df: Optional[pd.DataFrame] = None) -> Tuple[str, str, str, List[float], List[str]]:
    """Returns: (TechniqueIDsCSV, TechniqueDisplaysCSV, MatchSource, conf_scores[], tactics_list[])"""
    desc = (description or "").strip()
    if not desc:
        return "", "", "N/A", [], []

    # 1) Library match first
    lib_hits = library_fast_match(desc, library_df)
    if lib_hits:
        ids, disps, scores, tacts = [], [], [], []
        for label in lib_hits:
            label = label.strip()
            if label.startswith("T"):
                tid = label.split(" ")[0].strip()
                m = mitre_df[mitre_df["TechniqueID"] == tid]
            else:
                m = mitre_df[mitre_df["TechniqueName"].str.lower() == label.lower()]
            if not m.empty:
                tid = m.iloc[0]["TechniqueID"]
                disp = m.iloc[0]["TechniqueDisplay"]
                ids.append(tid); disps.append(disp); scores.append(0.9); tacts.append(m.iloc[0]["Tactics"] or "")
        if ids:
            return ",".join(ids), ",".join(disps), "Library", scores, tacts

    # 2) Two-stage retrieval + rerank (non-LLM)
    cand_idx = candidate_generation(desc, tech_emb, emb_model, TOPK_CANDIDATES)
    candidates_df = mitre_df.iloc[cand_idx].copy()
    reranked = rerank_candidates(desc, candidates_df, reranker)

    chosen = choose_labels(reranked, mitre_df)
    if not chosen:
        return "", "", "N/A", [], []

    ids  = [c[0] for c in chosen]
    disps= [c[1] for c in chosen]
    confs= [c[2] for c in chosen]
    tacs = [c[3] for c in chosen]
    return ",".join(ids), ",".join(disps), "Model", confs, tacs

# -----------------------------
# Streamlit UI (preserves your design/flow)
# -----------------------------
def run_app():
    st.set_page_config(page_title="ATT&CK Mapper", layout="wide")
    st.sidebar.title("Navigation")
    st.sidebar.button("Home", disabled=True)
    st.title("ATT&CK Mapper")

    with st.expander("Process", expanded=False):
        st.markdown("""
1. **Upload** your security use cases CSV  
2. The tool **checks** if the use case exists in the library  
3. If found, it uses the **pre-mapped** MITRE data  
4. If not, it **analyzes** the use case using embeddings + reranker and maps it  
5. **View** mapped results, analytics, and export options  
6. **Discover** additional relevant use cases based on your log sources
""")

    # Inputs
    mitre_file = st.file_uploader("Upload MITRE Techniques JSON", type=["json"])
    lib_file   = st.file_uploader("Upload Library CSV (optional)", type=["csv"])
    usecase_csv= st.file_uploader("Upload Use Cases CSV", type=["csv"])

    if not mitre_file or not usecase_csv:
        st.info("Upload MITRE JSON and Use Cases CSV to proceed.")
        return

    # Load data
    mitre_df = load_mitre_json(mitre_file)
    emb_model, reranker = load_models()
    mitre_df, tech_emb = embed_techniques(mitre_df)

    library_df = None
    if lib_file is not None:
        library_df = pd.read_csv(lib_file)

    # Use cases CSV requires a "Description" column
    df = pd.read_csv(usecase_csv)
    if "Description" not in df.columns:
        st.error("CSV must include a 'Description' column.")
        return

    # Map rows
    out_rows = []
    layer_counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        desc = str(row.get("Description", ""))
        tids_csv, tdisp_csv, source, confs, tacts = map_row_to_mitre(desc, mitre_df, tech_emb, emb_model, reranker, library_df)

        # Maintain layer counts correctly (comma split)
        for tid in tids_csv.split(","):
            tid = tid.strip()
            if tid:
                layer_counts[tid] = layer_counts.get(tid, 0) + 1

        out_rows.append({
            **row.to_dict(),
            "TechniqueIDs": tids_csv,
            "Techniques": tdisp_csv,
            "Match Source": source,
            "ConfScores": ";".join([f"{c:.3f}" for c in confs]),
            "Tactics": ";".join(tacts)
        })

    mapped_df = pd.DataFrame(out_rows)

    # Frequency counts per technique for coverage-gap component
    freq_counts: Dict[str, int] = {}
    for val in mapped_df["TechniqueIDs"].fillna("").astype(str):
        for tid in [t.strip() for t in val.split(",") if str(t).strip()]:
            freq_counts[tid] = freq_counts.get(tid, 0) + 1

    # Compute Priority Score per row
    prios = []
    for _, r in mapped_df.iterrows():
        conf_scores = [float(x) for x in str(r.get("ConfScores","")).split(";") if x]
        tactics_list= [x for x in str(r.get("Tactics","")).split(";") if x]
        tech_ids    = [t.strip() for t in str(r.get("TechniqueIDs","")).split(",") if t.strip()]
        match_src   = str(r.get("Match Source","N/A"))
        has_log     = str(r.get("Log Source","")).strip() != ""
        prio = compute_row_priority(conf_scores, tactics_list, tech_ids, freq_counts, match_src, has_log)
        prios.append(prio)
    mapped_df["Priority Score"] = prios

    # Sort results by priority (desc)
    mapped_df = mapped_df.sort_values(["Priority Score"], ascending=False).reset_index(drop=True)

    st.success("Mapping complete.")
    st.dataframe(mapped_df, use_container_width=True)
    st.download_button("Download Results as CSV", data=mapped_df.to_csv(index=False).encode("utf-8"),
                       file_name="mapped_usecases.csv", mime="text/csv")

    # Navigator preview stats (unchanged)
    st.subheader("Navigator Layer Preview (counts)")
    st.write(f"Unique techniques in mapping: **{len(layer_counts)}**")
    st.json(layer_counts)

if __name__ == "__main__":
    run_app()
