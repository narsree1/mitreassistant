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

TOPK_CANDIDATES  = 15       # stage-1 candidates
TOPN_LABELS      = 3        # allow up to N labels if confident
TAU_ACCEPT       = 0.42     # accept threshold (tune on eval set)
DELTA_MARGIN     = 0.06     # winner margin over next best to accept as single-label

@st.cache_resource(show_spinner=False)
def load_models():
    emb = SentenceTransformer(EMBED_MODEL_NAME)
    rer = CrossEncoder(RERANKER_NAME)  # lightweight cross-encoder
    return emb, rer

# -----------------------------
# MITRE ATT&CK Loading
# -----------------------------
def normalize_technique_row(tid: str, name: str) -> Tuple[str, str, str]:
    """
    Returns: (technique_id, technique_name, display_string)
    display = "T####[.###] - Name"
    """
    tid = tid.strip()
    name = name.strip()
    display = f"{tid} - {name}"
    return tid, name, display

def load_mitre_json(mitre_json_path: str) -> pd.DataFrame:
    """
    Load techniques + sub-techniques from a MITRE JSON (exported Attack Navigator or official ATT&CK bundle).
    Expected keys: technique_id (e.g., "T1110" or "T1110.003"), name, description, tactic(s) optional.
    """
    with open(mitre_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for obj in data.get("techniques", data.get("objects", [])):
        # Attempt to be permissive about shape
        tid = obj.get("technique_id") or obj.get("external_id") or obj.get("id") or ""
        name = obj.get("name") or obj.get("technique", "")
        desc = obj.get("description", "")
        tactic = ", ".join(obj.get("tactic", obj.get("tactics", []))) if isinstance(obj.get("tactic", obj.get("tactics", [])), list) else obj.get("tactic", obj.get("tactics", ""))

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
    # sort descending
    return idx[np.argsort(-sims[idx])].tolist()

def rerank_candidates(query: str, candidates_df: pd.DataFrame, reranker: CrossEncoder) -> List[Tuple[int, float]]:
    pairs = [(query, f"{row.TechniqueDisplay}: {row.Description}") for _, row in candidates_df.iterrows()]
    scores = reranker.predict(pairs).tolist()  # higher is better
    order = np.argsort(scores)[::-1]
    return [(candidates_df.index[i], float(scores[i])) for i in order]

def choose_labels(reranked: List[Tuple[int, float]], mitre_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
    """
    Apply thresholds + multi-label logic.
    Returns list of (TechniqueID, TechniqueDisplay, score)
    """
    if not reranked:
        return []

    results: List[Tuple[str, str, float]] = []
    # accept all above TAU_ACCEPT up to TOPN_LABELS
    for idx, score in reranked[:TOPN_LABELS]:
        if score >= TAU_ACCEPT:
            tid = mitre_df.at[idx, "TechniqueID"]
            disp = mitre_df.at[idx, "TechniqueDisplay"]
            results.append((tid, disp, score))

    # If nothing passed threshold, abstain
    if not results:
        return []

    # If we picked >1 but first is not sufficiently separated, keep the multi-labels
    if len(results) == 1 and len(reranked) > 1:
        best = results[0][2]
        next_best = reranked[1][1]
        if best - next_best < DELTA_MARGIN:
            # permit second if it is also above threshold
            idx2, score2 = reranked[1]
            if score2 >= TAU_ACCEPT and len(results) < TOPN_LABELS:
                tid2 = mitre_df.at[idx2, "TechniqueID"]
                disp2 = mitre_df.at[idx2, "TechniqueDisplay"]
                results.append((tid2, disp2, score2))

    return results

# -----------------------------
# Library Matching (exact/heuristic)
# -----------------------------
def library_fast_match(usecase: str, library_df: pd.DataFrame, lib_threshold: float = 0.80) -> List[str]:
    """
    If the library contains pre-labeled mappings, try to surface them.
    Here we implement a simple heuristic: exact/substring match over canonical title column,
    with optional embedding similarity for borderline cases (future extension).
    Returns a list of TechniqueDisplay strings (comma-separated later).
    """
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

    # De-dup and normalize formatting
    hits = [h for h in dict.fromkeys(hits) if h]
    return hits

# -----------------------------
# Mapping Pipeline
# -----------------------------
def map_row_to_mitre(description: str,
                     mitre_df: pd.DataFrame,
                     tech_emb: np.ndarray,
                     emb_model: SentenceTransformer,
                     reranker: CrossEncoder,
                     library_df: Optional[pd.DataFrame] = None) -> Tuple[str, str, str]:
    """
    Returns: (TechniqueIDsCSV, TechniqueDisplaysCSV, MatchSource)
    MatchSource in {"Library", "Model", "N/A"}
    """
    desc = (description or "").strip()
    if not desc:
        return "", "", "N/A"

    # 1) Library exact/heuristic match first
    lib_hits = library_fast_match(desc, library_df)
    if lib_hits:
        # Ensure "T#### - Name" format; if the library has names only, try to align via mitre_df
        ids, disps = [], []
        for label in lib_hits:
            label = label.strip()
            # Accept either "T#### - Name" or TechniqueID alone
            if label.startswith("T"):
                tid = label.split(" ")[0].strip()
                m = mitre_df[mitre_df["TechniqueID"] == tid]
            else:
                m = mitre_df[mitre_df["TechniqueName"].str.lower() == label.lower()]
            if not m.empty:
                tid = m.iloc[0]["TechniqueID"]
                disp = m.iloc[0]["TechniqueDisplay"]
                ids.append(tid)
                disps.append(disp)
        if ids:
            return ",".join(ids), ",".join(disps), "Library"

    # 2) Two-stage retrieval + rerank (non-LLM)
    cand_idx = candidate_generation(desc, tech_emb, emb_model, TOPK_CANDIDATES)
    candidates_df = mitre_df.iloc[cand_idx].copy()
    reranked = rerank_candidates(desc, candidates_df, reranker)

    chosen = choose_labels(reranked, mitre_df)
    if not chosen:
        return "", "", "N/A"

    ids  = [c[0] for c in chosen]
    disps= [c[1] for c in chosen]
    return ",".join(ids), ",".join(disps), "Model"

# -----------------------------
# Navigator Count Helper
# -----------------------------
def count_techniques_for_layer(techniques_csv: str) -> Dict[str, int]:
    """
    Properly split comma-separated technique IDs and count each once.
    """
    counts: Dict[str, int] = {}
    if not techniques_csv:
        return counts
    for part in techniques_csv.split(","):
        tid = part.strip()
        if not tid:
            continue
        counts[tid] = counts.get(tid, 0) + 1
    return counts

# -----------------------------
# Streamlit UI (simplified core)
# -----------------------------
def run_app():
    st.set_page_config(page_title="ATT&CK Mapper (Improved)", layout="wide")
    st.title("ATT&CK Mapper (Improved, Non‑LLM Rerank)")

    with st.sidebar:
        st.header("Inputs")
        mitre_file = st.file_uploader("Upload MITRE Techniques JSON (Navigator export or bundle)", type=["json"])
        lib_file   = st.file_uploader("Upload Labeled Library CSV (optional)", type=["csv"])
        usecase_csv= st.file_uploader("Upload Use Cases CSV", type=["csv"])

        st.caption("Two‑stage pipeline: Embedding retrieval ➜ Cross‑encoder rerank ➜ Threshold & multi‑label.")

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
        tids_csv, tdisp_csv, source = map_row_to_mitre(desc, mitre_df, tech_emb, emb_model, reranker, library_df)

        # Maintain layer counts correctly (comma split)
        for tid in tids_csv.split(","):
            tid = tid.strip()
            if tid:
                layer_counts[tid] = layer_counts.get(tid, 0) + 1

        out_rows.append({
            **row.to_dict(),
            "TechniqueIDs": tids_csv,
            "Techniques": tdisp_csv,
            "Match Source": source
        })

    mapped_df = pd.DataFrame(out_rows)

    st.success("Mapping complete.")
    st.dataframe(mapped_df, use_container_width=True)

    st.download_button("Download Mapped CSV", data=mapped_df.to_csv(index=False).encode("utf-8"), file_name="mapped_usecases.csv", mime="text/csv")

    # Simple layer preview stats
    st.subheader("Navigator Layer Preview (counts)")
    st.write(f"Unique techniques in mapping: **{len(layer_counts)}**")
    st.json(layer_counts)

if __name__ == "__main__":
    run_app()
