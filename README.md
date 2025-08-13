# ATT&CK Mapping & Analytics App (Improved)

## Overview
This Streamlit-based application maps use case descriptions to MITRE ATT&CK techniques using a **two-stage retrieval + rerank pipeline** without relying on LLMs.  
It also provides coverage analytics and log-source–aware suggestions.

**Features:**
- Embedding-based candidate retrieval (`all-mpnet-base-v2`)
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) for improved accuracy
- Configurable **threshold & abstain** policy
- Multi-label support (returns top-N techniques above threshold)
- Handles sub-techniques (T####.###)
- Stores results as `TechniqueIDs` + `TechniqueDisplays` for consistency
- ATT&CK framework coverage calculation (default total = **211** techniques)
- Log source–aware suggestions from a library CSV

---

## Project Structure
- `app.txt` – main Streamlit app for mapping use cases to ATT&CK techniques
- `analytics.txt` – analytics dashboard for coverage & mapped techniques
- `suggestions.txt` – suggestions engine based on log sources
- `requirements.txt` – Python dependencies
- `README.md` – project documentation

---

## Installation
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Apps

1. **Mapping App**  
   ```bash
   streamlit run app.txt
   ```

2. **Analytics App**  
   ```bash
   streamlit run analytics.txt
   ```

3. **Suggestions App**  
   ```bash
   streamlit run suggestions.txt
   ```

---

## Input Requirements

### MITRE Techniques JSON
- Must include `technique_id` (or equivalent), `name`, `description`.
- Can be exported from ATT&CK Navigator or MITRE ATT&CK bundle.

### Use Cases CSV
- Must contain a `Description` column with the use case text.
- Optional: `Log Source` column for better candidate filtering.

### Library CSV (Optional)
- Should contain columns for `Use Case` and `Technique` for exact/heuristic matching.

---

## Configuration
You can adjust mapping behavior by editing constants in `app.txt`:
- `TOPK_CANDIDATES` – number of candidates retrieved from embeddings
- `TOPN_LABELS` – max techniques returned per use case
- `TAU_ACCEPT` – score threshold for acceptance
- `DELTA_MARGIN` – score margin between first and second candidate for single-label decisions

---

## Coverage Formula
Coverage (%) is calculated as:
```
coverage = (number of unique TechniqueIDs / TOTAL_TECHNIQUES) * 100
```
where `TOTAL_TECHNIQUES` defaults to **211** (update as ATT&CK evolves).

---

## License
MIT License
