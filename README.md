# RAG-Based Profile Matching

A simple LangChain RAG assignment, following the supplied
`Copy_of_LangChain_RAG_OpenRouter.ipynb`:

**Read resumes -> split sections -> local Hugging Face embeddings -> Chroma
retrieval -> skill filtering/ranking -> OpenRouter explanations.**

Chroma replaces the reference notebook's FAISS because the assignment explicitly
lists Chroma. Local embeddings use the same
`sentence-transformers/all-MiniLM-L6-v2` model as the reference; no paid embedding
API is needed.

## Required files and deliverables

| File/directory | Purpose |
|---|---|
| `resume_rag.py` | Read documents, extract metadata, split sections, embed, persist vectors |
| `job_matcher.py` | Semantic/keyword matching, must-have filters, top-10 output, LangChain/OpenRouter chains |
| `fs_tools.py` | Read-only PDF/TXT/DOCX tools reused from the file-system assistant |
| `resumes/` | 32 unique fictional candidates: 24 TXT, four PDF, four DOCX |
| `job_descriptions/` | Six jobs, each as plain text and JSON with explicit requirements |
| `relevance_labels.json` | Author-curated relevance grades used only for evaluation |
| `experiments.ipynb` | Chunking experiments, retrieval comparisons, accuracy/latency measurements, analysis |
| `test_rag.py` | Focused offline tests |
| `requirements.txt`, `.env.example` | Dependencies and OpenRouter configuration |

## Setup

Python 3.10+ is required. From this project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your key in `.env`, which is ignored by Git:

```env
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=openrouter/free
```

You may explicitly choose another OpenRouter chat model. Free model availability
and rate limits vary. API failures are reported, not replaced with fabricated
responses. Never commit the key. The first indexing run downloads the public
Hugging Face model; subsequent runs can use its local cache.

## Run

Build the local persistent Chroma collection:

```powershell
python resume_rag.py
```

Match a JSON job with explicit requirements and OpenRouter-generated explanations:

```powershell
python job_matcher.py --job .\job_descriptions\python_backend.json
```

Match a text JD, extracting its requirements through a LangChain/OpenRouter chain:

```powershell
python job_matcher.py --jd .\job_descriptions\python_backend.txt
```

Run the same real local RAG pipeline without OpenRouter calls:

```powershell
python job_matcher.py --job .\job_descriptions\python_backend.json --no-llm
```

`--no-llm` uses explicit JSON requirements and deterministic evidence-based
explanations. It does **not** replace semantic retrieval with fake embeddings.
Both indexing and this mode work without an API key or OpenRouter credits.

## Matching behavior

- Extract name, skills, stated total experience, education, and explicitly stated
  skill-specific years from resume sections; store metadata alongside every chunk.
- Preserve section boundaries before applying LangChain's recursive text splitter
  (700 characters, 100-character overlap).
- Aggregate matching sections into unique candidates before selecting top 10.
- Hybrid relevance score:
  `100 * (0.7 * semantic_similarity + 0.3 * requested_skill_coverage)`.
  Cosine similarity is mapped from [-1, 1] to [0, 1]. Scores are heuristics, not
  hiring probabilities.
- Enforce required skills, minimum total years, and skill-specific years.
  Eight years total plus Python in a skills list does not establish five years
  of Python. Unknown requirements do not pass hard filters.
- Return fewer than ten candidates when fewer qualify, or an empty list if none do.
- Return the assignment's exact JSON fields: `job_description`, `top_matches`,
  and each match's `candidate_name`, `resume_path`, `match_score`,
  `matched_skills`, `relevant_excerpts`, and `reasoning`.

Re-run indexing after dataset changes. It replaces this collection's records
without duplicating chunks. `storage/` is generated locally and ignored.

## Notebook and validation

Open `experiments.ipynb` in VS Code, select this project's `.venv` interpreter,
and run all cells. It compares keyword, semantic, hybrid, and filtered-hybrid
retrieval against fixed labels and reports Precision@10, Recall@10, nDCG@10,
returned precision, must-have violations, and p50/p95 query latency.

The notebook runs local embeddings and retrieval by default. OpenRouter calls
are explicitly disabled with `RUN_OPENROUTER = False`: the earlier paid API
check returned HTTP 402 and the user requested no more live measurements.
OpenRouter parsing/generation latency must not be confused with measured local
retrieval latency. Enable that cell only when ready to make API calls.

```powershell
python -m unittest test_rag -v
```

### Measured results

The executed notebook records real local model/vector-search measurements:

| Retrieval variant | Recall@10 | nDCG@10 | Returned precision | Must-have violations |
|---|---:|---:|---:|---:|
| Keyword | 1.0000 | 0.9504 | 0.2833 | 0.7167 |
| Semantic | 1.0000 | 0.9033 | 0.2833 | 0.7167 |
| Hybrid | 1.0000 | 0.9519 | 0.2833 | 0.7167 |
| Hybrid + strict filters | 1.0000 | 0.9932 | 1.0000 | 0.0000 |

Precision@10 is 0.2833 for all variants because each job has only two or three
relevant candidates. Filtered search correctly returns only those candidates,
not ten padded results. See the notebook's saved outputs for per-job metrics,
indexing time, and measured p50/p95 latency. These are local retrieval results;
live OpenRouter generation remains unmeasured.

The dataset is small, synthetic, and deliberately includes near-miss candidates.
Results demonstrate the assignment, not production hiring accuracy. Skill parsing
expects labeled sections and explicit experience declarations. OCR, date-based
experience inference, UI, and deployment are intentionally omitted. LLM
commentary should be reviewed; it never determines eligibility or scores.