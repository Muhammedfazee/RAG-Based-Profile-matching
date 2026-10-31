# RAG-Based Profile Matching

Find relevant candidates for a job description using semantic resume search,
explicit skill and experience filters, and evidence-based explanations.

Built with **LangChain**, **Hugging Face embeddings**, **Chroma**, and
**OpenRouter**. Resume embeddings and retrieval run locally; OpenRouter handles
natural-language job requirement extraction and generated match explanations.

## Features

- Read PDF, DOCX, and TXT resumes.
- Extract candidate names, skills, education, and explicitly stated experience.
- Preserve resume sections when splitting documents into searchable chunks.
- Combine semantic similarity with keyword skill coverage.
- Filter candidates by must-have skills, total experience, and skill-specific years.
- Return up to ten unique candidates with scores, relevant excerpts, and reasoning.
- Run local matching without API calls when job requirements are supplied as JSON.
- Explore retrieval quality and latency in an executed Jupyter notebook.

## How it works

```text
Resumes (PDF / DOCX / TXT)
    |
Read text and extract metadata
    |
Split by section, then by chunk size
    |
Local Hugging Face embeddings
    |
Persistent Chroma vector store
    |
Job description -> semantic search + skill coverage
    |
Must-have filters -> candidate ranking
    |
Resume excerpts + optional OpenRouter explanations
```

### Technology

| Component | Implementation |
|---|---|
| Pipeline and prompt chains | LangChain |
| Local embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Chroma with cosine distance |
| Job parsing and generated explanations | OpenRouter through LangChain's `ChatOpenAI` |
| Document reading | `pypdf`, `python-docx`, and Python file tools |
| Evaluation | Jupyter notebook with fixed relevance labels |

## Quick start

Use Python 3.10+ and run these commands from the project directory.

### 1. Install dependencies

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Configure OpenRouter

If you do not already have a local environment file:

```powershell
Copy-Item .env.example .env
```

Set your API key and preferred chat model:

```env
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=openrouter/free
```

The environment file is ignored by Git. Never commit your API key.
Free model availability, rate limits, response time, and output quality can vary.
You can select another OpenRouter chat model explicitly.

An API key is only needed for job parsing and generated explanations, not local
embeddings or retrieval. The first indexing run downloads the public embedding
model; later runs can reuse its local cache.

### 3. Index the resumes

```powershell
python resume_rag.py
```

This reads the sample resumes and persists their embeddings in the generated
local vector database. Re-run indexing after adding, changing, or removing
resumes. Rebuilding replaces the collection's records without duplicating chunks.

### 4. Match a job description

**Plain-text JD:** extract requirements and generate explanations with OpenRouter.

```powershell
python job_matcher.py --jd .\job_descriptions\python_backend.txt
```

**Structured JSON job:** use explicit requirements and generate explanations
with OpenRouter.

```powershell
python job_matcher.py --job .\job_descriptions\python_backend.json
```

**Local-only matching:** use explicit requirements and deterministic explanations,
without OpenRouter calls.

```powershell
python job_matcher.py --job .\job_descriptions\python_backend.json --no-llm
```

The local-only mode still uses real Hugging Face embeddings and Chroma semantic
retrieval. It requires a JSON job because plain-text requirement extraction uses
the LLM.

Limit the number of results:

```powershell
python job_matcher.py --job .\job_descriptions\python_backend.json --top-k 3
```

## Job descriptions and output

Each sample job has a plain-text version and a JSON version. The JSON format
separates the description from its explicit filtering requirements:

```json
{
  "description": "Build Python backend services. Python and Django are required, with 5+ years of Python experience. AWS is preferred.",
  "requirements": {
    "required_skills": ["Python", "Django"],
    "preferred_skills": ["AWS"],
    "minimum_experience": 5,
    "skill_minimum_years": {"Python": 5}
  }
}
```

Matching returns the following structure. The abbreviated example illustrates
the format; scores depend on the job and indexed resumes.

```json
{
  "job_description": "Build Python backend services...",
  "top_matches": [
    {
      "candidate_name": "Aarav Sharma",
      "resume_path": "c01_aarav_sharma.pdf",
      "match_score": 93.11,
      "matched_skills": ["AWS", "Django", "Docker", "PostgreSQL", "Python"],
      "relevant_excerpts": [
        "[Experience] Senior Backend Engineer, Fictional Studio 01\nBuilt order-processing APIs in Django with transactional PostgreSQL storage."
      ],
      "reasoning": "The resume provides evidence of Python backend experience and the requested skills."
    }
  ]
}
```

Resume paths are relative to the resume input directory.

## Ranking and filtering

The default hybrid score is:

```text
match_score = 100 * (0.7 * semantic_similarity + 0.3 * requested_skill_coverage)
```

- **Semantic similarity:** the strongest matching resume chunk's cosine
  similarity, mapped from [-1, 1] to [0, 1].
- **Skill coverage:** the fraction of distinct requested skills present in the
  resume's Skills section, including required, preferred, and skill-year requirements.
- **Hard filters:** required skills, minimum total experience, and skill-specific
  experience must all be satisfied before a candidate is returned.

Aliases such as `Postgres`/`PostgreSQL` and `k8s`/`Kubernetes` are normalized.
Multiple matching chunks are grouped into one candidate result.

Experience is based on explicit declarations. Eight years of total experience
and Python in a skills list do not establish five years of Python experience.
Unknown experience does not satisfy a positive minimum.

The matcher returns fewer than ten results when fewer candidates qualify,
or an empty list when none qualify. OpenRouter commentary does not determine
eligibility or scores. Scores indicate relative relevance, **not hiring probability**.

## Project structure

| File or directory | Purpose |
|---|---|
| [resume_rag.py](./resume_rag.py) | Metadata extraction, section-aware chunking, local embeddings, and vector persistence |
| [job_matcher.py](./job_matcher.py) | Requirement parsing, hybrid matching, filters, ranking, and explanations |
| [fs_tools.py](./fs_tools.py) | Read-only document tools reused from the file-system assistant |
| [resumes/](./resumes/) | 32 unique fictional candidates: 24 TXT, four PDF, and four DOCX |
| [job_descriptions/](./job_descriptions/) | Six roles, each provided as TXT and structured JSON |
| [experiments.ipynb](./experiments.ipynb) | Executed experiments, measured retrieval metrics, latency, and analysis |
| [relevance_labels.json](./relevance_labels.json) | Author-curated evaluation labels, never supplied to the matcher |
| [test_rag.py](./test_rag.py) | Focused tests for document processing, matching, persistence, and prompt chains |
| [requirements.txt](./requirements.txt) | Python dependencies |
| [.env.example](./.env.example) | OpenRouter configuration template |

## Experiments and measured results

Open [experiments.ipynb](./experiments.ipynb) in VS Code, select the project's
virtual environment, and run all cells. It compares keyword, semantic, hybrid,
and filtered-hybrid retrieval against fixed relevance judgments.

### Retrieval quality

Results from the saved execution on 32 fictional resumes and six jobs:

| Retrieval variant | Recall@10 | nDCG@10 | Returned precision | Must-have violation rate |
|---|---:|---:|---:|---:|
| Keyword | 1.0000 | 0.9504 | 0.2833 | 0.7167 |
| Semantic | 1.0000 | 0.9033 | 0.2833 | 0.7167 |
| Hybrid | 1.0000 | 0.9519 | 0.2833 | 0.7167 |
| Hybrid + strict filters | 1.0000 | 0.9932 | 1.0000 | 0.0000 |

Precision@10 is 0.2833 for all variants because each job has only two or three
relevant candidates. Filtered search correctly returns only eligible candidates;
returned precision divides by the actual result count rather than ten.

### Latency

The saved notebook execution measured filtered-hybrid local retrieval at
**39.32 ms p50** and **43.94 ms p95**, using 60 timed queries per variant
(ten per job after a warm-up). Query timings include local query
embedding, vector search, aggregation, filtering, scoring, and evidence formatting.
They exclude model/index initialization and OpenRouter calls, and vary by machine.

The notebook also compares 500/50 and 700/100 character chunk settings.
Both produced 224 chunks because the sample resume sections are short; this
dataset does not establish an optimal chunk size.

### OpenRouter verification

A separate live smoke test using `openrouter/free` successfully verified:

- Job requirement extraction, including five years of Python experience.
- Retrieval and filtering of three eligible backend candidates.
- Generated explanations for all three matches.

That check took approximately 61 seconds, including model loading and API calls;
it is not a repeated latency benchmark. The notebook defaults to
`RUN_OPENROUTER = False` for reproducible local evaluation without API calls.
Set it to `True` to generate live explanations for the example.

## Tests

```powershell
python -m unittest test_rag -v
```

All nine tests passed in the verified environment. Tests use deterministic fake
embeddings and simulated chat responses where appropriate; the notebook's
retrieval measurements use the real local embedding model.

## Limitations and data privacy

- The dataset is small, synthetic, and includes deliberate near-miss candidates.
  Its metrics are illustrative, not evidence of production hiring accuracy.
- Metadata extraction expects labeled resume sections and explicitly stated
  experience. It does not infer years from employment dates.
- Scanned PDFs require OCR, which is not included.
- Retrieval examines all indexed chunks before grouping candidates, which is
  suitable for this dataset but should be optimized for larger collections.
- Generated explanations can contain errors and should be reviewed by a person.
- Embeddings and the vector store remain local. OpenRouter receives job text and,
  when explanations are enabled, selected candidate metadata and excerpts.
  Do not process real personal resumes through external APIs without authorization.
