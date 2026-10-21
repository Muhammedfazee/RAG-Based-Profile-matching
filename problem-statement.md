Learning Objectives

    Implement document chunking and embedding

    Build vector databases

    Create retrieval pipelines

    Understand semantic search

Assignment Requirements

Part A: RAG System Setup (50%)

Create resume_rag.py:

    Document Processing Pipeline

        Load resumes using file system tools from Milestone 1

        Chunk documents intelligently (preserve sections like Education, Experience)

        Generate embeddings using OpenAI/Cohere/HuggingFace models

        Store in vector database (ChromaDB, Pinecone, or Weaviate)

    Metadata Extraction

        Extract key fields: Name, Skills, Experience Years, Education

        Store metadata alongside embeddings for filtering

Part B: Job Matching Engine (50%)

Create job_matcher.py:

    Semantic Search

        Accept job description as input

        Convert JD to embedding

        Retrieve top-K similar resumes (K=10)

        Implement hybrid search (semantic + keyword for critical skills)

    Ranking & Scoring

        Score matches (0-100 scale)

        Provide match reasoning (which sections matched)

        Filter by must-have requirements (e.g., “5+ years Python”)

Output Format
{

  "job_description": "...",

  "top_matches": [

    {

      "candidate_name": "John Doe",

      "resume_path": "resumes/john_doe.pdf",

      "match_score": 92,

      "matched_skills": ["Python", "Machine Learning"],

      "relevant_excerpts": ["..."],

      "reasoning": "Strong match for ML experience..."

    }

  ]

}
Submission guidelines

Deliverables

    Complete RAG implementation

    Dataset: 30+ diverse resumes, 5+ job descriptions

    Jupyter notebook with experimentation and analysis

    Performance metrics: retrieval accuracy, latency