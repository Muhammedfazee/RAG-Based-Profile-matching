"""Load resumes, split sections, embed locally, and store them with LangChain."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fs_tools import SUPPORTED_EXTENSIONS, list_files, read_file

ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "resume_profiles"
HEADINGS = {
    "summary", "professional summary", "skills", "technical skills",
    "skill experience", "experience", "work experience", "professional experience",
    "education", "projects", "certifications",
}
ALIASES = {
    "postgres": "postgresql", "k8s": "kubernetes", "reactjs": "react",
    "nodejs": "node.js", "js": "javascript", "sklearn": "scikit-learn",
}


def normalize_skill(skill: str) -> str:
    value = skill.strip().casefold()
    return ALIASES.get(value, value)


def split_sections(text: str) -> list[tuple[str, str]]:
    result = []
    heading, lines = "Header", []
    for line in text.splitlines():
        label = line.strip().rstrip(":").casefold()
        if label in HEADINGS:
            if lines:
                result.append((heading, "\n".join(lines)))
            heading, lines = label.title(), []
        elif line.strip():
            lines.append(line.strip())
    if lines:
        result.append((heading, "\n".join(lines)))
    return result


def extract_metadata(text: str, resume_path: str) -> dict:
    """Use explicit declarations; never equate total years with skill years."""
    sections = split_sections(text)
    if not sections or sections[0][0] != "Header":
        raise ValueError(f"Resume needs a candidate name before its sections: {resume_path}")
    skills = [
        skill.strip() for heading, content in sections
        if heading in {"Skills", "Technical Skills"}
        for skill in re.split(r"[,;|\n]", content) if skill.strip()
    ]
    total = re.search(r"(?:total experience|experience years)\s*:\s*(\d+(?:\.\d+)?)", text, re.I)
    if total is None:
        total = re.search(r"(\d+(?:\.\d+)?)\s+years? of (?:professional |total )?experience", text, re.I)
    years = float(total.group(1)) if total else -1
    if not -1 <= years <= 80:
        raise ValueError(f"Invalid total experience in {resume_path}")
    skill_years = {}
    for heading, content in sections:
        if heading == "Skill Experience":
            for match in re.finditer(r"(?m)^(.+?):\s*(\d+(?:\.\d+)?)\s+years?\b", content):
                value = float(match.group(2))
                if value > 80 or (years >= 0 and value > years):
                    raise ValueError(f"Inconsistent skill experience in {resume_path}")
                skill = normalize_skill(match.group(1))
                if skill not in {normalize_skill(item) for item in skills}:
                    raise ValueError(f"Skill experience is absent from the skills section: {skill}")
                skill_years[skill] = value
    return {
        "candidate_name": sections[0][1].splitlines()[0],
        "resume_path": resume_path,
        "skills": json.dumps(skills),
        "experience_years": years,
        "education": json.dumps([content for heading, content in sections if heading == "Education"]),
        "skill_years": json.dumps(skill_years),
    }


def load_resumes(resumes_dir: Path = ROOT / "resumes") -> list[Document]:
    resumes_dir = resumes_dir.resolve()
    files = list_files(str(resumes_dir))
    if any(item.get("success") is False for item in files):
        raise ValueError(f"Could not list resumes: {files}")
    documents = []
    seen = set()
    for item in files:
        if item["extension"] not in SUPPORTED_EXTENSIONS:
            continue
        path = (resumes_dir / item["name"]).resolve()
        if not path.is_relative_to(resumes_dir):
            raise ValueError(f"Resume is outside the workspace: {item['name']}")
        result = read_file(str(path))
        if not result["success"]:
            raise ValueError(f"Cannot read {item['name']}: {result['error']}")
        text = result["content"].strip()
        if not text:
            raise ValueError(f"No text in {item['name']}; scanned PDFs require OCR.")
        fingerprint = " ".join(text.split())
        if fingerprint in seen:
            raise ValueError(f"Duplicate resume: {item['name']}")
        seen.add(fingerprint)
        documents.append(Document(
            page_content=text, metadata=extract_metadata(text, item["name"])
        ))
    if not documents:
        raise ValueError("No supported resume documents found.")
    return documents


def chunk_resumes(
    documents: list[Document], chunk_size: int = 700, chunk_overlap: int = 100
) -> list[Document]:
    if chunk_size < 100 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size >= 100 and 0 <= overlap < chunk_size.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sections = [
        Document(
            page_content=content,
            metadata={**document.metadata, "section": heading},
        )
        for document in documents for heading, content in split_sections(document.page_content)
    ]
    return splitter.split_documents(sections)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm() -> ChatOpenAI:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or key == "replace-with-your-key":
        raise RuntimeError("Set OPENROUTER_API_KEY in .env to use OpenRouter.")
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openrouter/free"),
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=2500,
        timeout=60,
        max_retries=2,
    )


def build_index(
    resumes_dir: Path = ROOT / "resumes", persist_dir: Path = ROOT / "storage"
) -> Chroma:
    chunks = chunk_resumes(load_resumes(resumes_dir))
    embeddings = get_embeddings()
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )
    if (store._collection.metadata or {}).get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Existing collection uses another model. Choose a new storage directory.")
    previous = store.get()["ids"]
    ids = [f"chunk-{index}" for index in range(len(chunks))]
    # LangChain embeds before upserting; failures do not first erase the old records.
    store.add_documents(chunks, ids=ids)
    stale = sorted(set(previous) - set(ids))
    if stale:
        store.delete(ids=stale)
    return store


def load_index(persist_dir: Path = ROOT / "storage") -> Chroma:
    if not persist_dir.is_dir():
        raise ValueError("Index not found. Run python resume_rag.py first.")
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
        create_collection_if_not_exists=False,
    )
    if not store.get()["ids"]:
        raise ValueError("Resume index is empty. Run python resume_rag.py first.")
    if (store._collection.metadata or {}).get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Embedding model differs. Rebuild with python resume_rag.py.")
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumes", type=Path, default=ROOT / "resumes")
    parser.add_argument("--storage", type=Path, default=ROOT / "storage")
    args = parser.parse_args()
    index = build_index(args.resumes, args.storage)
    print(f"Indexed {len(index.get()['ids'])} resume sections in {args.storage}")
