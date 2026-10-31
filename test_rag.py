"""Small offline tests; fake embeddings are never used for notebook metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from job_matcher import Requirements, match_jobs, meets_requirements, parse_requirements
from resume_rag import (
    ROOT, build_index, chunk_resumes, extract_metadata, get_llm, load_index, load_resumes,
)


class TestEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        words = text.casefold()
        return [float(words.count(term) + 1) for term in ["python", "django", "react", "data"]]


class RagTests(unittest.TestCase):
    def test_dataset_and_section_chunking(self):
        documents = load_resumes()
        self.assertEqual(len(documents), 32)
        self.assertEqual(len({doc.metadata["candidate_name"] for doc in documents}), 32)
        self.assertEqual(
            {Path(doc.metadata["resume_path"]).suffix for doc in documents},
            {".pdf", ".docx", ".txt"},
        )
        chunks = chunk_resumes(documents)
        self.assertGreater(len(chunks), len(documents))
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 700)
            self.assertIn("section", chunk.metadata)
        self.assertEqual(len(list((ROOT / "job_descriptions").glob("*.json"))), 6)
        labels = json.loads((ROOT / "relevance_labels.json").read_text())["jobs"]
        paths = {doc.metadata["resume_path"] for doc in documents}
        self.assertTrue(all(set(job["grades"]) == paths for job in labels.values()))

    def test_explicit_skill_years_and_aliases(self):
        metadata = extract_metadata(
            "Jane Doe\nSUMMARY\nTotal experience: 8 years\nSKILLS\nPython, Postgres, JavaScript\n"
            "SKILL EXPERIENCE\nPython: 4 years\nEDUCATION\nB.S.\n", "jane.txt"
        )
        self.assertFalse(meets_requirements(metadata, Requirements(
            required_skills=["Python"], skill_minimum_years={"Python": 5},
        )))
        self.assertTrue(meets_requirements(metadata, Requirements(
            required_skills=["PostgreSQL"], minimum_experience=5,
        )))
        self.assertFalse(meets_requirements(metadata, Requirements(required_skills=["Java"])))
        unknown = extract_metadata("Someone\nSKILLS\nPython", "someone.txt")
        self.assertEqual(unknown["experience_years"], -1)
        self.assertFalse(meets_requirements(unknown, Requirements(minimum_experience=1)))

    def test_bad_documents_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "No supported"):
                load_resumes(root)
            (root / "empty.txt").write_text("")
            with self.assertRaisesRegex(ValueError, "OCR"):
                load_resumes(root)
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            extract_metadata(
                "Someone\nSUMMARY\nTotal experience: 2 years\nSKILLS\nPython\n"
                "SKILL EXPERIENCE\nPython: 5 years", "someone.txt",
            )

    def test_index_matching_and_top_k(self):
        store = Chroma(
            collection_name="test_" + uuid4().hex,
            embedding_function=TestEmbeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )
        try:
            store.add_documents(chunk_resumes(load_resumes()))
            job = json.loads((ROOT / "job_descriptions" / "python_backend.json").read_text())
            requirements = Requirements.model_validate(job["requirements"])
            result = match_jobs(
                store, job["description"], requirements, llm_reasoning=False,
            )
            self.assertEqual(set(result), {"job_description", "top_matches"})
            self.assertEqual(len(result["top_matches"]), 3)
            fields = {
                "candidate_name", "resume_path", "match_score",
                "matched_skills", "relevant_excerpts", "reasoning",
            }
            metadata = {item["resume_path"]: item for item in store.get()["metadatas"]}
            scores = []
            for match in result["top_matches"]:
                self.assertEqual(set(match), fields)
                self.assertTrue(meets_requirements(metadata[match["resume_path"]], requirements))
                self.assertTrue(0 <= match["match_score"] <= 100)
                self.assertTrue(match["relevant_excerpts"])
                scores.append(match["match_score"])
            self.assertEqual(scores, sorted(scores, reverse=True))
            unfiltered = match_jobs(
                store, job["description"], requirements, strict=False, llm_reasoning=False,
            )
            self.assertEqual(len(unfiltered["top_matches"]), 10)
            self.assertEqual(len({item["resume_path"] for item in unfiltered["top_matches"]}), 10)
            self.assertEqual(match_jobs(
                store, "Rust engineer", Requirements(required_skills=["Rust"]),
                llm_reasoning=False,
            )["top_matches"], [])
            with self.assertRaises(ValueError):
                match_jobs(store, "", requirements, llm_reasoning=False)
        finally:
            store.delete_collection()
            store._client.close()

    @patch("job_matcher.get_llm")
    def test_langchain_requirement_chain(self, get_model):
        get_model.return_value = FakeListChatModel(responses=[
            '{"required_skills":["Python"],"preferred_skills":[],"minimum_experience":5,'
            '"skill_minimum_years":{"Python":5}}'
        ])
        result = parse_requirements("Must have Python and 5+ years Python.")
        self.assertEqual(result.required_skills, ["Python"])
        self.assertEqual(result.skill_minimum_years, {"Python": 5})

    @patch("job_matcher.get_llm")
    def test_langchain_explanation_chain(self, get_model):
        get_model.return_value = FakeListChatModel(responses=[
            "The Skills section supports Python; total experience is unknown."
        ])
        store = Mock()
        metadata = extract_metadata("Jane Doe\nSKILLS\nPython", "jane.txt")
        store.get.return_value = {
            "ids": ["one"], "documents": ["Python"],
            "metadatas": [{**metadata, "section": "Skills"}],
        }
        result = match_jobs(
            store, "Python required", Requirements(required_skills=["Python"]), mode="keyword",
        )
        self.assertIn("The Skills section supports Python", result["top_matches"][0]["reasoning"])

    @patch("resume_rag.get_embeddings", return_value=TestEmbeddings())
    def test_persistent_rebuild_and_reload(self, embeddings):
        with tempfile.TemporaryDirectory() as folder:
            persist = Path(folder) / "chroma"
            store = build_index(persist_dir=persist)
            count = len(store.get()["ids"])
            store._client.close()
            rebuilt = build_index(persist_dir=persist)
            self.assertEqual(len(rebuilt.get()["ids"]), count)
            rebuilt._client.close()
            loaded = load_index(persist)
            self.assertEqual(len(loaded.get()["ids"]), count)
            loaded._client.close()

    @patch("resume_rag.load_dotenv")
    @patch("resume_rag.ChatOpenAI")
    def test_openrouter_endpoint_and_missing_key(self, model, load):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}, clear=True):
            get_llm()
            self.assertEqual(model.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                get_llm()

    def test_requirement_years_validation(self):
        with self.assertRaises(ValueError):
            Requirements(skill_minimum_years={"Python": float("nan")})
        with self.assertRaises(ValueError):
            Requirements(minimum_experience=-1)


if __name__ == "__main__":
    unittest.main()
