import json
import unittest
from pathlib import Path

from retrieval_engine import AP25RetrievalEngine


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "AP25_parser_workbench.ipynb"


class RegressionTests(unittest.TestCase):
    def test_notebook_no_longer_contains_extractive_answer_path(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertNotIn("def build_extractive_answer", code)
        self.assertNotIn("query_requests_extractive_fact", code)

    def test_detect_query_mode_routes_requirement_question_to_normative_lookup(self):
        engine = AP25RetrievalEngine(chunks=[], vectordb=None, reranker=None)
        query = "Какие требования к независимости подачи топлива в двигатели?"
        self.assertEqual(engine.detect_query_mode(query), "normative_lookup")

    def test_detect_query_mode_routes_explicit_shortlist_question_to_issue_spotting(self):
        engine = AP25RetrievalEngine(chunks=[], vectordb=None, reranker=None)
        query = "Какие пункты нужно проанализировать при оценке изменения пассажировместимости?"
        self.assertEqual(engine.detect_query_mode(query), "issue_spotting")


if __name__ == "__main__":
    unittest.main()
