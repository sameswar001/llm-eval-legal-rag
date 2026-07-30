"""Core RAG metrics: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall.

Run with:
    uv run deepeval test run tests/test_rag_e2e.py
"""
import pytest
from typing import Any, List, cast
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from legal_rag.pipeline import LegalRAGPipeline
from tests.goldens.legal_qa_goldens import IN_SCOPE_GOLDENS

pipeline = LegalRAGPipeline()


@pytest.mark.parametrize("golden", IN_SCOPE_GOLDENS)
def test_legal_rag_core_metrics(golden):
    result = pipeline.answer(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.answer,
        expected_output=golden.expected_output,
        retrieval_context=cast(List[Any], result.contexts),
    )

    assert_test(
        test_case=test_case,
        metrics=[
            FaithfulnessMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
            ContextualRecallMetric(threshold=0.7),
        ],
    )
