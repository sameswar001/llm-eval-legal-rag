"""Domain-specific precision suite, on top of the generic RAG metrics.

Run with:
    uv run deepeval test run tests/test_legal_precision.py
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from typing import Any, List, cast
from legal_rag.pipeline import LegalRAGPipeline
from tests.goldens.legal_qa_goldens import IN_SCOPE_GOLDENS
from tests.metrics.legal_precision import legal_precision_metric

pipeline = LegalRAGPipeline()


@pytest.mark.parametrize("golden", IN_SCOPE_GOLDENS)
def test_legal_precision(golden):
    result = pipeline.answer(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.answer,
        expected_output=golden.expected_output,
        retrieval_context=cast(List[Any], result.contexts),
    )

    assert_test(test_case=test_case, metrics=[legal_precision_metric])
