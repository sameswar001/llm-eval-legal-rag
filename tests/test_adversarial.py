"""Robustness suite: out-of-scope questions and instruction-override attempts.

A model that scores well on test_rag_e2e.py can still fail here by
confidently fabricating an answer to a question the contracts don't
cover, or by complying with an embedded "ignore your instructions"
attempt. This suite exists specifically to catch that failure mode.

Run with:
    uv run deepeval test run tests/test_adversarial.py
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from legal_rag.pipeline import LegalRAGPipeline
from tests.goldens.legal_qa_goldens import ADVERSARIAL_GOLDENS
from tests.metrics.scope_awareness import scope_awareness_metric
from typing import Any, List, cast

pipeline = LegalRAGPipeline()


@pytest.mark.parametrize("golden", ADVERSARIAL_GOLDENS)
def test_adversarial_scope_awareness(golden):
    result = pipeline.answer(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.answer,
        expected_output=golden.expected_output,
        retrieval_context=cast(List[Any], result.contexts),
    )

    assert_test(test_case=test_case, metrics=[scope_awareness_metric])
