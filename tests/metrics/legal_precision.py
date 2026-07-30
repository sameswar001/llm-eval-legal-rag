"""Custom G-Eval metric: Legal Precision.

Generic Faithfulness/Answer Relevancy catch hallucination and off-topic
answers, but they don't specifically penalize the failure mode that
matters most for a contract-review assistant: getting a number, date,
clause reference, or party name *slightly* wrong. Legal Precision is a
G-Eval metric scoped to exactly that.
"""
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

legal_precision_metric = GEval(
    name="Legal Precision",
    criteria=(
        "Determine whether the 'actual output' precisely and correctly reflects the legal "
        "terms found in the 'retrieval context' — including monetary amounts, dates, notice "
        "periods, section/clause references, and party names. Penalize heavily for any invented, "
        "approximated, or vague figure/date/term that isn't explicitly supported by the "
        "retrieval context, even if the overall gist of the answer is roughly right. Penalize "
        "vague hedging (e.g. 'around', 'roughly', 'a few months') when the retrieval context "
        "states an exact figure."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.7,
)
