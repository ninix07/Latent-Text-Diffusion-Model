"""ROUGE and BLEU metrics for text generation evaluation."""

from __future__ import annotations

import warnings


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Requires the ``rouge_score`` package. Returns an empty dict with a
    warning if the package is not installed.

    Parameters
    ----------
    predictions : list[str]
        Model-generated texts.
    references : list[str]
        Reference texts (one per prediction).

    Returns
    -------
    dict
        Keys: rouge1, rouge2, rougeL (each as F-measure float), or {} on
        import failure.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        warnings.warn(
            "rouge_score package not found. Install with: uv add rouge-score",
            ImportWarning,
            stacklevel=2,
        )
        return {}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )

    totals: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    return {key: val / n if n else 0.0 for key, val in totals.items()}


def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute corpus-level BLEU score using sacrebleu.

    Requires the ``sacrebleu`` package. Returns an empty dict with a
    warning if the package is not installed.

    Parameters
    ----------
    predictions : list[str]
        Model-generated texts.
    references : list[str]
        Reference texts (one per prediction).

    Returns
    -------
    dict
        Keys: bleu (float 0-100), or {} on import failure.
    """
    try:
        import sacrebleu
    except ImportError:
        warnings.warn(
            "sacrebleu package not found. Install with: uv add sacrebleu",
            ImportWarning,
            stacklevel=2,
        )
        return {}

    if not predictions:
        return {"bleu": 0.0}

    result = sacrebleu.corpus_bleu(predictions, [references])
    return {"bleu": result.score}
