from collections import Counter

from speechbrain.utils.edit_distance import accumulatable_wer_stats


def compute_wer(references: list[str], hypotheses: list[str]):
    if isinstance(references, str):
        references = [references.split()]
    else:
        references = [ref.split() for ref in references]
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses.split()]
    else:
        hypotheses = [hyp.split() for hyp in hypotheses]
    if len(references) != len(hypotheses):
        raise Exception("Number of references is not equal to the number of hypotheses")
    stats = accumulatable_wer_stats(references, hypotheses, Counter())
    return stats["WER"]
