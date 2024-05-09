from collections import Counter

from speechbrain.utils.edit_distance import accumulatable_wer_stats


def compute_wer(references: list[str], hypotheses: list[str], lightweight=False):
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
    if lightweight:
        return leven_wer(references, hypotheses)
    else:
        stats = accumulatable_wer_stats(references, hypotheses, Counter())
        return stats["WER"]


def levenshtein(x, y):
    prev = list(range(len(y) + 1))
    curr = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        curr[0] = i
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(
                    curr[j - 1],  # Insertion
                    prev[j],  # Deletion
                    prev[j - 1],  # Substitution
                )
        prev = curr.copy()
    return curr[len(y)]


def leven_wer(references, hypotheses):
    total_error = 0
    total_length = sum(len(reference) for reference in references)
    for reference, hypothesis in zip(references, hypotheses):
        total_error += levenshtein(reference, hypothesis)
    return total_error / total_length * 100
