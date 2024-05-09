import pytest

from benchmark.wer import compute_wer


class TestWER:
    def test_raises_exception_on_mismatched_lengths(self):
        # GIVEN
        #      the number of references and hypotheses do not match
        references = ["a", "b"]
        hypotheses = ["a"]
        error = "Number of references is not equal to the number of hypotheses"

        # WHEN
        #      WER is computed
        with pytest.raises(Exception) as exc_info:
            compute_wer(references, hypotheses)

        # THEN
        #      An exception is raised for the mismatched number of references and hypotheses
        assert str(exc_info.value) == error

    @pytest.mark.parametrize(
        "references,hypotheses,expected_wer,lightweight",
        [
            ("reference string", "hypothesis string", 50.0, True),
            ("reference string", ["hypothesis list"], 100.0, True),
            (["this reference is a list"], "this hypothesis is a string", 40.0, True),
            (
                ["both of these are lists", "this reference is a list"],
                ["this is a list too", "hypothesis"],
                100.0,
                True,
            ),
            ("reference string", "hypothesis string", 50.0, False),
            ("reference string", ["hypothesis list"], 100.0, False),
            (["this reference is a list"], "this hypothesis is a string", 40.0, False),
            (
                ["both of these are lists", "this reference is a list"],
                ["this is a list too", "hypothesis"],
                100.0,
                False,
            ),
        ],
    )
    def test_compute_wer_returns_correct_wer(
        self, references, hypotheses, expected_wer, lightweight
    ):
        # GIVEN
        #      references are given as a string or a list of strings
        #      hypotheses are given as a string or a list of strings
        #      the number of references and hypotheses match
        # WHEN
        #      the WER is computed
        wer = compute_wer(references, hypotheses, lightweight)

        # THEN
        #      the computed WER is correct
        #      no exceptions are raised
        #      both the speechbrain version
        #          and the lcoal levenshtein version
        #          return the same result
        assert wer == pytest.approx(expected_wer)
