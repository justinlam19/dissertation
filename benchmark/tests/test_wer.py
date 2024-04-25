import pytest

from benchmark.wer import compute_wer


class TestWER:
    def test_raises_exception_on_mismatched_lengths(self):
        # Arrange
        references = ["a", "b"]
        hypotheses = ["a"]
        error = "Number of references is not equal to the number of hypotheses"

        # Act
        with pytest.raises(Exception) as exc_info:
            compute_wer(references, hypotheses)

        # Assert
        assert str(exc_info.value) == error

    @pytest.mark.parametrize(
        "references,hypotheses,expected_wer",
        [
            ("reference string", "hypothesis string", 50.0),
            ("reference string", ["hypothesis list"], 100.0),
            (["this reference is a list"], "this hypothesis is a string", 40.0),
            (["both of these are lists"], ["this is a list too"], 100.0),
        ],
    )
    def test_compute_wer_returns_correct_wer(self, references, hypotheses, expected_wer):
        # Arrange
        # Act
        wer = compute_wer(references, hypotheses)

        # Assert
        assert wer == pytest.approx(expected_wer)
