"""Fast unit tests: no data download, no browser. Run with:  pytest"""
import math

import pytest

from src import single_file, slides
from src.style import DECK, css_family


class FakeNumbers(dict):
    """Stands in for slides.numbers(): any number the slides ask for is 0.42 unless set below."""
    def __missing__(self, key):
        return 0.42


@pytest.fixture
def n():
    d = FakeNumbers(top_name="Player A", lucky_name="Player B", star="Player C", r0_score="1–1", perm_txt="p ≈ 0.0100",
                    perm_p=0.01, seasons=12, above=10, best_season=2023, hits=11, n=36, n_sims=10_000, perm_B=10_000)
    return d


def test_every_slide_has_a_title_and_no_missing_numbers(n):
    deck = slides.slide_list(n)
    assert len(deck) > 15
    for s in deck:
        assert s["title"].strip(), s["id"]
        text = " ".join([s["title"], *s.get("bullets", []), *s.get("cards", [])])
        assert "nan" not in text.lower().split(), s["id"]


def test_slide_ids_are_unique(n):
    ids = [s["id"] for s in slides.slide_list(n)]
    assert len(ids) == len(set(ids))


def test_presenter_hints_go_to_notes_not_the_screen():
    s = {"bullets": ["Shown on screen", "Press play to start", "Drag to rotate"]}
    assert slides.shown(s) == ["Shown on screen"]
    assert slides.hints(s) == ["Press play to start", "Drag to rotate"]


def test_numbered_cards():
    assert slides.numbered(["1 Regression", "Plain card"]) == [("01", "Regression"), ("02", "Plain card")]


def test_latex_to_text_for_powerpoint():
    assert slides.latex_to_text(r"\lambda\mid k\sim\text{Gamma}(\alpha+k,\;\beta+t)") == "λ| k~Gamma(α+k, β+t)"
    assert "√" in slides.latex_to_text(r"\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}")


def test_kicker_uses_a_slash_separator():
    assert 'class="sep">/<' in slides.kick("Part 1 · regression")


def test_font_names_with_spaces_are_quoted():
    assert css_family(["Times New Roman", "serif"]) == "'Times New Roman', serif"
    assert DECK["mono"], "theme defines a font list"


def test_single_file_never_closes_a_script_early():
    assert "</script" not in single_file.safe_js("var a = '</script>';")
    assert "</" not in single_file.js_string("<b>x</b>")


def test_poisson_win_probability_matches_scoreline_sum():
    """The El Clasico slide adds Poisson scoreline probabilities; check the maths on a small case by hand."""
    from scipy import stats
    lh, la = 1.7, 1.4
    p = sum(stats.poisson.pmf(i, lh) * stats.poisson.pmf(j, la) for i in range(30) for j in range(30) if i > j)
    skellam = 1 - stats.skellam.cdf(0, lh, la)
    assert math.isclose(p, skellam, rel_tol=1e-9)
