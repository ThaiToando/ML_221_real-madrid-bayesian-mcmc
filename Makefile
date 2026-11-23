# Common tasks. On Windows without make, run the command after each target by hand (they are plain Python).
PY ?= python

.PHONY: help install dev data analysis deck videos pdf single readme test lint format check serve app docker-build docker-up docker-pipeline clean

help:            ## list the tasks
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:         ## runtime packages (no Manim)
	grep -v -E '^(manim)' requirements.txt > .requirements-core.txt && $(PY) -m pip install -r .requirements-core.txt && rm .requirements-core.txt

dev: install     ## plus lint and test tools, and the git hooks
	$(PY) -m pip install -r requirements-dev.txt && pre-commit install

data:            ## download the Transfermarkt tables (~200 MB) into data/raw
	$(PY) -c "from src.data_io import download_data, check_club_ids; download_data(); print(check_club_ids())"

analysis:        ## parts 1-7: tables, models, figures, results
	$(PY) run_all.py

deck:            ## analysis + interactive visuals + docs/index.html and deck.pptx
	$(PY) run_all.py --deck

videos:          ## the 8 Manim explainers (needs Manim, Pango and LaTeX)
	$(PY) -m src.manim_videos

pdf:             ## docs/deck.pdf (needs: playwright install chromium)
	$(PY) -m src.slides --pdf

single:          ## one self-contained HTML file for presenting anywhere
	$(PY) -m src.single_file

readme:          ## refresh the results table in README.md from data/processed
	$(PY) -m src.readme_results

check:           ## go-live check: READY TO PRESENT or what to fix
	$(PY) -m tests.check_real

test:            ## unit tests
	$(PY) -m pytest

lint:            ## static checks
	$(PY) -m ruff check .

format:          ## auto-fix what the linter can
	$(PY) -m ruff check . --fix

serve:           ## open the deck at http://localhost:8000
	$(PY) -m http.server 8000 --directory docs

app:             ## the Streamlit app at http://localhost:8501
	$(PY) -m streamlit run app/app.py

docker-build:    ## build the image
	docker compose build

docker-up:       ## deck on :8000 and app on :8501
	docker compose up deck app

docker-pipeline: ## rebuild everything inside Docker (downloads the data)
	docker compose --profile pipeline run --rm pipeline

clean:           ## caches only (never data or figures)
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ */__pycache__
