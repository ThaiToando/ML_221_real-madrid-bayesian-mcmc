# One image for the whole project: serve the deck, run the Streamlit app, or rebuild everything from the raw data.
#   docker compose up deck app                         -> http://localhost:8000 (deck), http://localhost:8501 (app)
#   docker compose --profile pipeline run --rm pipeline -> download the data and rebuild figures + deck
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg

# ffmpeg: the matplotlib animations as mp4; Liberation Serif: a Times New Roman match for the charts
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first, so code edits do not reinstall them (Manim and Playwright stay optional)
COPY requirements.txt .
RUN grep -v -E '^(manim|playwright)' requirements.txt > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt

COPY . .

RUN useradd --create-home analyst && chown -R analyst /app
USER analyst

EXPOSE 8000 8501
CMD ["python", "-m", "http.server", "8000", "--directory", "docs"]
