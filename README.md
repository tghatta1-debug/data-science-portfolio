# Trishanth Ghattamaneni Portfolio

Personal portfolio and exploratory data analysis project for DTSC 2301.

## Project question

How do average net price and six-year graduation rates compare among public four-year universities in North Carolina?

## Run the analysis

Set `COLLEGE_SCORECARD_API_KEY` in your shell, then run:

```bash
python3 analysis/fetch_college_scorecard.py
python3 analysis/analyze_college_scorecard.py
```

## Data source

U.S. Department of Education College Scorecard API. The analysis filters to North Carolina public institutions whose predominant award is a bachelor's degree.

## AI disclosure

OpenAI Codex was used to help organize the website, write Python scaffolding, and draft prose. Trishanth Ghattamaneni reviewed the work, ran the analysis, and is responsible for verifying the data, conclusions, and citations before submission.
