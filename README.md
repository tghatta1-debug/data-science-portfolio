# Trishanth Ghattamaneni Portfolio

Personal portfolio and data-science case studies for DTSC 2301.

## Project 1 — College cost and completion

**Question:** How do average net price and six-year graduation rates compare among public four-year universities in North Carolina?

Set `COLLEGE_SCORECARD_API_KEY` in your shell, then run:

```bash
python3 analysis/fetch_college_scorecard.py
python3 analysis/analyze_college_scorecard.py
```

Source: U.S. Department of Education College Scorecard API.

## Project 2 — Online purchase intention

**Question:** Can a customer's browsing behavior predict whether the session will end in a purchase?

The analysis compares a majority-class baseline, class-weighted logistic regression, and a tuned class-weighted random forest. It uses an 80/20 stratified holdout split, five-fold cross-validation on the training data, class-aware metrics, permutation importance, subgroup error checks, and a sensitivity analysis that removes `PageValues`.

Install the Python dependencies and reproduce every published result and figure:

```bash
python3 -m pip install -r requirements.txt
python3 analysis/analyze_online_shoppers.py
```

Source: Sakar, C., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q. The dataset is distributed under CC BY 4.0.

## AI disclosure

OpenAI Codex (GPT-5, October 2026) assisted with analysis scaffolding, code review, visualization styling, and drafting the Project Two case study. Trishanth Ghattamaneni reviewed the requirements, results, claims, and citations and remains responsible for the final work.
