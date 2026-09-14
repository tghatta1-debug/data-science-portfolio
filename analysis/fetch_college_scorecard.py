import json
import os
from pathlib import Path

import requests

api_key = os.environ["COLLEGE_SCORECARD_API_KEY"]
params = {
    "api_key": api_key,
    "school.state": "NC",
    "school.ownership": 1,
    "school.degrees_awarded.predominant": 3,
    "fields": ",".join([
        "id",
        "school.name",
        "latest.cost.avg_net_price.public",
        "latest.completion.rate_suppressed.overall",
        "latest.student.size",
    ]),
    "per_page": 100,
}
response = requests.get(
    "https://api.data.gov/ed/collegescorecard/v1/schools.json",
    params=params,
    timeout=30,
)
response.raise_for_status()
Path("data").mkdir(exist_ok=True)
Path("data/college_scorecard_nc_public_four_year.json").write_text(
    json.dumps(response.json(), indent=2), encoding="utf-8"
)
