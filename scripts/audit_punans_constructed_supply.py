"""Count Wikidata's supply for draft 45's constructed P-UNANS questions (Study 002).

Draft 45 proposes questions built from Wikidata entities (CC0): future winners of recurring competitions, and
future dated measurements in named cities. This script records, at one retrieval date, how many entities each
family could draw on, and why a third family (open problems) is not proposed. Wikidata changes daily, so the
artifact is a dated record, not a pin: the builder, if 45 is adopted, snapshots its own entities and pins that
snapshot by sha256.

Counts only. It never writes an entity label into the artifact.

Usage:
    python scripts/audit_punans_constructed_supply.py   # query Wikidata and write the dated record
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/source-screening/study-002-punans-constructed/wikidata-supply.json"
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "OpenGrad-screening/0.1 (research; https://github.com/arjhinety/OpenGrad)"
QUERIES = {
    "competition_series_with_5_editions_naming_a_winner": (
        "SELECT (COUNT(*) AS ?n) WHERE { { SELECT ?series WHERE { ?ed wdt:P3450 ?series ; wdt:P1346 ?w . } "
        "GROUP BY ?series HAVING (COUNT(?ed) >= 5) } }"
    ),
    "competition_series_with_a_winner_named_since_2021": (
        "SELECT (COUNT(DISTINCT ?series) AS ?n) WHERE { ?ed wdt:P3450 ?series ; wdt:P1346 ?w ; wdt:P585 ?t . "
        "FILTER(YEAR(?t) >= 2021) }"
    ),
    "cities_over_one_million_with_a_population_figure": (
        "SELECT (COUNT(DISTINCT ?c) AS ?n) WHERE { ?c wdt:P31/wdt:P279* wd:Q515 ; wdt:P1082 ?p . "
        "FILTER(?p > 1000000) }"
    ),
    "conjectures": "SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE { ?x wdt:P31 wd:Q319141 . }",
    "conjectures_recording_who_proved_them": (
        "SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE { ?x wdt:P31 wd:Q319141 ; wdt:P1318 ?proved . }"
    ),
}


def count(query: str) -> int:
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'query': query})}"
    request = urllib.request.Request(
        url, headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        rows = json.loads(response.read())["results"]["bindings"]
    return int(rows[0]["n"]["value"]) if rows else 0


def main() -> int:
    record = {
        "screening": "study-002-punans-constructed",
        "built_by": "scripts/audit_punans_constructed_supply.py",
        "source": "Wikidata (CC0-1.0), SPARQL endpoint",
        "endpoint": ENDPOINT,
        "retrieved": dt.datetime.now(tz=dt.UTC).date().isoformat(),
        "note": "A dated record of supply, not a pin: Wikidata changes daily.",
        "queries": {
            name: {"sparql": query, "count": count(query)} for name, query in QUERIES.items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({k: v["count"] for k, v in record["queries"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
