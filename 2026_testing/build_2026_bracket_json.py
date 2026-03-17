import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRACKET_CSV = ROOT / "2026_testing" / "bracket.csv"
MTEAMS_CSV = ROOT / "raw_data" / "MTeams.csv"
OUTPUT_JSON = ROOT / "pre_tourney_data" / "2026_Bracket.json"


def load_team_ids() -> dict[str, int]:
    with MTEAMS_CSV.open(newline="") as f:
        return {row["TeamName"]: int(row["TeamID"]) for row in csv.DictReader(f)}


def expand_team_names(raw_name: str) -> list[str]:
    return [name.strip() for name in raw_name.split("/")]


def main() -> None:
    team_ids = load_team_ids()
    bracket_data: dict[str, dict[str, int | str]] = {}

    with BRACKET_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            seed = int(row["Seed"])
            region = row["Region"]

            for team_name in expand_team_names(row["Team"]):
                if team_name not in team_ids:
                    raise KeyError(f"Team name not found in MTeams.csv: {team_name}")

                team_id = str(team_ids[team_name])
                bracket_data[team_id] = {"region": region, "seed": seed}

    OUTPUT_JSON.write_text(json.dumps(dict(sorted(bracket_data.items())), indent=2, ensure_ascii=True))
    print(f"Wrote {len(bracket_data)} teams to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
