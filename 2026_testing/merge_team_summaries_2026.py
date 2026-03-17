import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH_GLOB = "team_summaries_batch*_recent.json"
OUTPUT_RAW = ROOT / "pre_tourney_data" / "team_sum_2026.json"
OUTPUT_CLEAN = ROOT / "pre_tourney_data" / "team_sum_clean_2026.json"


def load_batches() -> list[dict]:
    items: list[dict] = []
    seen: set[int] = set()

    for path in sorted((ROOT / "2026_testing").glob(BATCH_GLOB)):
        batch = json.loads(path.read_text())
        for item in batch:
            team_id = int(item["team_id"])
            if team_id in seen:
                raise ValueError(f"Duplicate team_id {team_id} in {path}")
            seen.add(team_id)
            items.append(item)

    return items


def main() -> None:
    items = load_batches()
    raw = {str(item["team_id"]): item for item in items}
    clean = {str(item["team_id"]): item["summary"] for item in items}

    OUTPUT_RAW.write_text(json.dumps(raw, indent=2, ensure_ascii=True))
    OUTPUT_CLEAN.write_text(json.dumps(clean, indent=2, ensure_ascii=True))

    print(f"Wrote {len(items)} team summaries to {OUTPUT_RAW}")
    print(f"Wrote {len(items)} clean team summaries to {OUTPUT_CLEAN}")


if __name__ == "__main__":
    main()
