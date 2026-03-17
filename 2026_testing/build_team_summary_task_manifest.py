import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BRACKET_PATH = ROOT / "2026_testing" / "bracket.csv"
TEAMS_PATH = ROOT / "raw_data" / "MTeams.csv"
OUTPUT_JSON_PATH = ROOT / "2026_testing" / "team_summary_news_tasks_2026.json"
OUTPUT_MD_PATH = ROOT / "2026_testing" / "team_summary_batches_2026.md"

DATE_START = "2026-02-15"
DATE_END = "2026-03-16"

NEWS_DOMAINS = [
    "apnews.com",
    "espn.com",
    "cbssports.com",
    "usatoday.com",
    "foxsports.com",
    "si.com",
    "sportingnews.com",
    "theathletic.com",
]


def load_team_ids():
    with TEAMS_PATH.open(newline="") as handle:
        return {row["TeamName"]: int(row["TeamID"]) for row in csv.DictReader(handle)}


def expand_bracket_rows():
    rows = []
    with BRACKET_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            team_names = [name.strip() for name in row["Team"].split("/")]
            for team_name in team_names:
                rows.append(
                    {
                        "team_name": team_name,
                        "seed": int(row["Seed"]),
                        "region": row["Region"],
                    }
                )
    return rows


def build_task(team_row, team_ids):
    team_name = team_row["team_name"]
    team_id = team_ids.get(team_name)
    if team_id is None:
        raise KeyError(f"Team not found in MTeams.csv: {team_name}")

    search_queries = [
        f'"{team_name}" men\'s basketball 2025 2026 season recap March 2026',
        f'"{team_name}" basketball 2025 2026 NCAA tournament profile March 2026',
        f'"{team_name}" basketball 2025 2026 news injuries coach stars March 2026',
    ]

    return {
        "team_id": team_id,
        "team_name": team_name,
        "seed": team_row["seed"],
        "region": team_row["region"],
        "season_window": {
            "start_date": DATE_START,
            "end_date": DATE_END,
        },
        "source_rules": {
            "allowed_domains": NEWS_DOMAINS,
            "minimum_articles": 3,
            "maximum_articles": 6,
            "prefer_recent_days": 21,
            "disallow": [
                "Wikipedia",
                "forums",
                "Reddit",
                "team roster pages",
                "game box scores without recap text",
                "articles published outside the 2025-2026 season window",
            ],
        },
        "search_queries": search_queries,
        "summary_requirements": {
            "tone": "reported scouting-note voice, like a college basketball reporter writing a bracket capsule",
            "length_sentences": "5-8",
            "must_cover": [
                "playing style",
                "best or most important players by name",
                "strengths",
                "weaknesses",
                "late-season form",
                "injuries or availability concerns when recently reported",
                "coach or program context when relevant",
                "specific offensive or defensive actions when clearly supported by reporting",
            ],
            "style_rules": [
                "sound like an informed reporter, not an encyclopedia entry",
                "allow light first-person observational phrasing such as 'what stands out is' or 'the thing I keep coming back to'",
                "mention at least 2 specific players when possible",
                "if the reporting supports it, mention actions like ball screens, post touches, transition attacks, drop coverage, switching, or three-point volume",
                "if the reporting mentions injuries, suspensions, or minute restrictions, explain the basketball impact briefly",
                "do not invent scheme details that are not supported by the sources",
            ],
            "output_fields": [
                "team_id",
                "team_name",
                "summary",
                "source_urls",
            ],
        },
    }


def write_batch_markdown(tasks):
    batch_size = 9
    lines = [
        "# 2026 Team Summary News Tasks",
        "",
        "Use these tasks to generate `team_sum`-style summaries from 2025-2026 season news only.",
        "",
        "Rules:",
        f"- Only use articles published between {DATE_START} and {DATE_END}.",
        f"- Only use news/reporting domains: {', '.join(NEWS_DOMAINS)}.",
        "- Ignore forums, social posts, roster pages, and generic stat-only pages.",
        "- Output one paragraph per team in the style of `pre_tourney_data/team_sum_clean.json`, but with a more reported scouting-note voice.",
        "- Use recent coverage only, with strong preference for the last 2-3 weeks before 2026-03-16.",
        "- Keep article evidence grounded in the late 2025-2026 season and cite 3-6 URLs per team.",
        "- Mention specific players by name, include concrete actions or schemes when the reporting supports them, and note real injury/availability context when recently reported.",
        "",
    ]

    for batch_index in range(0, len(tasks), batch_size):
        batch_number = batch_index // batch_size + 1
        batch = tasks[batch_index:batch_index + batch_size]
        lines.append(f"## Batch {batch_number}")
        lines.append("")
        lines.append("Teams:")
        for task in batch:
            lines.append(
                f"- {task['team_name']} ({task['region']} #{task['seed']}, TeamID {task['team_id']})"
            )
        lines.append("")
        lines.append("Prompt:")
        lines.append("")
        lines.append("```text")
        lines.append(
            "For each team below, search only recent 2025-2026 season news coverage and write a summary "
            "matching the tone and structure of `pre_tourney_data/team_sum_clean.json`, but in a more reported scouting-note voice."
        )
        lines.append(
            f"Use only articles published between {DATE_START} and {DATE_END} from these domains: "
            + ", ".join(NEWS_DOMAINS)
        )
        lines.append(
            "Strongly prefer articles from the last 2-3 weeks before 2026-03-16 and weight the most recent reporting highest."
        )
        lines.append(
            "Each team summary must be 5-8 sentences and cover playing style, key players, strengths, "
            "weaknesses, late-season form, and any meaningful injury or availability context. Return `team_id`, `team_name`, `summary`, and `source_urls`."
        )
        lines.append(
            "Write like a reporter who has been following the team closely: name specific players, use light observational phrasing, mention concrete actions or scheme details when supported by the reporting, and explain the impact of any recent injuries or absences."
        )
        lines.append("")
        lines.append("Teams:")
        for task in batch:
            lines.append(
                f"- {task['team_name']} | TeamID {task['team_id']} | {task['region']} #{task['seed']}"
            )
        lines.append("```")
        lines.append("")

    OUTPUT_MD_PATH.write_text("\n".join(lines))


def main():
    team_ids = load_team_ids()
    tasks = [build_task(row, team_ids) for row in expand_bracket_rows()]
    tasks.sort(key=lambda item: (item["region"], item["seed"], item["team_name"]))

    OUTPUT_JSON_PATH.write_text(json.dumps(tasks, indent=2))
    write_batch_markdown(tasks)

    print(f"Wrote {OUTPUT_JSON_PATH}")
    print(f"Wrote {OUTPUT_MD_PATH}")
    print(f"Task count: {len(tasks)}")


if __name__ == "__main__":
    main()
