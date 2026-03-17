import json
import sys
from pathlib import Path


def render_batch(json_path: Path) -> Path:
    items = json.loads(json_path.read_text())
    lines = [f"# {json_path.stem}", ""]

    for item in items:
        lines.append(f"## {item['team_name']} ({item['team_id']})")
        lines.append("")
        lines.append(item["summary"])
        lines.append("")
        lines.append("Sources:")
        for url in item["source_urls"]:
            lines.append(f"- {url}")
        lines.append("")

    output_path = json_path.with_suffix(".md")
    output_path.write_text("\n".join(lines))
    return output_path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: render_team_summary_batch_markdown.py <batch.json> [<batch.json> ...]")

    for raw_path in sys.argv[1:]:
        output_path = render_batch(Path(raw_path))
        print(output_path)


if __name__ == "__main__":
    main()
