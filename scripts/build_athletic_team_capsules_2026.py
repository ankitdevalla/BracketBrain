import json
import re
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "bracket_summary.pdf"
OUTPUT_PATH = ROOT / "pre_tourney_data" / "athletic_team_capsules_2026.json"
OUTPUT_TEXT_PATH = ROOT / "pre_tourney_data" / "athletic_team_capsules_2026_extracted.txt"


PAGE_TREE_OBJECTS = [15756, 15757, 15758, 15759, 15760, 15761, 15762, 15763, 15764, 15765, 15766]

# Article header names normalized to repo naming where needed.
TEAM_ALIASES = [
    ("Duke", "Duke"),
    ("Arizona", "Arizona"),
    ("Michigan", "Michigan"),
    ("Florida", "Florida"),
    ("Connecticut", "Connecticut"),
    ("Purdue", "Purdue"),
    ("Iowa State", "Iowa St"),
    ("Houston", "Houston"),
    ("Michigan State", "Michigan St"),
    ("Gonzaga", "Gonzaga"),
    ("Virginia", "Virginia"),
    ("Illinois", "Illinois"),
    ("Kansas", "Kansas"),
    ("Arkansas", "Arkansas"),
    ("Alabama", "Alabama"),
    ("Nebraska", "Nebraska"),
    ("St. John's", "St John's"),
    ("Wisconsin", "Wisconsin"),
    ("Texas Tech", "Texas Tech"),
    ("Vanderbilt", "Vanderbilt"),
    ("Louisville", "Louisville"),
    ("BYU", "BYU"),
    ("Tennessee", "Tennessee"),
    ("North Carolina", "North Carolina"),
    ("UCLA", "UCLA"),
    ("Miami", "Miami FL"),
    ("Kentucky", "Kentucky"),
    ("Saint Mary's", "St Mary's CA"),
    ("Santa Clara", "Santa Clara"),
    ("Ohio State", "Ohio St"),
    ("Villanova", "Villanova"),
    ("Georgia", "Georgia"),
    ("Clemson", "Clemson"),
    ("TCU", "TCU"),
    ("Utah State", "Utah St"),
    ("Saint Louis", "St Louis"),
    ("Iowa", "Iowa"),
    ("UCF", "UCF"),
    ("Missouri", "Missouri"),
    ("Texas A&M", "Texas A&M"),
    ("South Florida", "South Florida"),
    ("Texas", "Texas"),
    ("NC State", "NC State"),
    ("Miami Ohio", "Miami OH"),
    ("SMU", "SMU"),
    ("VCU", "VCU"),
    ("Northern Iowa", "Northern Iowa"),
    ("High Point", "High Point"),
    ("Akron", "Akron"),
    ("McNeese", "McNeese St"),
    ("Cal Baptist", "Cal Baptist"),
    ("Hawaii", "Hawaii"),
    ("Hofstra", "Hofstra"),
    ("Troy", "Troy"),
    ("North Dakota State", "N Dakota St"),
    ("Kennesaw State", "Kennesaw"),
    ("Wright State", "Wright St"),
    ("Penn", "Penn"),
    ("Furman", "Furman"),
    ("Queens", "Queens NC"),
    ("Tennessee State", "Tennessee St"),
    ("Idaho", "Idaho"),
    ("Siena", "Siena"),
    ("Long Island", "LIU Brooklyn"),
    ("UMBC", "UMBC"),
    ("Howard", "Howard"),
    ("Prairie View A&M", "Prairie View"),
    ("Lehigh", "Lehigh"),
]


def load_pdf_objects(pdf_bytes: bytes) -> dict[int, bytes]:
    objects: dict[int, bytes] = {}
    for match in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", pdf_bytes):
        obj_num = int(match.group(1))
        start = match.end()
        end = pdf_bytes.find(b"endobj", start)
        if end != -1:
            objects[obj_num] = pdf_bytes[start:end]
    return objects


def decompress_stream(obj_body: bytes) -> bytes:
    match = re.search(rb"stream\r?\n(.*)endstream", obj_body, re.S)
    if not match:
        return b""
    return zlib.decompress(match.group(1).rstrip(b"\r\n"))


def parse_cmap(cmap_bytes: bytes) -> dict[int, str]:
    text = cmap_bytes.decode("latin1")
    cmap: dict[int, str] = {}

    for block in re.finditer(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-F]+)>\s*<([0-9A-F]+)>", block.group(1)):
            cmap[int(src, 16)] = bytes.fromhex(dst).decode("utf-16-be")

    for block in re.finditer(r"beginbfrange(.*?)endbfrange", text, re.S):
        for line in block.group(1).splitlines():
            line = line.strip()
            direct = re.match(r"<([0-9A-F]+)>\s*<([0-9A-F]+)>\s*<([0-9A-F]+)>", line)
            if direct:
                start_code, end_code, start_char = [int(x, 16) for x in direct.groups()]
                for offset, code in enumerate(range(start_code, end_code + 1)):
                    cmap[code] = chr(start_char + offset)
                continue

            array_range = re.match(r"<([0-9A-F]+)>\s*<([0-9A-F]+)>\s*\[(.*)\]", line)
            if array_range:
                start_code, end_code, raw_targets = array_range.groups()
                targets = re.findall(r"<([0-9A-F]+)>", raw_targets)
                for code, dst in zip(range(int(start_code, 16), int(end_code, 16) + 1), targets):
                    cmap[code] = bytes.fromhex(dst).decode("utf-16-be")

    return cmap


def build_font_maps(objects: dict[int, bytes]) -> dict[int, dict[int, str]]:
    font_maps: dict[int, dict[int, str]] = {}
    for obj_num, body in objects.items():
        match = re.search(rb"/ToUnicode\s+(\d+)\s+0\s+R", body)
        if not match:
            continue
        try:
            cmap_bytes = decompress_stream(objects[int(match.group(1))])
            font_maps[obj_num] = parse_cmap(cmap_bytes)
        except Exception:
            continue
    return font_maps


def ordered_page_objects(objects: dict[int, bytes]) -> list[int]:
    page_objects: list[int] = []
    for tree_obj in PAGE_TREE_OBJECTS:
        body = objects[tree_obj]
        kids_match = re.search(rb"/Kids \[(.*?)\]", body, re.S)
        if not kids_match:
            continue
        page_objects.extend(int(x) for x in re.findall(rb"(\d+)\s+0\s+R", kids_match.group(1)))
    return page_objects


def extract_page_text(page_obj: int, objects: dict[int, bytes], font_maps: dict[int, dict[int, str]]) -> str:
    body = objects[page_obj]
    contents_obj = int(re.search(rb"/Contents\s+(\d+)\s+0\s+R", body).group(1))
    font_block = re.search(rb"/Font\s*<<(.*?)>>", body, re.S)
    fonts = {
        name.decode(): int(obj_num)
        for name, obj_num in re.findall(rb"/([A-Za-z0-9]+)\s+(\d+)\s+0\s+R", font_block.group(1))
    }

    content = decompress_stream(objects[contents_obj]).decode("latin1", "ignore")
    output: list[str] = []
    current_font: str | None = None

    token_pattern = re.compile(r"/F\d+\s+[0-9.]+\s+Tf|<([0-9A-F]+)>\s*Tj|\[(.*?)\]\s*TJ|ET", re.S)
    for token in token_pattern.finditer(content):
        raw = token.group(0)
        if " Tf" in raw:
            current_font = raw.split()[0][1:]
            output.append("\n")
            continue

        if raw == "ET":
            output.append("\n")
            continue

        font_map = font_maps.get(fonts.get(current_font or "", -1), {})

        if raw.endswith(" Tj"):
            hex_string = token.group(1)
            output.append("".join(font_map.get(int(hex_string[i : i + 4], 16), "") for i in range(0, len(hex_string), 4)))
            continue

        if raw.endswith(" TJ"):
            segment = []
            for hex_string in re.findall(r"<([0-9A-F]+)>", token.group(2)):
                segment.extend(font_map.get(int(hex_string[i : i + 4], 16), "") for i in range(0, len(hex_string), 4))
            output.append("".join(segment))

    text = re.sub(r"\n+", "\n", "".join(output)).strip()
    text = re.sub(
        r"Sports Betting\nNFL Picks.*?https://www\.nytimes\.com/athletic/7090849/2026/03/15/mens-march-madness-team-preview-big-board/\n\d+\n/\n84\n?",
        "",
        text,
        flags=re.S,
    )
    return text.strip()


def extract_full_text() -> str:
    pdf_bytes = PDF_PATH.read_bytes()
    objects = load_pdf_objects(pdf_bytes)
    font_maps = build_font_maps(objects)
    page_objects = ordered_page_objects(objects)
    full_text = "\n".join(extract_page_text(page_obj, objects, font_maps) for page_obj in page_objects)
    return full_text.strip()


def find_team_headers(full_text: str) -> list[tuple[int, str, str]]:
    headers: list[tuple[int, str, str]] = []
    for article_name, team_name in TEAM_ALIASES:
        for match in re.finditer(rf"(?m)^{re.escape(article_name)}$", full_text):
            headers.append((match.start(), article_name, team_name))
    headers.sort()

    seen: set[str] = set()
    unique_headers: list[tuple[int, str, str]] = []
    for pos, article_name, team_name in headers:
        if team_name in seen:
            continue
        seen.add(team_name)
        unique_headers.append((pos, article_name, team_name))

    return unique_headers


def clean_text(text: str) -> str:
    text = text.replace("ʼ", "'").replace("ʻ", "'").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def sentence_without_numbers(sentence: str) -> str:
    sentence = re.sub(r"\([^)]*?\d[^)]*?\)", "", sentence)
    sentence = re.sub(r"\b\d+(?:\.\d+)?%?\b", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip(" ,;")
    return sentence


PLAY_STYLE_TERMS = [
    "transition",
    "tempo",
    "pace",
    "paint",
    "rim",
    "post",
    "pick-and-roll",
    "pick and roll",
    "ball screen",
    "halfcourt",
    "half-court",
    "spacing",
    "rebound",
    "rebounding",
    "turnover",
    "perimeter",
    "arc",
    "frontcourt",
    "backcourt",
    "guards",
    "defense",
    "offense",
    "shoot",
    "shooting",
    "switch",
    "drop",
    "zone",
    "press",
    "athletic",
    "size",
    "depth",
]

NON_NAME_SECOND_TOKENS = {"The", "A", "An", "And", "But", "Or", "To", "For", "Of", "In", "On", "At", "By", "With"}


def soft_clean_sentence(sentence: str) -> str:
    sentence = clean_text(sentence)
    sentence = re.sub(r"\([^)]*?\d[^)]*?\)", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip(" ,;")


def sentence_score(sentence: str) -> int:
    score = 0
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", sentence):
        score += 3
    lowered = sentence.lower()
    score += sum(1 for term in PLAY_STYLE_TERMS if term in lowered)
    if any(token in lowered for token in [" featuring ", " led by ", " anchored by ", " frontcourt ", " backcourt "]):
        score += 1
    return score


def has_player_name(sentence: str) -> bool:
    for first, second in re.findall(r"\b([A-Z][A-Za-z'.-]+) ([A-Z][A-Za-z'.-]+)\b", sentence):
        if second in NON_NAME_SECOND_TOKENS:
            continue
        if len(first) < 3 or len(second) < 2:
            continue
        return True
    return False


def choose_key_sentences(text: str, limit: int = 2) -> list[str]:
    sentences = [soft_clean_sentence(s) for s in split_sentences(text)]
    sentences = [s for s in sentences if s]
    if not sentences:
        return []

    scored = [(idx, sentence_score(sentence), sentence) for idx, sentence in enumerate(sentences)]
    scored.sort(key=lambda item: (-item[1], item[0]))
    chosen = scored[:limit]

    if any(has_player_name(sentence) for sentence in sentences) and not any(has_player_name(sentence) for _, _, sentence in chosen):
        named_candidates = [item for item in scored if has_player_name(item[2])]
        if named_candidates:
            if chosen:
                chosen[-1] = named_candidates[0]
            else:
                chosen = [named_candidates[0]]

    deduped = []
    seen_sentences = set()
    for item in chosen:
        if item[2] in seen_sentences:
            continue
        seen_sentences.add(item[2])
        deduped.append(item)

    target_len = min(limit, len(sentences))
    if len(deduped) < target_len:
        for item in scored:
            if item[2] in seen_sentences:
                continue
            seen_sentences.add(item[2])
            deduped.append(item)
            if len(deduped) == target_len:
                break

    chosen = sorted(deduped, key=lambda item: item[0])
    return [sentence for _, _, sentence in chosen]


def tidy_lead(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(where to begin\?\s*)", "", text, flags=re.I)
    text = re.sub(r"^(for the sake of nitpicking,\s*)", "", text, flags=re.I)
    text = re.sub(r"^(candidly,\s*)", "", text, flags=re.I)
    text = re.sub(r"^(this scribe is a buyer that\s*)", "", text, flags=re.I)
    text = re.sub(r"^(is\s+)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;")


def summarize_field(team_name: str, text: str, kind: str) -> str:
    sentences = choose_key_sentences(text)
    if not sentences:
        return ""

    lead = tidy_lead(sentences[0])
    tail = sentences[1] if len(sentences) > 1 else ""

    if kind == "strengths":
        lead = re.sub(rf"^(The\s+)?{re.escape(team_name)}\s+(is|are)\s+", "", lead, flags=re.I)
        summary = lead.rstrip(".") + "."
        if tail:
            summary += " " + tail
        return summary

    if kind == "weaknesses":
        lead = re.sub(rf"^(The\s+)?{re.escape(team_name)}\s+", "", lead, flags=re.I)
        lead = tidy_lead(lead)
        summary = f"The concern for {team_name} is {lead[:1].lower() + lead[1:]}" if lead else f"The concern for {team_name} is its margin for error."
        summary = summary.rstrip(".") + "."
        if tail:
            summary += " " + tail
        return summary

    lead = re.sub(rf"^{re.escape(team_name)}\s+(is|are)\s+", "", lead, flags=re.I)
    lead = tidy_lead(lead)
    summary = f"The outlook for {team_name}: {lead[:1].lower() + lead[1:]}" if lead else f"The outlook for {team_name}: still in flux."
    summary = summary.rstrip(".") + "."
    if tail:
        summary += " " + tail
    return summary


def infer_strengths_from_outlook(team_name: str, outlook: str) -> str:
    positive_sentences = []
    for sentence in split_sentences(outlook):
        lowered = sentence.lower()
        if any(word in lowered for word in ["can", "capable", "dangerous", "strong", "shoot", "defense", "offense", "depth", "guards", "frontcourt", "rebound", "experience", "momentum"]):
            positive_sentences.append(sentence_without_numbers(sentence))
    return summarize_field(team_name, " ".join(positive_sentences), "strengths")


def infer_weaknesses_from_outlook(team_name: str, outlook: str) -> str:
    negative_sentences = []
    for sentence in split_sentences(outlook):
        lowered = sentence.lower()
        if any(word in lowered for word in ["however", "but", "unlikely", "lack", "problem", "concern", "flaw", "depth", "size", "defense", "rebounding", "underdog", "early exit", "overmatched"]):
            negative_sentences.append(sentence_without_numbers(sentence))
    if not negative_sentences:
        return f"The concern with {team_name} is that the PDF's capsule leans more on upside than on one clean flaw, so the bigger question is how well its formula scales against stronger tournament athletes."
    return summarize_field(team_name, " ".join(negative_sentences), "weaknesses")


def append_named_sentence_if_needed(summary: str, source_text: str) -> str:
    if has_player_name(summary):
        return summary

    for sentence in split_sentences(source_text):
        cleaned = soft_clean_sentence(sentence)
        if cleaned and has_player_name(cleaned) and cleaned not in summary:
            return f"{summary} {cleaned}".strip()

    return summary


def build_team_capsules(full_text: str) -> list[dict]:
    headers = find_team_headers(full_text)
    if len(headers) != len(TEAM_ALIASES):
        raise ValueError(f"Expected {len(TEAM_ALIASES)} team headers, found {len(headers)}")

    block_headers: list[tuple[int, str, str]] = []
    prev_team_pos = 0
    for team_pos, article_name, team_name in headers:
        window = full_text[prev_team_pos:team_pos]
        lowdown_idx = window.rfind("LaTulip")
        record_idx = window.rfind("Record:")
        if lowdown_idx != -1:
            block_start = prev_team_pos + lowdown_idx
        elif record_idx != -1:
            block_start = prev_team_pos + record_idx
        else:
            block_start = team_pos
        block_headers.append((block_start, article_name, team_name))
        prev_team_pos = team_pos

    records = []
    for idx, (start, article_name, team_name) in enumerate(block_headers):
        end = block_headers[idx + 1][0] if idx + 1 < len(block_headers) else len(full_text)
        block = full_text[start:end]

        strengths_match = re.search(r"Strengths:\s*(.*?)\nWeaknesses:", block, re.S)
        weaknesses_match = re.search(r"Weaknesses:\s*(.*?)\nOutlook:", block, re.S)
        outlook_match = re.search(r"Outlook:\s*(.*?)(?:\n-[^\n]+|$)", block, re.S)

        raw_strengths = clean_text(strengths_match.group(1)) if strengths_match else ""
        raw_weaknesses = clean_text(weaknesses_match.group(1)) if weaknesses_match else ""
        raw_outlook = clean_text(outlook_match.group(1)) if outlook_match else ""

        strengths = summarize_field(team_name, raw_strengths, "strengths") if raw_strengths else infer_strengths_from_outlook(team_name, raw_outlook)
        weaknesses = summarize_field(team_name, raw_weaknesses, "weaknesses") if raw_weaknesses else infer_weaknesses_from_outlook(team_name, raw_outlook)
        outlook = summarize_field(team_name, raw_outlook, "outlook")

        strengths = append_named_sentence_if_needed(strengths, raw_strengths or raw_outlook)
        outlook = append_named_sentence_if_needed(outlook, raw_outlook)

        records.append(
            {
                "team_name": team_name,
                "article_header_name": article_name,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "outlook": outlook,
                "source_excerpt": {
                    "strengths": raw_strengths,
                    "weaknesses": raw_weaknesses,
                    "outlook": raw_outlook,
                },
            }
        )

    return records


def main() -> None:
    full_text = extract_full_text()
    OUTPUT_TEXT_PATH.write_text(full_text)

    records = build_team_capsules(full_text)
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=True))

    print(f"Wrote extracted text to {OUTPUT_TEXT_PATH}")
    print(f"Wrote {len(records)} team capsules to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
