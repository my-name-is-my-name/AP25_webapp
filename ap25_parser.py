from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pypdf import PdfReader


DEFAULT_DOCUMENT_METADATA = {
    "document_id": "ap25",
    "document_code": "АП-25",
    "document_title": "Авиационные правила. Нормы летной годности самолетов транспортной категории",
    "document_type": "Нормы летной годности",
}

ROMAN_MARKER_RE = re.compile(r"^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\*?$", re.IGNORECASE)
NUMERIC_MARKER_RE = re.compile(r"^\d+\*?$")
ALPHA_MARKER_RE = re.compile(r"^[a-z]\*?$")
MARKER_LINE_RE = re.compile(r"^\(([^)]+)\)\s*(.*)$")
PARAGRAPH_SUFFIX_PATTERN = r"[ABАВ]"
PARAGRAPH_WITH_SUFFIX_RE = re.compile(rf"^(25\.\d+)\s*({PARAGRAPH_SUFFIX_PATTERN})\.\s*(.*)$")
PARAGRAPH_RE = re.compile(r"^(25\.\d+)\.\s*(.*)$")
SECTION_RE = re.compile(r"^раздел\s+([A-Za-zА-Яа-я])\s*[–-]\s*(.+)$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\d+$")
RANGE_REF_RE = re.compile(rf"(25\.\d+(?:\s*{PARAGRAPH_SUFFIX_PATTERN})?)\s*[–-]\s*(25\.\d+(?:\s*{PARAGRAPH_SUFFIX_PATTERN})?)")
SINGLE_REF_RE = re.compile(rf"(?<![\d.])(25\.\d+(?:\s*{PARAGRAPH_SUFFIX_PATTERN})?)(?:\(([A-Za-zА-Яа-я0-9*]+)\))?")
LETTER_HYPHEN_RE = re.compile(r"(?<=[A-Za-zА-Яа-я])\s*-\s*(?=[A-Za-zА-Яа-я])")


@dataclass
class ParagraphBlock:
    paragraph_id: str
    paragraph_suffix: Optional[str]
    paragraph_key: str
    first_title_text: str
    heading_path: list[str]
    context_heading: str
    page_start: int
    page_end: int
    raw_lines: list[str] = field(default_factory=list)


def normalize_spaces(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def normalize_dashes(text: str) -> str:
    return normalize_spaces(text).replace("–", "-").replace("—", "-")


def normalize_heading(text: str) -> str:
    return normalize_dashes(text).upper()


def clean_line(text: str) -> str:
    return normalize_spaces(text)


def normalize_suffix(suffix: str) -> str:
    value = normalize_spaces(suffix).upper()
    if value == "А":
        return "A"
    if value == "В":
        return "B"
    if value == "С":
        return "C"
    if value == "Е":
        return "E"
    return value


def transliterate_marker(text: str) -> str:
    mapping = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "j",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "c",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ы": "y",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
    normalized = normalize_spaces(text).lower()
    return "".join(mapping.get(char, char) for char in normalized)


def canonical_paragraph_key(paragraph_id: str, paragraph_suffix: Optional[str]) -> str:
    if paragraph_suffix:
        return f"{paragraph_id}{normalize_suffix(paragraph_suffix)}"
    return paragraph_id


def paragraph_label(paragraph_key: str) -> str:
    return f"§{paragraph_key}"


def is_header_line(line: str) -> bool:
    normalized = normalize_heading(line)
    if not normalized:
        return False
    if "АВИАЦИОННЫЕ ПРАВИЛА" in normalized and "ЧАСТЬ 25" in normalized:
        return True
    if normalized in {"АВИАЦИОННЫЕ ПРАВИЛА", "ЧАСТЬ 25", "АВИАЦИОННЫЕ", "ПРАВИЛА"}:
        return True
    return False


def clean_page_lines(page_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in page_text.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue
        if PAGE_NUMBER_RE.fullmatch(line):
            continue
        if is_header_line(line):
            continue
        lines.append(line)
    return lines


def detect_section_heading(line: str) -> Optional[str]:
    match = SECTION_RE.match(normalize_dashes(line))
    if not match:
        return None
    section_code = normalize_suffix(match.group(1))
    section_title = normalize_heading(match.group(2))
    return f"РАЗДЕЛ {section_code} - {section_title}"


def is_upper_heading(line: str) -> bool:
    normalized = normalize_heading(line)
    if not normalized:
        return False
    if detect_section_heading(line):
        return False
    if PAGE_NUMBER_RE.fullmatch(normalized):
        return False
    if "...." in normalized:
        return False
    if PARAGRAPH_RE.match(normalized) or PARAGRAPH_WITH_SUFFIX_RE.match(normalized):
        return False
    if MARKER_LINE_RE.match(normalized):
        return False
    if any(char.islower() for char in line):
        return False
    alpha_count = sum(char.isalpha() for char in normalized)
    if alpha_count < 3:
        return False
    return True


def match_paragraph_start(line: str) -> Optional[tuple[str, Optional[str], str]]:
    normalized = normalize_spaces(line)
    match = PARAGRAPH_WITH_SUFFIX_RE.match(normalized)
    if match:
        paragraph_id = match.group(1)
        paragraph_suffix = normalize_suffix(match.group(2))
        title_part = match.group(3).strip()
        return paragraph_id, paragraph_suffix, title_part
    match = PARAGRAPH_RE.match(normalized)
    if match:
        paragraph_id = match.group(1)
        title_part = match.group(2).strip()
        return paragraph_id, None, title_part
    return None


def is_valid_paragraph_start(title_part: str, next_line: Optional[str]) -> bool:
    title_part = title_part.strip()
    next_line = next_line.strip() if next_line else None

    if title_part and is_line_marker(title_part):
        return False

    if not title_part:
        if next_line is None:
            return False
        if is_line_marker(next_line):
            return False
        if match_paragraph_start(next_line):
            return False
        if detect_section_heading(next_line) or is_upper_heading(next_line):
            return False
    return True


def is_line_marker(line: str) -> bool:
    marker_match = MARKER_LINE_RE.match(line)
    if not marker_match:
        return False
    return classify_marker(marker_match.group(1).strip()) != "other"


def normalize_marker(marker: str) -> str:
    return transliterate_marker(marker.replace(" ", ""))


def classify_marker(marker: str) -> str:
    normalized = normalize_marker(marker)
    if NUMERIC_MARKER_RE.fullmatch(normalized):
        return "numeric"
    if ROMAN_MARKER_RE.fullmatch(normalized):
        return "roman"
    if ALPHA_MARKER_RE.fullmatch(normalized):
        return "alpha"
    return "other"


def marker_level(marker: str) -> int:
    marker_kind = classify_marker(marker)
    if marker_kind == "alpha":
        return 0
    if marker_kind == "numeric":
        return 1
    return 2


def normalize_word_hyphen_spacing(text: str) -> str:
    return LETTER_HYPHEN_RE.sub("-", text)


def is_reserved_text(text: str) -> bool:
    normalized = normalize_spaces(text).lower()
    normalized = normalized.replace("[", "").replace("]", "").replace(".", "").replace(":", "")
    normalized = normalize_spaces(normalized)
    return normalized == "зарезервирован"


def join_lines(lines: list[str]) -> str:
    parts: list[str] = []
    for raw_line in lines:
        line = normalize_word_hyphen_spacing(clean_line(raw_line))
        if not line:
            continue
        if not parts:
            parts.append(line)
            continue
        previous = parts[-1]
        if re.search(r"[A-Za-zА-Яа-я]\s*-$", previous) and re.match(r"^[A-Za-zА-Яа-я]", line):
            previous = re.sub(r"\s*-$", "", previous)
            parts[-1] = previous + line
        else:
            parts.append(line)
    return normalize_spaces(" ".join(parts))


def probable_title_before_marker(line: str, next_line: Optional[str]) -> bool:
    if is_line_marker(line) or match_paragraph_start(line):
        return False
    if len(line) <= 50:
        return True
    if next_line and is_line_marker(next_line) and len(line) <= 90:
        return True
    return False


def probable_title_without_markers(line: str, next_line: Optional[str]) -> bool:
    if is_line_marker(line) or match_paragraph_start(line):
        return False
    if re.search(r"[.:;!?]$", line):
        return False
    if len(line) > 30:
        return False
    if next_line and len(next_line) < len(line):
        return True
    return True


def split_title_and_body(first_title_text: str, raw_lines: list[str]) -> tuple[str, list[str]]:
    lines = [clean_line(line) for line in raw_lines if clean_line(line)]
    title_lines = [first_title_text] if first_title_text else []
    if not lines:
        return join_lines(title_lines), []

    first_marker_index = next((index for index, line in enumerate(lines) if is_line_marker(line)), None)

    if first_marker_index is not None:
        taken = 0
        for index, line in enumerate(lines[:first_marker_index]):
            next_line = lines[index + 1] if index + 1 < len(lines) else lines[first_marker_index]
            if probable_title_before_marker(line, next_line):
                title_lines.append(line)
                taken += 1
            else:
                break
        body_lines = lines[taken:]
    else:
        taken = 0
        for index, line in enumerate(lines):
            next_line = lines[index + 1] if index + 1 < len(lines) else None
            if probable_title_without_markers(line, next_line):
                title_lines.append(line)
                taken += 1
            else:
                break
        body_lines = lines[taken:]

    title = join_lines([line for line in title_lines if line])
    if not title and lines:
        title = lines[0]
        body_lines = lines[1:]
    return title, body_lines


def build_tree_from_body_lines(body_lines: list[str]) -> tuple[str, list[dict[str, Any]]]:
    intro_lines: list[str] = []
    entries: list[dict[str, Any]] = []
    current_entry: Optional[dict[str, Any]] = None

    for line in body_lines:
        marker_match = MARKER_LINE_RE.match(line)
        if marker_match and classify_marker(marker_match.group(1).strip()) != "other":
            marker = marker_match.group(1).strip()
            marker_type = classify_marker(marker)
            line_remainder = marker_match.group(2).strip()
            # If a line starts with something like "(b) данного параграфа"
            # or "(c) (1) данного параграфа", this is usually a wrapped
            # cross-reference inside the current alpha clause, not a new clause.
            if (
                current_entry is not None
                and marker_type == "alpha"
                and line_remainder
                and (line_remainder.startswith("(") or line_remainder[0].islower())
            ):
                current_entry["text_lines"].append(line)
                continue
            current_entry = {
                "marker": f"({marker})",
                "normalized_marker": normalize_marker(marker),
                "marker_type": marker_type,
                "level": marker_level(marker),
                "text_lines": [line_remainder] if line_remainder else [],
                "inline_items": [],
            }
            entries.append(current_entry)
            continue

        if current_entry is None:
            intro_lines.append(line)
        else:
            current_entry["text_lines"].append(line)

    roots: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []

    for entry in entries:
        node = {
            "marker": entry["marker"],
            "normalized_marker": entry["normalized_marker"],
            "marker_type": entry["marker_type"],
            "text": join_lines(entry["text_lines"]),
            "inline_items": [],
            "level": entry["level"],
        }
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            stack[-1]["inline_items"].append(node)
        else:
            roots.append(node)
        stack.append(node)

    return join_lines(intro_lines), roots


def flatten_inline_item(node: dict[str, Any], include_self_marker: bool = True) -> str:
    parts: list[str] = []
    if include_self_marker:
        if node["text"]:
            parts.append(f"{node['marker']} {node['text']}")
        else:
            parts.append(node["marker"])
    elif node["text"]:
        parts.append(node["text"])

    for child in node.get("inline_items", []):
        child_text = flatten_inline_item(child, include_self_marker=True)
        if child_text:
            parts.append(child_text)
    return normalize_spaces(" ".join(parts))


def render_clause_text(clause: dict[str, Any]) -> str:
    if clause["flattened_text"]:
        return f"{clause['marker']} {clause['flattened_text']}"
    return clause["marker"]


def render_clause_group_text(intro_text: str, clauses: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if intro_text:
        parts.append(intro_text)
    parts.extend(render_clause_text(clause) for clause in clauses)
    return normalize_spaces(" ".join(part for part in parts if part))


def paragraph_stem_for_retrieval(paragraph: dict[str, Any], prefer_title: bool = False) -> str:
    if paragraph["intro_text"]:
        return paragraph["intro_text"]
    title = paragraph["paragraph_title"]
    if title and (prefer_title or (paragraph["clauses"] and len(title) > 80)):
        return title
    return ""


def build_retrieval_group_text(
    paragraph: dict[str, Any],
    clauses: list[dict[str, Any]],
    min_informative_chars: int,
) -> str:
    primary_stem = paragraph_stem_for_retrieval(paragraph, prefer_title=False)
    text = render_clause_group_text(primary_stem, clauses)

    if len(text) >= min_informative_chars:
        return text

    fallback_stem = paragraph_stem_for_retrieval(paragraph, prefer_title=True)
    if fallback_stem and fallback_stem != primary_stem:
        return render_clause_group_text(fallback_stem, clauses)
    return text


def serialize_inline_item(node: dict[str, Any]) -> dict[str, Any]:
    item = {
        "marker": node["marker"],
        "normalized_marker": node["normalized_marker"],
        "text": node["text"],
    }
    if node.get("inline_items"):
        item["inline_items"] = [serialize_inline_item(child) for child in node["inline_items"]]
    return item


def paragraph_sort_key(paragraph_key: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(25)\.(\d+)([A-Z]?)", paragraph_key)
    if not match:
        return (25, 0, 0)
    major = int(match.group(1))
    minor = int(match.group(2))
    suffix = match.group(3)
    suffix_index = 0 if not suffix else (ord(suffix) - ord("A") + 1)
    return (major, minor, suffix_index)


def normalize_reference_key(reference: str) -> str:
    normalized = normalize_dashes(reference).replace(" ", "")
    match = re.fullmatch(rf"(25\.\d+)({PARAGRAPH_SUFFIX_PATTERN}?)", normalized)
    if not match:
        return normalized
    suffix = normalize_suffix(match.group(2)) if match.group(2) else ""
    return f"{match.group(1)}{suffix}"


def build_paragraph_registry(paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry = []
    for paragraph in paragraphs:
        registry.append(
            {
                "paragraph_key": paragraph["paragraph_key"],
                "paragraph_id": paragraph["paragraph_id"],
                "paragraph_suffix": paragraph["paragraph_suffix"],
                "paragraph_label": paragraph["paragraph_label"],
                "paragraph_title": paragraph["paragraph_title"],
                "sort_key": paragraph_sort_key(paragraph["paragraph_key"]),
                "page_start": paragraph["page_start"],
                "page_end": paragraph["page_end"],
            }
        )
    registry.sort(key=lambda item: item["sort_key"])
    return registry


def expand_range(start_key: str, end_key: str, registry: list[dict[str, Any]]) -> list[str]:
    start_sort = paragraph_sort_key(start_key)
    end_sort = paragraph_sort_key(end_key)
    low, high = sorted([start_sort, end_sort])
    return [
        item["paragraph_key"]
        for item in registry
        if low <= item["sort_key"] <= high
    ]


def extract_cross_references(text: str, registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = normalize_dashes(text)
    references: list[dict[str, Any]] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in RANGE_REF_RE.finditer(source):
        start_key = normalize_reference_key(match.group(1))
        end_key = normalize_reference_key(match.group(2))
        references.append(
            {
                "reference_type": "range",
                "surface_form": match.group(0),
                "range_start": start_key,
                "range_end": end_key,
                "expanded_targets": expand_range(start_key, end_key, registry),
            }
        )
        occupied_spans.append(match.span())

    def inside_range_span(span: tuple[int, int]) -> bool:
        for start, end in occupied_spans:
            if start <= span[0] and span[1] <= end:
                return True
        return False

    for match in SINGLE_REF_RE.finditer(source):
        if inside_range_span(match.span()):
            continue
        target_key = normalize_reference_key(match.group(1))
        clause_marker = match.group(2)
        item = {
            "reference_type": "single",
            "surface_form": match.group(0),
            "target_paragraph_key": target_key,
        }
        if clause_marker:
            item["target_clause_marker"] = f"({clause_marker})"
        references.append(item)

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reference in references:
        fingerprint = json.dumps(reference, ensure_ascii=False, sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(reference)
    return unique


def extract_document_metadata(pdf_path: str) -> dict[str, Any]:
    metadata = dict(DEFAULT_DOCUMENT_METADATA)
    metadata["source_file"] = str(Path(pdf_path).resolve())
    return metadata


def collect_paragraph_blocks(reader: PdfReader) -> list[ParagraphBlock]:
    blocks: list[ParagraphBlock] = []
    current_section = ""
    current_context_heading = ""
    pending_heading_lines: list[str] = []
    current_block: Optional[ParagraphBlock] = None

    def flush_pending_heading() -> None:
        nonlocal current_context_heading, pending_heading_lines
        if not pending_heading_lines:
            return
        current_context_heading = join_lines(pending_heading_lines)
        pending_heading_lines = []

    def finalize_current_block() -> None:
        nonlocal current_block
        if current_block is not None:
            blocks.append(current_block)
            current_block = None

    for page_number, page in enumerate(reader.pages, start=1):
        page_lines = clean_page_lines(page.extract_text() or "")
        for index, line in enumerate(page_lines):
            next_line = page_lines[index + 1] if index + 1 < len(page_lines) else None
            section_heading = detect_section_heading(line)
            if section_heading:
                finalize_current_block()
                pending_heading_lines = []
                current_section = section_heading
                current_context_heading = ""
                continue

            if is_upper_heading(line):
                finalize_current_block()
                pending_heading_lines.append(normalize_heading(line))
                continue

            flush_pending_heading()

            paragraph_match = match_paragraph_start(line)
            if paragraph_match:
                paragraph_id, paragraph_suffix, title_part = paragraph_match
                if not is_valid_paragraph_start(title_part, next_line):
                    if current_block is not None:
                        current_block.raw_lines.append(line)
                        current_block.page_end = page_number
                    continue
                finalize_current_block()
                key = canonical_paragraph_key(paragraph_id, paragraph_suffix)
                heading_path = [current_section] if current_section else []
                context_heading = current_context_heading or current_section
                if current_context_heading and current_context_heading != current_section:
                    heading_path.append(current_context_heading)
                current_block = ParagraphBlock(
                    paragraph_id=paragraph_id,
                    paragraph_suffix=paragraph_suffix,
                    paragraph_key=key,
                    first_title_text=title_part,
                    heading_path=heading_path,
                    context_heading=context_heading,
                    page_start=page_number,
                    page_end=page_number,
                )
                continue

            if current_block is not None:
                current_block.raw_lines.append(line)
                current_block.page_end = page_number

    if current_block is not None:
        blocks.append(current_block)
    return blocks


def build_paragraph_record(block: ParagraphBlock) -> dict[str, Any]:
    title, body_lines = split_title_and_body(block.first_title_text, block.raw_lines)
    intro_text, roots = build_tree_from_body_lines(body_lines)
    clauses: list[dict[str, Any]] = []

    clause_index = 0
    for root in roots:
        if root["marker_type"] != "alpha":
            continue
        clause_index += 1
        flattened_text = flatten_inline_item(root, include_self_marker=False)
        if is_reserved_text(flattened_text or root["text"]):
            continue
        clauses.append(
            {
                "clause_id": f"{block.paragraph_key}::clause_{clause_index:02d}",
                "clause_index": clause_index,
                "marker": root["marker"],
                "normalized_marker": root["normalized_marker"],
                "text": root["text"],
                "inline_items": [serialize_inline_item(child) for child in root["inline_items"]],
                "flattened_text": flattened_text,
            }
        )

    if clauses:
        body_text = normalize_spaces(
            " ".join(
                [intro_text] + [f"{clause['marker']} {clause['flattened_text']}" for clause in clauses]
            )
        )
    else:
        if roots:
            body_text = normalize_spaces(
                " ".join([intro_text] + [flatten_inline_item(root, include_self_marker=True) for root in roots])
            )
        else:
            body_text = intro_text

    full_text = normalize_spaces(" ".join(part for part in [title, body_text] if part))

    paragraph = {
        "paragraph_id": block.paragraph_id,
        "paragraph_suffix": block.paragraph_suffix,
        "paragraph_key": block.paragraph_key,
        "paragraph_label": paragraph_label(block.paragraph_key),
        "paragraph_title": title,
        "heading_path": block.heading_path,
        "context_heading": block.context_heading,
        "page_start": block.page_start,
        "page_end": block.page_end,
        "intro_text": intro_text,
        "body_text": body_text,
        "full_text": full_text,
        "clauses": clauses,
    }

    if not clauses and roots:
        paragraph["inline_items"] = [serialize_inline_item(root) for root in roots]

    return paragraph


def paragraph_candidate_score(paragraph: dict[str, Any]) -> float:
    title = paragraph["paragraph_title"]
    normalized_title = normalize_heading(title)
    score = 0.0

    score += 20.0 if paragraph["clauses"] else 0.0
    score += 8.0 if paragraph["heading_path"] else 0.0
    score += 6.0 if len(title) <= 80 else -6.0
    score += 4.0 if not re.search(r"[:;]$", title) else -8.0
    score += 2.0 if paragraph["page_end"] >= paragraph["page_start"] else 0.0
    score -= len(title) * 0.05

    suspicious_prefixes = (
        "ПРИ ЭТОМ",
        "КРОМЕ ТОГО",
        "НАСКОЛЬКО",
        "РАССМАТРИВАЮТСЯ СЛЕДУЮЩИЕ",
    )
    if any(normalized_title.startswith(prefix) for prefix in suspicious_prefixes):
        score -= 40.0

    if len(title.split()) > 14:
        score -= 12.0

    return score


def deduplicate_paragraphs(paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for paragraph in paragraphs:
        grouped.setdefault(paragraph["paragraph_key"], []).append(paragraph)

    deduplicated: list[dict[str, Any]] = []
    for paragraph_key, candidates in grouped.items():
        if len(candidates) == 1:
            deduplicated.append(candidates[0])
            continue

        best = max(
            candidates,
            key=lambda candidate: (
                paragraph_candidate_score(candidate),
                -candidate["page_start"],
                -len(candidate["paragraph_title"]),
            ),
        )
        deduplicated.append(best)

    deduplicated.sort(key=lambda paragraph: (paragraph["page_start"], paragraph_sort_key(paragraph["paragraph_key"])))
    return deduplicated


def build_clause_retrieval_groups(
    paragraph: dict[str, Any],
    min_informative_chars: int = 220,
) -> list[dict[str, Any]]:
    clauses = paragraph["clauses"]
    groups: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        groups.append(
            {
                "chunk_kind": "clause_group" if len(buffer) > 1 else "clause",
                "clauses": buffer[:],
                "text": build_retrieval_group_text(paragraph, buffer, min_informative_chars),
            }
        )
        buffer = []

    for clause in clauses:
        candidate_text = build_retrieval_group_text(paragraph, [clause], min_informative_chars)
        should_buffer = len(candidate_text) < min_informative_chars

        if should_buffer:
            buffer.append(clause)
            if len(build_retrieval_group_text(paragraph, buffer, min_informative_chars)) >= min_informative_chars:
                flush_buffer()
            continue

        flush_buffer()
        groups.append(
            {
                "chunk_kind": "clause",
                "clauses": [clause],
                "text": candidate_text,
            }
        )

    flush_buffer()
    return groups


def build_retrieval_chunks(parsed_document: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    registry = parsed_document["registry"]
    doc_code = parsed_document["document_code"]
    doc_type = parsed_document["document_type"]

    for paragraph in parsed_document["paragraphs"]:
        common = {
            "document_id": parsed_document["document_id"],
            "document_code": doc_code,
            "document_title": parsed_document["document_title"],
            "document_type": doc_type,
            "source_file": parsed_document["source_file"],
            "heading_path": paragraph["heading_path"],
            "context_heading": paragraph["context_heading"],
            "paragraph_id": paragraph["paragraph_id"],
            "paragraph_suffix": paragraph["paragraph_suffix"],
            "paragraph_key": paragraph["paragraph_key"],
            "paragraph_label": paragraph["paragraph_label"],
            "paragraph_title": paragraph["paragraph_title"],
            "page_start": paragraph["page_start"],
            "page_end": paragraph["page_end"],
        }

        if paragraph["clauses"]:
            retrieval_groups = build_clause_retrieval_groups(paragraph)
            for group_index, group in enumerate(retrieval_groups, start=1):
                source_clauses = group["clauses"]
                chunk_text = group["text"]
                clause_markers = [clause["marker"] for clause in source_clauses]
                source_clause_ids = [clause["clause_id"] for clause in source_clauses]
                local_chunk_id = f"{paragraph['paragraph_key']}::chunk_{group_index:02d}"
                search_text = normalize_spaces(
                    " ".join(
                        part
                        for part in [
                            doc_code,
                            doc_type,
                            paragraph["context_heading"],
                            paragraph["paragraph_label"],
                            paragraph["paragraph_title"],
                            chunk_text,
                        ]
                        if part
                    )
                )
                chunk = {
                    **common,
                    "chunk_id": f"{parsed_document['document_id']}__{local_chunk_id}",
                    "local_chunk_id": local_chunk_id,
                    "chunk_kind": group["chunk_kind"],
                    "clause_marker": clause_markers[0] if len(clause_markers) == 1 else None,
                    "clause_markers": clause_markers,
                    "normalized_clause_marker": source_clauses[0]["normalized_marker"] if len(source_clauses) == 1 else None,
                    "source_clause_ids": source_clause_ids,
                    "source_clause_count": len(source_clauses),
                    "text": chunk_text,
                    "flattened_text": chunk_text,
                    "inline_items": [item for clause in source_clauses for item in clause["inline_items"]],
                    "search_text": search_text,
                }
                chunk["cross_references"] = extract_cross_references(chunk_text, registry)
                chunks.append(chunk)
            continue

        chunk_text = paragraph["body_text"] or paragraph["paragraph_title"]
        if is_reserved_text(chunk_text):
            continue
        search_text = normalize_spaces(
            " ".join(
                part
                for part in [
                    doc_code,
                    doc_type,
                    paragraph["context_heading"],
                    paragraph["paragraph_label"],
                    paragraph["paragraph_title"],
                    chunk_text,
                ]
                if part
            )
        )
        chunk = {
            **common,
            "chunk_id": f"{parsed_document['document_id']}__{paragraph['paragraph_key']}::chunk_01",
            "local_chunk_id": f"{paragraph['paragraph_key']}::chunk_01",
            "chunk_kind": "paragraph",
            "clause_marker": None,
            "clause_markers": [],
            "normalized_clause_marker": None,
            "source_clause_ids": [],
            "source_clause_count": 0,
            "text": chunk_text,
            "flattened_text": chunk_text,
            "inline_items": paragraph.get("inline_items", []),
            "search_text": search_text,
        }
        chunk["cross_references"] = extract_cross_references(chunk_text, registry)
        chunks.append(chunk)

    return chunks


def chunks_for_notebook(parsed_document: dict[str, Any]) -> list[dict[str, Any]]:
    notebook_chunks: list[dict[str, Any]] = []
    for chunk in parsed_document["chunks"]:
        notebook_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_code": chunk["document_code"],
                "document_title": chunk["document_title"],
                "document_type": chunk["document_type"],
                "paragraph": chunk["paragraph_key"],
                "paragraph_label": chunk["paragraph_label"],
                "paragraph_title": chunk["paragraph_title"],
                "chunk_kind": chunk["chunk_kind"],
                "clause_marker": chunk["clause_marker"],
                "clause_markers": chunk.get("clause_markers", []),
                "normalized_clause_marker": chunk.get("normalized_clause_marker"),
                "context": chunk["context_heading"],
                "heading_path": chunk["heading_path"],
                "text": chunk["flattened_text"],
                "search_text": chunk["search_text"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "cross_references": chunk["cross_references"],
                "inline_items": chunk["inline_items"],
                "source_clause_ids": chunk.get("source_clause_ids", []),
                "source_clause_count": chunk.get("source_clause_count", 0),
            }
        )
    return notebook_chunks


def parse_ap25_for_notebook(pdf_path: str) -> list[dict[str, Any]]:
    parsed_document = parse_ap25(pdf_path)
    return chunks_for_notebook(parsed_document)


def to_langchain_documents(chunks: list[dict[str, Any]]) -> list[Any]:
    from langchain_core.documents import Document

    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "document_code": chunk["document_code"],
                    "document_title": chunk["document_title"],
                    "document_type": chunk["document_type"],
                    "paragraph": chunk["paragraph"],
                    "paragraph_label": chunk["paragraph_label"],
                    "paragraph_title": chunk["paragraph_title"],
                    "chunk_kind": chunk.get("chunk_kind"),
                    "clause_marker": chunk["clause_marker"],
                    "clause_markers": chunk.get("clause_markers", []),
                    "normalized_clause_marker": chunk.get("normalized_clause_marker"),
                    "context": chunk["context"],
                    "heading_path": chunk["heading_path"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "cross_references": chunk["cross_references"],
                    "source_clause_ids": chunk.get("source_clause_ids", []),
                    "source_clause_count": chunk.get("source_clause_count", 0),
                },
            )
        )
    return documents


def print_chunk_examples(chunks: list[dict[str, Any]], limit: int = 5, preview_chars: int = 300) -> None:
    for chunk in chunks[:limit]:
        if chunk.get("clause_marker"):
            clause = f" {chunk['clause_marker']}"
        elif chunk.get("clause_markers"):
            clause = " " + ", ".join(chunk["clause_markers"])
        else:
            clause = ""
        print("\n---")
        print(
            f"{chunk['paragraph_label']}{clause} | context: {chunk['context']} | "
            f"title: {chunk['paragraph_title']} | kind: {chunk.get('chunk_kind')} | "
            f"source_clause_count: {chunk.get('source_clause_count', 0)}"
        )
        print(chunk["text"][:preview_chars])


def diagnose_chunks(
    chunks: list[dict[str, Any]],
    sample_size: int = 5,
    short_threshold: int = 50,
    gap_threshold: int = 5,
    plot: bool = True,
) -> dict[str, Any]:
    if not chunks:
        print("=" * 80)
        print("ОБЩАЯ ИНФОРМАЦИЯ")
        print("Чанки отсутствуют")
        return {
            "chunk_count": 0,
            "paragraph_count": 0,
        }

    def paragraph_key_of(chunk: dict[str, Any]) -> str:
        return chunk.get("paragraph") or chunk.get("paragraph_key") or ""

    def context_of(chunk: dict[str, Any]) -> str:
        return chunk.get("context") or chunk.get("context_heading") or ""

    def title_of(chunk: dict[str, Any]) -> str:
        return chunk.get("paragraph_title") or ""

    def chunk_id_of(chunk: dict[str, Any]) -> str:
        return chunk.get("chunk_id") or ""

    def clause_of(chunk: dict[str, Any]) -> Optional[str]:
        return chunk.get("clause_marker")

    def clause_markers_of(chunk: dict[str, Any]) -> list[str]:
        if chunk.get("clause_markers"):
            return chunk["clause_markers"]
        if chunk.get("clause_marker"):
            return [chunk["clause_marker"]]
        return []

    def text_of(chunk: dict[str, Any]) -> str:
        return chunk.get("text") or ""

    def chunk_kind_of(chunk: dict[str, Any]) -> str:
        return chunk.get("chunk_kind") or "paragraph"

    def clause_sort_key(marker: Optional[str]) -> tuple[int, str]:
        if not marker:
            return (-1, "")
        raw = marker.strip()[1:-1] if marker.startswith("(") and marker.endswith(")") else marker
        normalized = normalize_marker(raw)
        return (marker_level(raw), normalized)

    def numeric_minor_of(paragraph_key_value: str) -> Optional[int]:
        match = re.fullmatch(r"25\.(\d+)([A-Z]?)", paragraph_key_value)
        if not match:
            return None
        return int(match.group(1))

    lengths = [len(text_of(chunk)) for chunk in chunks]
    paragraphs = [paragraph_key_of(chunk) for chunk in chunks if paragraph_key_of(chunk)]
    unique_paragraphs = sorted(set(paragraphs), key=paragraph_sort_key)
    chunk_ids = [chunk_id_of(chunk) for chunk in chunks if chunk_id_of(chunk)]
    unique_contexts = sorted({context_of(chunk) for chunk in chunks if context_of(chunk)})
    clause_chunks = [chunk for chunk in chunks if chunk_kind_of(chunk) == "clause"]
    grouped_clause_chunks = [chunk for chunk in chunks if chunk_kind_of(chunk) == "clause_group"]
    reference_count = sum(len(chunk.get("cross_references", [])) for chunk in chunks)
    inline_count = sum(1 for chunk in chunks if chunk.get("inline_items"))
    avg_chunks_per_paragraph = len(chunks) / len(unique_paragraphs) if unique_paragraphs else 0.0

    print("=" * 80)
    print("ОБЩАЯ ИНФОРМАЦИЯ")
    print(f"Всего чанков: {len(chunks)}")
    print(f"Уникальных параграфов: {len(unique_paragraphs)}")
    print(f"Среднее число чанков на параграф: {avg_chunks_per_paragraph:.2f}")
    print(f"Одно-подпунктных чанков: {len(clause_chunks)}")
    print(f"Групповых чанков по подпунктам: {len(grouped_clause_chunks)}")
    print(f"Чанков с вложенными списками: {inline_count}")
    print(f"Всего перекрестных ссылок: {reference_count}")

    print("\n" + "=" * 80)
    print("ПРИМЕРЫ ЧАНКОВ")
    for chunk in chunks[:sample_size]:
        if clause_of(chunk):
            clause_suffix = f" {clause_of(chunk)}"
        elif clause_markers_of(chunk):
            clause_suffix = " " + ", ".join(clause_markers_of(chunk))
        else:
            clause_suffix = ""
        print("\n---")
        print(
            f'{chunk.get("paragraph_label", paragraph_key_of(chunk))}{clause_suffix} | '
            f'context: {context_of(chunk)} | title: {title_of(chunk)} | '
            f'kind: {chunk_kind_of(chunk)} | source_clause_count: {chunk.get("source_clause_count", 0)}'
        )
        print(text_of(chunk)[:300])

    print("\n" + "=" * 80)
    print("СТАТИСТИКА ДЛИНЫ")
    print(f"min: {min(lengths)}")
    print(f"max: {max(lengths)}")
    print(f"avg: {sum(lengths) // len(lengths)}")
    sorted_lengths = sorted(lengths)
    median = sorted_lengths[len(sorted_lengths) // 2]
    p95 = sorted_lengths[min(len(sorted_lengths) - 1, int(len(sorted_lengths) * 0.95))]
    print(f"median: {median}")
    print(f"p95: {p95}")

    if plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.hist(lengths, bins=50)
            plt.title("Распределение длины чанков")
            plt.xlabel("Длина текста")
            plt.ylabel("Количество")
            plt.show()
        except Exception as exc:
            print(f"\nГистограмма не построена: {exc}")

    print("\n" + "=" * 80)
    print("ПРОВЕРКА ПОСЛЕДОВАТЕЛЬНОСТИ")
    print("Крупные разрывы в нумерации показаны только как диагностика, это не всегда ошибка.")

    gaps: list[tuple[str, str, int]] = []
    previous_paragraph: Optional[str] = None
    previous_minor: Optional[int] = None
    for paragraph in unique_paragraphs:
        current_minor = numeric_minor_of(paragraph)
        if previous_paragraph is not None and previous_minor is not None and current_minor is not None:
            gap = current_minor - previous_minor
            if gap > gap_threshold:
                gaps.append((previous_paragraph, paragraph, gap))
        previous_paragraph = paragraph
        previous_minor = current_minor

    if gaps:
        for left, right, gap in gaps[:20]:
            print(f"Разрыв: {left} -> {right} (delta={gap})")
        if len(gaps) > 20:
            print(f"... еще {len(gaps) - 20} разрывов")
    else:
        print("Крупных разрывов по заданному порогу не найдено")

    print("\n" + "=" * 80)
    print("ДУБЛИКАТЫ")
    duplicate_chunk_ids = sorted({chunk_id for chunk_id in chunk_ids if chunk_ids.count(chunk_id) > 1})
    if duplicate_chunk_ids:
        print(f"Найдены дубликаты chunk_id: {duplicate_chunk_ids[:10]}")
        if len(duplicate_chunk_ids) > 10:
            print(f"... еще {len(duplicate_chunk_ids) - 10} duplicate chunk_id")
    else:
        print("✔ Дубликатов chunk_id нет")
    print("Повторяющиеся paragraph допустимы, потому что один параграф может быть разбит на несколько retrieval-чанков.")

    print("\n" + "=" * 80)
    print("КОНТЕКСТ")
    no_context = [chunk for chunk in chunks if not context_of(chunk)]
    print(f"Без контекста: {len(no_context)} ({len(no_context) / len(chunks) * 100:.1f}%)")
    print(f"Уникальных контекстов: {len(unique_contexts)}")

    print("\n" + "=" * 80)
    print("ЗАГОЛОВКИ И МЕТАДАННЫЕ")
    no_title = [chunk for chunk in chunks if not title_of(chunk)]
    no_document_type = [chunk for chunk in chunks if not chunk.get("document_type")]
    print(f"Без paragraph_title: {len(no_title)}")
    print(f"Без document_type: {len(no_document_type)}")

    print("\n" + "=" * 80)
    print("ПОДОЗРИТЕЛЬНО КОРОТКИЕ ЧАНКИ")
    short_chunks = [chunk for chunk in chunks if len(text_of(chunk)) < short_threshold]
    print(f"Коротких чанков (< {short_threshold}): {len(short_chunks)}")
    for chunk in short_chunks[:5]:
        markers = clause_markers_of(chunk)
        clause_suffix = f" {', '.join(markers)}" if markers else ""
        print(f'  {chunk.get("paragraph_label", paragraph_key_of(chunk))}{clause_suffix}: {text_of(chunk)[:100]}')

    print("\n" + "=" * 80)
    print("ЧАНКИ БЕЗ ВНУТРЕННЕЙ СТРУКТУРЫ")
    no_inline_chunks = [chunk for chunk in chunks if not chunk.get("inline_items")]
    print(f"Без inline_items: {len(no_inline_chunks)} ({len(no_inline_chunks) / len(chunks) * 100:.1f}%)")

    print("\n" + "=" * 80)
    print("ПЕРЕКРЕСТНЫЕ ССЫЛКИ")
    with_refs = [chunk for chunk in chunks if chunk.get("cross_references")]
    print(f"Чанков со ссылками: {len(with_refs)} ({len(with_refs) / len(chunks) * 100:.1f}%)")
    range_refs = sum(
        1
        for chunk in chunks
        for reference in chunk.get("cross_references", [])
        if reference.get("reference_type") == "range"
    )
    single_refs = sum(
        1
        for chunk in chunks
        for reference in chunk.get("cross_references", [])
        if reference.get("reference_type") == "single"
    )
    print(f"Single references: {single_refs}")
    print(f"Range references: {range_refs}")

    paragraph_distribution: dict[str, int] = {}
    for paragraph in paragraphs:
        paragraph_distribution[paragraph] = paragraph_distribution.get(paragraph, 0) + 1
    multi_chunk_paragraphs = sorted(
        [(paragraph, count) for paragraph, count in paragraph_distribution.items() if count > 1],
        key=lambda item: (-item[1], paragraph_sort_key(item[0])),
    )

    duplicate_markers_by_paragraph: list[tuple[str, list[str]]] = []
    for paragraph in unique_paragraphs:
        paragraph_markers: list[str] = []
        for chunk in chunks:
            if paragraph_key_of(chunk) != paragraph:
                continue
            paragraph_markers.extend(clause_markers_of(chunk))
        repeated = sorted({marker for marker in paragraph_markers if paragraph_markers.count(marker) > 1}, key=clause_sort_key)
        if repeated:
            duplicate_markers_by_paragraph.append((paragraph, repeated))

    print("\n" + "=" * 80)
    print("ПАРАГРАФЫ С НЕСКОЛЬКИМИ ЧАНКАМИ")
    if multi_chunk_paragraphs:
        for paragraph, count in multi_chunk_paragraphs[:10]:
            paragraph_chunks = sorted(
                [chunk for chunk in chunks if paragraph_key_of(chunk) == paragraph],
                key=lambda chunk: clause_sort_key(clause_markers_of(chunk)[0] if clause_markers_of(chunk) else None),
            )
            markers = [", ".join(clause_markers_of(chunk)) if clause_markers_of(chunk) else "<full>" for chunk in paragraph_chunks]
            print(f"  {paragraph}: {count} чанков | markers={markers}")
    else:
        print("Все параграфы представлены одним чанком")

    print("\n" + "=" * 80)
    print("ПОВТОРЯЮЩИЕСЯ CLAUSE MARKERS")
    if duplicate_markers_by_paragraph:
        for paragraph, repeated in duplicate_markers_by_paragraph[:10]:
            print(f"  {paragraph}: duplicate markers={repeated}")
        if len(duplicate_markers_by_paragraph) > 10:
            print(f"... еще {len(duplicate_markers_by_paragraph) - 10} параграфов с повторяющимися markers")
    else:
        print("✔ Повторяющихся clause_marker внутри одного параграфа не найдено")

    return {
        "chunk_count": len(chunks),
        "paragraph_count": len(unique_paragraphs),
        "avg_chunks_per_paragraph": avg_chunks_per_paragraph,
        "single_clause_chunk_count": len(clause_chunks),
        "grouped_clause_chunk_count": len(grouped_clause_chunks),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": sum(lengths) // len(lengths),
        "median_length": median,
        "p95_length": p95,
        "unique_context_count": len(unique_contexts),
        "missing_context_count": len(no_context),
        "missing_title_count": len(no_title),
        "short_chunk_count": len(short_chunks),
        "chunk_id_duplicate_count": len(duplicate_chunk_ids),
        "single_reference_count": single_refs,
        "range_reference_count": range_refs,
        "multi_chunk_paragraph_count": len(multi_chunk_paragraphs),
        "duplicate_marker_paragraph_count": len(duplicate_markers_by_paragraph),
        "gaps": gaps,
    }


def parse_ap25(pdf_path: str) -> dict[str, Any]:
    reader = PdfReader(pdf_path)
    document = extract_document_metadata(pdf_path)
    blocks = collect_paragraph_blocks(reader)
    paragraphs = deduplicate_paragraphs([build_paragraph_record(block) for block in blocks])
    registry = build_paragraph_registry(paragraphs)

    for paragraph in paragraphs:
        paragraph["cross_references"] = extract_cross_references(paragraph["full_text"], registry)

    parsed_document = {
        **document,
        "page_count": len(reader.pages),
        "paragraph_count": len(paragraphs),
        "paragraphs": paragraphs,
        "registry": registry,
    }
    parsed_document["chunks"] = build_retrieval_chunks(parsed_document)
    parsed_document["chunk_count"] = len(parsed_document["chunks"])
    return parsed_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse AP-25 PDF into structured paragraphs and retrieval chunks.")
    parser.add_argument("pdf_path", help="Path to AP-25 PDF file")
    parser.add_argument("--output", help="Write full parsed document JSON to this file")
    parser.add_argument("--chunks-output", help="Write retrieval chunks JSONL to this file")
    parser.add_argument(
        "--inspect",
        action="append",
        default=[],
        help="Print one parsed paragraph by key, for example 25.29 or 25.123A",
    )
    args = parser.parse_args()

    parsed = parse_ap25(args.pdf_path)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.chunks_output:
        chunks_path = Path(args.chunks_output)
        chunks_path.write_text(
            "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in parsed["chunks"]) + "\n",
            encoding="utf-8",
        )

    if args.inspect:
        wanted = set(args.inspect)
        for paragraph in parsed["paragraphs"]:
            if paragraph["paragraph_key"] in wanted or paragraph["paragraph_id"] in wanted:
                print(json.dumps(paragraph, ensure_ascii=False, indent=2))
        return

    summary = {
        "document_code": parsed["document_code"],
        "document_type": parsed["document_type"],
        "paragraph_count": parsed["paragraph_count"],
        "chunk_count": parsed["chunk_count"],
        "first_paragraph": parsed["paragraphs"][0]["paragraph_key"] if parsed["paragraphs"] else None,
        "last_paragraph": parsed["paragraphs"][-1]["paragraph_key"] if parsed["paragraphs"] else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
