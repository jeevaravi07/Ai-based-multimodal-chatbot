"""
Step 1 of the case-report RAG pipeline.

- Loads data/cases.csv (image_name, article_id, case_id, case_text).
- Splits each case_text into CARE-guideline sections (Patient information,
  Clinical findings, Diagnostic assessment, ...) using the section headers
  that MultiCaRe-style narratives already contain. Cases without those
  headers are kept whole under "Overview".
- Detects the imaging modality mentioned near each image's own row
  (X-ray / CT / MRI / Ultrasound / Histopathology / Photograph / Other)
  from keywords in the case text, and copies that image into
  data/images_sorted/<modality>/ so images are auto-organized by scan type.
- Returns/saves one row per (case_id, image_name) with the parsed sections,
  ready for text_embeddings.py and image_retriever.py to consume.
"""
import csv
import os
import re
import shutil

import config

_SECTION_HEADER_RE = re.compile(
    r"(?im)^(" + "|".join(re.escape(s) for s in config.CARE_SECTIONS) + r")\s*:\s*"
)

_MODALITY_KEYWORDS = {
    "MRI": [r"\bmri\b", r"magnetic resonance"],
    "CT": [r"\bct scan\b", r"\bct\b", r"computed tomography"],
    "X-ray": [r"x-?ray", r"radiograph"],
    "Ultrasound": [r"ultrasound", r"sonograph", r"\busg\b"],
    "Histopathology": [r"histopatholog", r"biopsy", r"immunohistochem", r"\bihc\b"],
    "Photograph": [r"photograph", r"clinical photo", r"gross appearance"],
}


def split_into_sections(case_text: str) -> dict:
    """Split raw case narrative into {section_name: text}. Falls back to
    a single 'Overview' section when no CARE headers are found."""
    matches = list(_SECTION_HEADER_RE.finditer(case_text))
    if not matches:
        return {"Overview": case_text.strip()}

    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(case_text)
        sections[name] = case_text[start:end].strip()


    lead = case_text[: matches[0].start()].strip()
    if lead:
        sections.setdefault("Overview", lead)
    return sections


def detect_modality(case_text: str) -> str:
    """Best-effort scan-type guess from keywords in the narrative."""
    text_lower = case_text.lower()
    scores = {mod: 0 for mod in _MODALITY_KEYWORDS}
    for mod, patterns in _MODALITY_KEYWORDS.items():
        for pat in patterns:
            scores[mod] += len(re.findall(pat, text_lower))
    best_mod, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_mod if best_score > 0 else "Other"


def load_cases(csv_path: str = config.CASES_CSV) -> list:
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows


def build_dataset(csv_path: str = config.CASES_CSV, sort_images: bool = True) -> list:
    """Returns a list of dicts:
    {image_name, article_id, case_id, case_text, sections, modality, image_path}
    and, if sort_images=True, copies each image into data/images_sorted/<modality>/.
    """
    rows = load_cases(csv_path)
    records = []
    for row in rows:
        case_text = row["case_text"]
        sections = split_into_sections(case_text)
        modality = detect_modality(case_text)

        src_path = os.path.join(config.IMAGES_RAW_DIR, row["image_name"])
        dst_dir = os.path.join(config.IMAGES_SORTED_DIR, modality)
        dst_path = os.path.join(dst_dir, row["image_name"])
        if sort_images and os.path.exists(src_path):
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

        records.append({
            "image_name": row["image_name"],
            "article_id": row["article_id"],
            "case_id": row["case_id"],
            "case_text": case_text,
            "sections": sections,
            "modality": modality,
            "image_path": src_path,
        })
    return records


if __name__ == "__main__":
    recs = build_dataset()
    from collections import Counter
    counts = Counter(r["modality"] for r in recs)
    print(f"Parsed {len(recs)} case rows across {len({r['case_id'] for r in recs})} unique cases.")
    print("Modality breakdown (auto-sorted into data/images_sorted/):")
    for mod, c in counts.most_common():
        print(f"  {mod:15s} {c}")
