import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREAMBLES_DIR = PROJECT_ROOT / "preambles"


def _enabled_depts() -> set | None:
    val = os.environ.get("ENABLED_DEPTS", "").strip()
    if not val:
        return None
    return set(d.strip() for d in val.split(",") if d.strip())

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<fm>.*?)\n---\s*\n(?P<body>.*)",
    re.DOTALL,
)


def _parse_simple_yaml(yaml_text: str) -> dict:
    out = {}
    for line in yaml_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

                       
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1].replace('\\"', '"')

                        
        if value.isdigit():
            value = int(value)

        out[key] = value
    return out


def parse_md_file(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(content)
    if not m:
        raise ValueError(f"Файл {path} не содержит valid YAML frontmatter")
    fm = _parse_simple_yaml(m.group("fm"))
    body = m.group("body").strip()
    return {"metadata": fm, "text": body}


def load_l1() -> list[dict]:
    enabled = _enabled_depts()
    l1_dir = PREAMBLES_DIR / "L1"
    cards = []
    for md in sorted(l1_dir.glob("*.md")):
        parsed = parse_md_file(md)
        meta = parsed["metadata"]
        if enabled is not None and meta["department"] not in enabled:
            continue
        cards.append({
            "text": parsed["text"],
            "metadata": {
                "level": int(meta.get("level", 1)),
                "department": meta["department"],
                "subdepartment": None,
            },
        })
    return cards


def load_l2() -> list[dict]:
    enabled = _enabled_depts()
    l2_root = PREAMBLES_DIR / "L2"
    cards = []
    for dept_dir in sorted(l2_root.iterdir()):
        if not dept_dir.is_dir():
            continue
        if enabled is not None and dept_dir.name not in enabled:
            continue
        for md in sorted(dept_dir.glob("*.md")):
            parsed = parse_md_file(md)
            meta = parsed["metadata"]
            cards.append({
                "text": parsed["text"],
                "metadata": {
                    "level": int(meta.get("level", 2)),
                    "department": meta["department"],
                    "subdepartment": meta["subdepartment"],
                },
            })
    return cards


def load_all() -> list[dict]:
    return load_l1() + load_l2()
