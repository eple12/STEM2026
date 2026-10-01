"""Update one circuit's row in ai_sw/policy/RECORDS.md.

Called by train_track.bat right after a deploy is confirmed, so the table
stays a live snapshot without a manual edit every time. Also runnable by
hand for an out-of-band deploy:

    python -m tools.update_records --circuit Monza --lap 93.30 \
        --checkpoint Monza/Monza_v26k_wide_best.npz --date 2026-09-14

Rewrites the circuit's existing row if there is one (matched on the exact
"| Circuit |" cell), otherwise appends a new one before the "## How to
update" section. Never touches any other row.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path

RECORDS_PATH = Path(__file__).resolve().parent.parent / "policy" / "RECORDS.md"
ROW_RE_TEMPLATE = r"^\|\s*{}\s*\|.*\|$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--lap", type=float, required=True,
                    help="best lap time in seconds")
    ap.add_argument("--checkpoint", required=True,
                    help="path under ai_sw/policy/, e.g. Monza/Monza_v26k_best.npz")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD; defaults to today")
    ap.add_argument("--notes", default="v26k recipe")
    ap.add_argument("--records", default=str(RECORDS_PATH),
                    help="override for testing")
    args = ap.parse_args()

    date = args.date or _dt.date.today().isoformat()
    new_row = (f"| {args.circuit} | {args.lap:.2f} s | `{args.checkpoint}` "
               f"| {date} | {args.notes} |")

    path = Path(args.records)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    row_re = re.compile(ROW_RE_TEMPLATE.format(re.escape(args.circuit)))
    for i, line in enumerate(lines):
        if row_re.match(line.strip()):
            old = lines[i]
            lines[i] = new_row
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"updated {args.circuit} row in {path}:")
            print(f"  - {old}")
            print(f"  + {new_row}")
            return

    # No existing row: insert right after the header separator ("|---|...|"),
    # so a brand-new circuit still lands in the table rather than nowhere.
    sep_re = re.compile(r"^\|[-\s|]+\|$")
    for i, line in enumerate(lines):
        if sep_re.match(line.strip()):
            lines.insert(i + 1, new_row)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"added new {args.circuit} row to {path}:")
            print(f"  + {new_row}")
            return

    raise SystemExit(f"could not find the table header in {path} to insert into")


if __name__ == "__main__":
    main()
