#!/usr/bin/env python3
"""Build a frozen, browser-friendly manual review queue from QA output."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.review import build_review_rows  # noqa: E402
from dtlr_poc.selection import sha256_file  # noqa: E402


CSV_FIELDS = [
    "pair_id", "queue_group", "selection_rank_sha256", "line_id",
    "left_gt_index", "right_gt_index", "pair", "left_alignment",
    "right_alignment", "usable", "connected_dominant_core_v3",
    "connected_exclusive_core_v2_1", "unusable_reason_codes", "image",
    "manual_alignment", "manual_visual_connectivity", "manual_v3_assessment",
    "manual_visual_cause", "notes",
]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field, "") for field in CSV_FIELDS}
            output["unusable_reason_codes"] = json.dumps(row.get("unusable_reason_codes", []))
            writer.writerow(output)


def write_html(queue: dict, qa_root: Path, output_dir: Path, path: Path) -> None:
    browser_rows = []
    for row in queue["rows"]:
        output = {key: row.get(key) for key in (
            "pair_id", "queue_group", "line_id", "left_gt_index", "right_gt_index",
            "pair", "left_alignment", "right_alignment", "usable",
            "connected_dominant_core_v3", "connected_exclusive_core_v2_1",
            "unusable_reason_codes",
        )}
        output["image"] = (
            os.path.relpath(qa_root / row["image"], output_dir)
            if row.get("image") else None
        )
        browser_rows.append(output)
    embedded = json.dumps({**queue, "rows": browser_rows}, ensure_ascii=False).replace("</", "<\\/")
    document = """<!doctype html>
<meta charset="utf-8">
<title>IAM validation review queue</title>
<style>
body { font: 15px system-ui,sans-serif; margin: 20px; max-width: 1200px; }
header { position: sticky; top: 0; background: white; padding: 8px 0; border-bottom: 1px solid #bbb; }
button,select,input,textarea { font: inherit; margin: 4px; }
img { max-width: 100%; border: 1px solid #ccc; }
.meta { background: #f4f4f4; padding: 10px; white-space: pre-wrap; }
.field { margin: 10px 0; }
textarea { width: 95%; min-height: 60px; }
</style>
<header>
  <b id="progress"></b>
  <button id="prev">Previous</button><button id="next">Next</button>
  <select id="filter"><option value="all">All groups</option>
    <option value="v2.1-v3-disagreement">Disagreements</option>
    <option value="unusable">Unusable</option>
    <option value="agreement-audit">Agreement audit</option></select>
  <button id="export">Export review JSON</button>
</header>
<h1>IAM validation manual review</h1>
<div id="card"></div>
<script>
const queue = __QUEUE__;
const storageKey = "dtlr-review:" + queue.qa_manifest_sha256 + ":" + queue.seed;
let answers = JSON.parse(localStorage.getItem(storageKey) || "{}");
let visible = queue.rows.map((_, i) => i), cursor = 0;
const esc = s => String(s ?? "").replace(/[&<>\"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function answer(row) { return answers[row.pair_id] || {alignment:"pending",visual_connectivity:"pending",v3_assessment:"pending",visual_cause:"",notes:""}; }
function save(row, key, value) { const a=answer(row); a[key]=value; answers[row.pair_id]=a; localStorage.setItem(storageKey,JSON.stringify(answers)); render(); }
function options(name, values, selected) { return `<select data-key="${name}">${values.map(v=>`<option ${v===selected?'selected':''}>${v}</option>`).join('')}</select>`; }
function render() {
  if (!visible.length) { document.querySelector('#card').innerHTML='<p>No rows in this filter.</p>'; return; }
  cursor=Math.max(0,Math.min(cursor,visible.length-1)); const row=queue.rows[visible[cursor]], a=answer(row);
  const done=visible.filter(i => answer(queue.rows[i]).v3_assessment !== 'pending').length;
  document.querySelector('#progress').textContent=`${cursor+1}/${visible.length}; assessed ${done}/${visible.length}`;
  document.querySelector('#card').innerHTML=`
    <h2>${esc(row.queue_group)} — ${esc(row.pair_id)} — ${esc(row.pair)}</h2>
    <div class="meta">alignment=${esc(row.left_alignment)}/${esc(row.right_alignment)}\nv3=${esc(row.connected_dominant_core_v3)}  v2.1=${esc(row.connected_exclusive_core_v2_1)}\nreasons=${esc((row.unusable_reason_codes||[]).join(','))}</div>
    ${row.image?`<img src="${esc(row.image)}">`:'<p>No pair image: detection missing.</p>'}
    <div class="field">Alignment ${options('alignment',['pending','correct','incorrect','uncertain'],a.alignment)}</div>
    <div class="field">Visual connectivity ${options('visual_connectivity',['pending','connected','disconnected','uncertain'],a.visual_connectivity)}</div>
    <div class="field">v3 assessment ${options('v3_assessment',['pending','correct','incorrect','appropriate-abstention','unnecessary-abstention','uncertain'],a.v3_assessment)}</div>
    <div class="field">Visual cause <input data-key="visual_cause" value="${esc(a.visual_cause)}"></div>
    <div class="field">Notes<br><textarea data-key="notes">${esc(a.notes)}</textarea></div>`;
  document.querySelectorAll('[data-key]').forEach(el => el.addEventListener('change', e => save(row,e.target.dataset.key,e.target.value)));
}
document.querySelector('#prev').onclick=()=>{cursor--;render();}; document.querySelector('#next').onclick=()=>{cursor++;render();};
document.querySelector('#filter').onchange=e=>{visible=queue.rows.map((r,i)=>[r,i]).filter(([r])=>e.target.value==='all'||r.queue_group===e.target.value).map(([,i])=>i);cursor=0;render();};
document.querySelector('#export').onclick=()=>{const payload={schema_version:'dtlr.qa-review.v1',queue_schema_version:queue.schema_version,qa_manifest_sha256:queue.qa_manifest_sha256,seed:queue.seed,exported_utc:new Date().toISOString(),annotations:answers};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='manual_review.json';a.click();URL.revokeObjectURL(a.href);};
render();
</script>
""".replace("__QUEUE__", embedded)
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--agreement-audit-count", type=int, default=100)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    qa = json.loads(args.qa_manifest.read_text(encoding="utf-8"))
    rows = build_review_rows(qa["lines"], args.agreement_audit_count, args.seed)
    counts = Counter(row["queue_group"] for row in rows)
    queue = {
        "schema_version": "dtlr.qa-review-queue.v1",
        "qa_manifest_sha256": sha256_file(args.qa_manifest),
        "seed": args.seed,
        "agreement_audit_requested": args.agreement_audit_count,
        "queue_count": len(rows),
        "group_counts": dict(sorted(counts.items())),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "review_queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(rows, args.output_dir / "review_queue.csv")
    write_html(queue, args.qa_manifest.parent, args.output_dir, args.output_dir / "review_queue.html")
    print(json.dumps({key: value for key, value in queue.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
