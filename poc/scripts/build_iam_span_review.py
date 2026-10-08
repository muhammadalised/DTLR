#!/usr/bin/env python3
"""Build a pinned IAM-train visual pilot for selected longer subwords.

No fitting or threshold changes. Samples are frozen before images are opened.
Only images named in the pinned IAM train detection export may be read.
"""
import argparse
from collections import Counter
import gc
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))
from dtlr_poc.alignment import gt_detection_map
from dtlr_poc.ccl import label_ink, otsu_threshold, pair_component_evidence
from dtlr_poc.handwriting_subwords import prepare_corpus, collect_candidates
from dtlr_poc.selection import sha256_file
from dtlr_poc.span_review import select_span_review

COLORS = [(22, 127, 133), (180, 90, 40), (110, 65, 170), (45, 100, 190), (175, 45, 100), (80, 135, 45)]
FONT = ImageFont.truetype("DejaVuSans.ttf", 18)
SMALL = ImageFont.truetype("DejaVuSans.ttf", 15)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def normalize(value):
    return json.loads(json.dumps(value, sort_keys=True))


def decision(edges):
    if any(not e["dominant_core_usable"] for e in edges):
        return None
    ids = {e[k] for e in edges for k in ("left_dominant_component", "right_dominant_component")}
    return len(ids) == 1


def translated_box(box, bounds, scale=2):
    if not np.all(np.isfinite(box)) or box[2] < box[0] or box[3] < box[1]:
        return None
    x0, y0, _, _ = bounds
    return tuple(round(v * scale) for v in (box[0]-x0, box[1]-y0, box[2]-x0, box[3]-y0))


def panel(crop, title, boxes=()):
    scale = 2
    pic = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(pic)
    for box, color, label in boxes:
        if box is not None:
            draw.rectangle(box, outline=color, width=2)
            if label:
                draw.text((max(0,box[0]), max(0,box[1]-19)), label, fill=color, font=SMALL, stroke_width=1, stroke_fill="white")
    width = max(840, pic.width)
    out = Image.new("RGB", (width, pic.height+55), "white")
    out.paste(pic, ((width-pic.width)//2, 55))
    ImageDraw.Draw(out).text((12,8), title, fill=(25,40,55), font=FONT)
    return out


def colored_crop(gray, labels, bounds, ids):
    # Highlight full components, even portions outside the exclusive cores.
    rgb = np.full((*gray.shape, 3), 255, dtype=np.uint8)
    rgb[labels > 0] = (200,200,200)
    for i, component in enumerate(sorted(ids)):
        rgb[labels == component] = COLORS[i % len(COLORS)]
    return Image.fromarray(rgb).crop(bounds)


def render_span(row, record, original, labels, mapping, detections, out):
    edges = row["edges"]
    mapped = [(i,mapping[i], detections[mapping[i].detection_index] if mapping[i].detection_index is not None else None)
              for i in range(row["start"],row["end"])]
    boxes = [d["box_xyxy"] for _,_,d in mapped if d]
    if not boxes:
        bounds = (0,0,original.width,original.height)
    else:
        # Keep whole line height, preserving ascenders, dots and descenders.
        bounds = (max(0,int(np.floor(min(b[0] for b in boxes)))-28),0,
                  min(original.width,int(np.ceil(max(b[2] for b in boxes)))+28),original.height)
    raw = original.crop(bounds)
    panels = [panel(raw, f"{row['span_id']}  token={row['token']!r}  group={row['group']} | RAW IMAGE")]
    annotations = []
    for n,(i,alignment,detection) in enumerate(mapped):
        if detection:
            annotations.append((translated_box(detection["box_xyxy"],bounds),COLORS[n % len(COLORS)],
                                f"{i}:{record['transcription'][i]}={detection['predicted_char']}"))
    panels.append(panel(raw,"DTLR full boxes; labels are GT offset:character=prediction",annotations))
    gray = np.asarray(original.convert("L"))
    dom_ids = {e[k] for e in edges for k in ("left_dominant_component_id","right_dominant_component_id") if e.get(k)}
    overview = colored_crop(gray,labels,bounds,dom_ids)
    legend = ", ".join(str(i) for i in sorted(dom_ids))
    panels.append(panel(overview,"CCL components in cores (other ink gray); component IDs: " + legend))
    for e in edges:
        ids = {e[k] for k in ("left_dominant_component_id","right_dominant_component_id") if e.get(k)}
        diagnostic = colored_crop(gray,labels,bounds,ids)
        rectangles = []
        for side,color in (("left",(45,100,190)),("right",(180,90,40))):
            b = e.get(side+"_exclusive_core_xyxy")
            if b:
                rectangles.append((translated_box(b,bounds),color,side[0].upper()))
        title = (f"Pair {e['pair']!r} at {e['left_gt_index']} | "
                 f"IDs {e.get('left_dominant_component_id')}/{e.get('right_dominant_component_id')} | "
                 f"match {e['left_alignment']}/{e['right_alignment']} | usable={e['usable']}")
        panels.append(panel(diagnostic,title,rectangles))
    width,height = max(p.width for p in panels),sum(p.height for p in panels)+8*(len(panels)-1)
    canvas = Image.new("RGB",(width,height),(230,235,238))
    y=0
    for p in panels:
        canvas.paste(p,((width-p.width)//2,y)); y+=p.height+8
    image = f"crops/{row['line_id']}_{row['start']:03d}_{row['end']:03d}.png"
    canvas.save(out/image)
    raw_path = image.replace(".png","_raw.png")
    raw.resize((raw.width*3,raw.height*3),Image.Resampling.NEAREST).save(out/raw_path)
    row.update({"image":image,"raw_image":raw_path,"crop_bounds_xyxy":list(bounds),
                "character_mapping":[{"gt_index":i,"character":record['transcription'][i],
                    "operation":a.operation,"detection_index":a.detection_index,
                    "box_xyxy":d['box_xyxy'] if d else None} for i,a,d in mapped]})


def write_html(queue,out):
    cards=[]
    for number,row in enumerate(queue['rows'],1):
        meta={k:row[k] for k in ('span_id','token','group','model_membership','emitted_by_model_here',
                                'threshold_diagnostic','source_image_sha256')}
        cards.append(f"<article id='case-{number}'><h2>{number}. {html.escape(row['token'])} — {html.escape(row['group'])}</h2>"
                     f"<pre>{html.escape(json.dumps(meta,indent=2))}</pre>"
                     f"<img loading='lazy' src='{row['image']}' alt='Handwriting and localization diagnostics'>"
                     "<p>Review: character alignment; visible main-body connections; dominant-core attribution; ambiguous strokes/dots.</p></article>")
    document="""<!doctype html><html><head><meta charset='utf-8'><title>IAM training span QA pilot</title>
<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#162c3d}img{max-width:100%;height:auto}article{border-top:2px solid #ddd;padding:24px 0}pre{white-space:pre-wrap;background:#f1f5f6;padding:16px}nav a{margin-right:12px}h1{font-size:32px}</style></head><body>
<h1>IAM training: longer handwriting spans</h1>
<p>Fixed stratified diagnostic pilot. Training images only; no tokenizer fitting or threshold changes. These examples do not estimate population accuracy. Human labels remain pending; an AI-assisted inspection is recorded separately.</p>
<p>The first panel is the unmarked image. Later panels show approximate DTLR localization and the ink components used by the rule. Component IDs are local to each image. Pair-facing cores for the same interior character may differ.</p>
<nav>__NAV__</nav>__CARDS__</body></html>"""
    document=document.replace('__NAV__',' '.join(f"<a href='#case-{i}'>{i}</a>" for i in range(1,len(cards)+1))).replace('__CARDS__','\n'.join(cards))
    (out/'index.html').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iam-train-dir',type=Path,required=True)
    parser.add_argument('--iam-image-dir',type=Path,required=True,help='IAM_new/data/imgs/lines directory only')
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',default='iam-long-span-qa-pilot-20261008-v1')
    parser.add_argument('--per-stratum',type=int,default=2)
    parser.add_argument('--stress-count',type=int,default=4)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(REPO/'outputs'):
        raise ValueError('generated review must stay in DTLR/outputs')
    if out.exists():
        raise ValueError('use a new output directory to preserve the frozen queue')
    if not args.model_dir.resolve().is_relative_to(REPO/'outputs'):
        raise ValueError('use prototype models inside DTLR/outputs')
    image_root=args.iam_image_dir.resolve()
    if tuple(image_root.parts[-4:]) != ('IAM_new','data','imgs','lines'):
        raise ValueError('image directory must be IAM_new/data/imgs/lines')
    models={name:json.loads((args.model_dir/f'hw-{name}.json').read_text()) for name in ('bpe','unigram')}
    provenance=models['bpe']['evidence_provenance']
    if models['unigram']['evidence_provenance']['source_sha256'] != provenance['source_sha256']:
        raise ValueError('model evidence snapshots differ')
    paths={'selection':args.iam_train_dir/'selection.json','detections':args.iam_train_dir/'detections.jsonl',
           'evidence':args.iam_train_dir/'bigrams-dominant-core-v3/bigram_evidence.json',
           'manifest':args.iam_train_dir/'bigrams-dominant-core-v3/manifest.json',
           'scores':args.iam_train_dir/'bigrams-dominant-core-v3/bigram_scores.json',
           'frozen_bigram':REPO/'poc/frozen/iam-english-handwriting-bigram-v1.json'}
    hashes={k:sha256_file(p) for k,p in paths.items()}
    if hashes != provenance['source_sha256']:
        raise ValueError('input hashes differ from prototype evidence snapshot')
    records=[json.loads(line) for line in paths['detections'].read_text().splitlines() if line]
    evidence=json.loads(paths['evidence'].read_text())
    corpus=prepare_corpus(records,evidence,json.loads(paths['selection'].read_text()),json.loads(paths['manifest'].read_text()))
    policy=models['bpe']['candidate_policy']
    candidates=collect_candidates(corpus,policy['max_token_length'],policy['minimum_count'],policy['rate_threshold'])
    if candidates != json.loads((args.model_dir/'candidates.json').read_text()):
        raise ValueError('candidate inventory did not reconstruct exactly')
    tokens={t for m in models.values() for t in m['vocab'] if len(t)>2}
    if not tokens <= set(candidates):
        raise ValueError('model tokens outside evidence candidate inventory')
    selected,pool_counts=select_span_review(corpus,tokens,args.seed,args.per_stratum,args.stress_count)
    by_id={r['line_id']:r for r in records}
    edge_lookup={(e['line_id'],e['left_gt_index']):e for e in evidence}
    from dtlr_poc.handwriting_subwords import HandwritingSubwordTokenizer
    tokenizers={n:HandwritingSubwordTokenizer(m) for n,m in models.items()}
    for row in selected:
        row['edges']=[edge_lookup[(row['line_id'],i)] for i in range(row['start'],row['end']-1)]
        row['model_membership']=[n for n,m in models.items() if row['token'] in m['vocab']]
        text=by_id[row['line_id']]['transcription']
        row['text_context']=text[max(0,row['start']-15):row['end']+15]
        row['emitted_by_model_here']={n:any(s['start']==row['start'] and s['end']==row['end']
                    for s in tokenizer.segment(text)['segments']) for n,tokenizer in tokenizers.items()}
    queue={'schema_version':'dtlr.iam-span-review-queue.v1','dataset':'IAM','split':'train','seed':args.seed,
           'status':'frozen-sample-before-image-inspection','per_length_status':args.per_stratum,
           'interior_disagreement_requested':args.stress_count,'source_sha256':hashes,
           'model_sha256':{n:sha256_file(args.model_dir/f'hw-{n}.json') for n in models},
           'source_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
           'candidate_length_counts':dict(sorted(Counter(map(len,tokens)).items())),
           'pool_counts':pool_counts,'rows':selected,
           'note':'Train-only stratified pilot. No population accuracy or independent human validation.'}
    queue['implementation_sha256'] = {str(p.relative_to(REPO)):sha256_file(p) for p in (Path(__file__).resolve(), REPO/'poc/dtlr_poc/span_review.py')}
    out.mkdir(parents=True); (out/'crops').mkdir(); (out/'context').mkdir()
    write_json(out/'sampling_queue.json',queue)
    queue_hash=sha256_file(out/'sampling_queue.json')
    print(f'Frozen {len(selected)} spans from {len(set(r["line_id"] for r in selected))} train images.',flush=True)
    del evidence,edge_lookup,records,corpus;gc.collect()
    verified=missing_verified=0
    for line_id in sorted({r['line_id'] for r in selected}):
        record=by_id[line_id]
        relative=Path(record['image_relpath'])
        if tuple(relative.parts[:4])!=('IAM_new','data','imgs','lines') or len(relative.parts)!=5 or relative.name!=line_id+'.jpg':
            raise ValueError('unexpected IAM image path')
        image_path=(image_root/relative.name).resolve()
        if not image_path.is_relative_to(image_root):
            raise ValueError('image path escapes IAM image directory')
        source_hash=sha256_file(image_path)
        original=Image.open(image_path).convert('RGB')
        if original.size!=(record['image_width'],record['image_height']):
            raise ValueError('image dimensions differ from export')
        original.save(out/'context'/(line_id+'.png'))
        gray=np.asarray(original.convert('L'));threshold=otsu_threshold(gray)
        labels,count=label_ink(gray,threshold)
        detections=sorted(record['detections'],key=lambda d:d['box_xyxy'][0])
        mapping=gt_detection_map(record['transcription'],[d['predicted_char'] for d in detections])
        relevant=[r for r in selected if r['line_id']==line_id]
        for row in relevant:
            row['source_image_sha256']=source_hash
            row['otsu_threshold']=threshold
            row['context_image']='context/'+line_id+'.png'
            diagnostics={}
            for offset in (-10,0,10):
                candidate_labels=labels if offset==0 else label_ink(gray,max(0,min(255,threshold+offset)))[0]
                computed=[]
                for e in row['edges']:
                    i=e['left_gt_index']; a,b=mapping[i],mapping[i+1]
                    if a.detection_index is None or b.detection_index is None:
                        if offset==0:
                            if (e['left_alignment']!=a.operation or e['right_alignment']!=b.operation
                                    or e['usable'] or e['connected'] is not None or e['ccl_component_count']!=count):
                                raise ValueError('missing detection evidence mismatch')
                            missing_verified+=1
                        computed.append({'dominant_core_usable':False});continue
                    lb,rb=detections[a.detection_index]['box_xyxy'],detections[b.detection_index]['box_xyxy']
                    c=pair_component_evidence(candidate_labels,lb,rb);computed.append(c)
                    if offset==0:
                        expected={'left_dominant_component_id':c['left_dominant_component'],
                                  'right_dominant_component_id':c['right_dominant_component'],
                                  'left_exclusive_core_xyxy':c['left_core_box'],'right_exclusive_core_xyxy':c['right_core_box'],
                                  'left_core_component_pixel_counts':c['left_core_component_pixel_counts'],
                                  'right_core_component_pixel_counts':c['right_core_component_pixel_counts'],
                                  'usable':c['dominant_core_usable'],'connected':c['connected_dominant_core_v3'],
                                  'left_box_xyxy':lb,'right_box_xyxy':rb,'ccl_component_count':count,
                                  'left_alignment':a.operation,'right_alignment':b.operation}
                        if any(normalize(e.get(k))!=normalize(v) for k,v in expected.items()):
                            raise ValueError(f"image evidence mismatch for {line_id}:{i}")
                        verified+=1
                exact=all(e['left_alignment']==e['right_alignment']=='match' for e in row['edges'])
                diagnostics[str(offset)]=decision(computed) if exact else None
            expected_status=True if row['group']=='supported' else None if row['group']=='abstained' else False
            if diagnostics['0'] is not expected_status:
                raise ValueError('recomputed span decision differs from queue')
            row['threshold_diagnostic']=diagnostics
            render_span(row,record,original,labels,mapping,detections,out)
        print(f'Rendered {line_id}: {len(relevant)} span(s)',flush=True)
    queue['sampling_queue_sha256']=queue_hash
    queue['status']='rendered-evidence-reproduced-human-review-pending'
    queue['recomputed_pair_checks']=verified
    queue['recomputed_missing_pair_checks']=missing_verified
    queue['threshold_changed_decisions']=sum(len(set(r['threshold_diagnostic'].values()))>1 for r in selected)
    write_json(out/'review_queue.json',queue)
    write_json(out/'human_review_template.json',{'schema_version':'dtlr.iam-span-human-review.v1',
        'sampling_queue_sha256':queue_hash,'reviewer':None,'annotations':{r['span_id']:{'alignment':'pending',
            'visual_main_body_connectivity':'pending','span_rule_assessment':'pending','notes':''} for r in selected}})
    write_html(queue,out)
    print(json.dumps({k:v for k,v in queue.items() if k not in ('rows','source_sha256','model_sha256','pool_counts')},indent=2))


if __name__=='__main__':
    main()
