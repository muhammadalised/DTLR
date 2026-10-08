#!/usr/bin/env python3
"""Fit matched IAM text controls, then compare against unchanged HW prototypes."""
import argparse
from collections import Counter, defaultdict
import gc
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'poc'))
from dtlr_poc.handwriting_subwords import (
    HandwritingSubwordTokenizer, collect_text_candidates, compare_tokenizers,
    prepare_corpus, span_connected, train_bpe, train_unigram,
)
from dtlr_poc.selection import sha256_file, text_sha256
from dtlr_poc.tokenizer import HandwritingBigramTokenizer

REFERENCE_HASHES = {
    'hw-bpe.json': '9dedba1087e8e011633251b22159f857fb2e80babc15c877f7ffb03d7a985aad',
    'hw-unigram.json': '1ea98cc926fa3658dc6455a8d9e24ebb3a42b72e15848e9aca0bb7bfa00753c5',
    'hw-bigram-iam-train-alphabet.json': '06da23d5b33e743e66b943385eb1930146ed47843ffaaf16ded75980698411ac',
    'comparison.json': 'c7db4f3384e9e929cdf73d7379537f8df0b3f2511477dd8679180d9d8b1882ae',
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')


def length_support(corpus, tokenizer):
    counts = defaultdict(Counter)
    for line in corpus:
        for segment in tokenizer.segment(line['text'])['segments']:
            length = len(segment['token'])
            if length < 2:
                continue
            connected = span_connected(line, segment['start'], segment['end'])
            counts[length]['emitted'] += 1
            if connected is not None:
                counts[length]['known'] += 1
                counts[length]['supported'] += int(connected)
    return {length:{**count, 'abstained':count['emitted']-count['known'],
                    'support_rate':count['supported']/count['known'] if count['known'] else None}
            for length,count in sorted(counts.items())}


def percent(value):
    return 'n/a' if value is None else f'{value:.2%}'


def report_text(report):
    labels = {'hw_bigram':'HW bigram (derived alphabet)', 'hw_bpe':'HW BPE', 'text_bpe':'Text BPE',
              'hw_unigram':'HW Unigram', 'text_unigram':'Text Unigram'}
    rows = ['# IAM training comparison: handwriting and text controls', '',
        'All five models use the same 5,694 IAM training lines, 79 singleton characters,',
        '145 multi-character pieces, and CTC blank ID 0: 225 classes. The frozen',
        '222-class bigram artifact remains unchanged; the table uses its previously',
        'labelled training-alphabet extension. These are descriptive training-set',
        'measurements, not held-out recognition results.', '',
        '| Model | Tokens | Reduction | Connected edges cut | Whole-span support |',
        '|---|---:|---:|---:|---:|']
    for name,metrics in report['tokenizers'].items():
        rows.append(f"| {labels[name]} | {metrics['token_count']} | {percent(metrics['token_reduction_fraction'])} | "
                    f"{percent(metrics['connected_edge_cut_fraction'])} | {percent(metrics['multi_token_same_component_fraction'])} |")
    rows += ['', 'Connected-edge cuts divide by the same 78,957 observed connected edges.',
             'Whole-span support excludes abstentions. Token lengths and selected occurrences',
             'differ between models; raw support rates are not a controlled causal effect.',
             'Both metrics describe the existing automatic evidence, with its localization',
             'limitations. No CER/WER or recognition improvement is established.', '',
             '## Matched controls', '',
             'BPE uses the same ordered-merge encoder for both conditions; the text control',
             'ranks current pairs by raw occurrence count, while HW BPE uses connected support.',
             'Unigram uses the same EM, removal-loss pruning, and Viterbi encoder for both;',
             'the text control seeds and initializes pieces from text frequency, while the',
             'HW model admits and initializes pieces using handwriting evidence.', '',
             'Both use letter-only spans of length 2–6, minimum count 20, 145 pieces,',
             'eight EM iterations per stage, seed cap 580, and shrink factor 0.75.',
             'The count criterion differs deliberately: raw text frequency versus exact,',
             'usable image observations plus connectivity rate at least 0.5. This compares',
             'the complete handwriting-aware selection policy with a text-only policy;',
             'it does not isolate connectivity from evidence availability. The Unigram',
             'controls share a seed cap, not an identical initial inventory size.', '',
             '## Support within token lengths', '',
             '| Model | Length | Emitted | Known | Supported | Support rate |',
             '|---|---:|---:|---:|---:|---:|']
    for name,by_length in report['support_by_token_length'].items():
        for length,counts in by_length.items():
            rows.append(f"| {labels[name]} | {length} | {counts['emitted']} | {counts['known']} | "
                        f"{counts['supported']} | {percent(counts['support_rate'])} |")
    rows += ['', 'Length stratification reduces one confound, but the underlying token strings',
             'and occurrences still differ. This table is not held-out visual accuracy.', '',
             '## Vocabulary overlap', '', '| Family | Shared pieces / 145 | HW-only | Text-only |', '|---|---:|---:|---:|']
    for name,data in report['vocabulary_overlap'].items():
        rows.append(f"| {name} | {len(data['shared'])} | {len(data['handwriting_only'])} | {len(data['text_only'])} |")
    rows += ['', '## Illustrative text examples', '']
    for example in report['examples']:
        rows.append(f"Text: `{example['text']}`")
        rows.append('')
        for name,tokens in example['tokens'].items():
            visible=['SPACE' if t==' ' else t for t in tokens]
            rows.append(f"- {labels[name]}: `{' | '.join(visible)}`")
        rows.append('')
    rows += ['## Verification and limits', '',
             'Every model round-trips every IAM training line without emitting blank.',
             'The prior HW comparison metrics reproduce exactly. Inputs and implementation',
             'hashes, output hashes, vocabulary differences, first merges, and denominators',
             'are retained in the JSON report. No parameters were chosen by comparison outcomes.',
             'All prototypes remain unfrozen. No downstream dataset or TVA file is used.', '']
    return '\n'.join(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iam-train-dir',type=Path,required=True)
    parser.add_argument('--hw-model-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(REPO/'outputs') or out.exists():
        raise ValueError('use a new output directory within DTLR/outputs')
    if not args.hw_model_dir.resolve().is_relative_to(REPO/'outputs'):
        raise ValueError('HW reference models must be inside DTLR/outputs')
    reference_hashes={name:sha256_file(args.hw_model_dir/name) for name in REFERENCE_HASHES}
    if reference_hashes != REFERENCE_HASHES:
        raise ValueError('HW reference snapshot hashes differ')
    hw_models={name:json.loads((args.hw_model_dir/f'hw-{name}.json').read_text()) for name in ('bpe','unigram')}
    source_expected=hw_models['bpe']['evidence_provenance']['source_sha256']
    source=args.iam_train_dir
    paths={'selection':source/'selection.json','detections':source/'detections.jsonl',
           'manifest':source/'bigrams-dominant-core-v3/manifest.json',
           'scores':source/'bigrams-dominant-core-v3/bigram_scores.json',
           'evidence':source/'bigrams-dominant-core-v3/bigram_evidence.json',
           'frozen_bigram':REPO/'poc/frozen/iam-english-handwriting-bigram-v1.json'}
    if {k:sha256_file(p) for k,p in paths.items()} != source_expected:
        raise ValueError('IAM evidence input hashes differ')
    selection=json.loads(paths['selection'].read_text())
    records=[json.loads(line) for line in paths['detections'].read_text().splitlines() if line]
    if (selection['dataset']!='IAM' or selection['split']!='train' or len(records)!=selection['requested_count']
            or len(records)!=len(selection['lines'])):
        raise ValueError('not the pinned IAM training selection')
    text_corpus=[]
    for record,item in zip(records,selection['lines']):
        if (record['dataset']!='IAM' or record['split']!='train' or record['line_id']!=item['id']
                or text_sha256(record['transcription'])!=item['transcription_sha256']):
            raise ValueError('IAM training line identity/text mismatch')
        text_corpus.append({'line_id':record['line_id'],'text':record['transcription']})
    singletons={c for line in text_corpus for c in line['text']}
    policy={'dataset':'IAM','split':'train','max_token_length':6,'minimum_text_count':20,
            'pair_policy':'letters-only','subword_budget':145,'singleton_count':79,'ctc_classes':225,
            'em_iterations':8,'seed_multiplier':4,'shrink_factor':0.75,
            'normalization':'NFC','selection_evidence':'training text only',
            'settings_status':'predeclared inherited prototype settings, not tuned or validated'}
    if len(singletons)!=79:
        raise ValueError('unexpected IAM alphabet')
    out.mkdir(parents=True)
    implementation_files=[REPO/'poc/dtlr_poc/handwriting_subwords.py',Path(__file__).resolve()]
    implementation_hashes={str(p.relative_to(REPO)):sha256_file(p) for p in implementation_files}
    training_provenance={'source_selection_sha256':source_expected['selection'],
                         'source_transcription_export_sha256':source_expected['detections'],
                         'line_count':len(text_corpus),'implementation_sha256':implementation_hashes,
                         'source_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
                         'worktree_status':subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).splitlines(),
                         'source_snapshot':'source-snapshot/','fit_inputs':'line IDs and transcriptions only; no boxes, edges, or CCL statistics',
                         'python_version':sys.version}
    write_json(out/'predeclared_policy.json',{'policy':policy,'training_provenance':training_provenance,
                                            'hw_reference_sha256':reference_hashes})
    snapshot=out/'source-snapshot';snapshot.mkdir()
    for path in implementation_files:
        shutil.copyfile(path,snapshot/path.name)
    candidates=collect_text_candidates(text_corpus,6,20)
    write_json(out/'text_candidates.json',candidates)
    print(f"Pinned {len(text_corpus)} train lines; {len(candidates)} text-only candidates; policy saved before fitting.",flush=True)
    print('Training text-only BPE...',flush=True)
    bpe=train_bpe(text_corpus,candidates,145,singletons,handwriting_aware=False)
    print('Training text-only Unigram...',flush=True)
    unigram=train_unigram(text_corpus,candidates,145,singletons,8,4,0.75,handwriting_aware=False)
    for name,model in (('bpe',bpe),('unigram',unigram)):
        if model['size']!=225 or len(model['vocabulary'])!=145:
            raise ValueError('control did not attain matched vocabulary budget')
        model['candidate_policy']=policy
        model['training_provenance']=training_provenance
        write_json(out/f'text-{name}.json',model)
    print('Controls fitted without handwriting edges. Loading CCL evidence for descriptive comparison...',flush=True)
    evidence=json.loads(paths['evidence'].read_text())
    corpus=prepare_corpus(records,evidence,selection,json.loads(paths['manifest'].read_text()))
    del records,evidence,text_corpus;gc.collect()
    tokenizers={'hw_bigram':HandwritingBigramTokenizer(json.loads((args.hw_model_dir/'hw-bigram-iam-train-alphabet.json').read_text())),
                'hw_bpe':HandwritingSubwordTokenizer(hw_models['bpe']),'text_bpe':HandwritingSubwordTokenizer(bpe),
                'hw_unigram':HandwritingSubwordTokenizer(hw_models['unigram']),'text_unigram':HandwritingSubwordTokenizer(unigram)}
    report=compare_tokenizers(corpus,tokenizers)
    previous=json.loads((args.hw_model_dir/'comparison.json').read_text())['tokenizers']
    for name in ('hw_bigram','hw_bpe','hw_unigram'):
        if json.loads(json.dumps(report['tokenizers'][name]))!=previous[name]:
            raise ValueError('existing HW metrics changed')
    if any(m['round_trip_or_blank_failures'] for m in report['tokenizers'].values()):
        raise ValueError('round-trip or blank-emission failure')
    report['support_by_token_length']={n:length_support(corpus,t) for n,t in tokenizers.items()}
    report['vocabulary_overlap']={}
    for name,text_model in (('bpe',bpe),('unigram',unigram)):
        hw={t for t in hw_models[name]['vocab'] if len(t)>1}
        text={t for t in text_model['vocab'] if len(t)>1}
        report['vocabulary_overlap'][name]={'shared':sorted(hw&text),'handwriting_only':sorted(hw-text),'text_only':sorted(text-hw)}
    report['first_bpe_merges']={'handwriting':hw_models['bpe']['merges'][:10],'text_only':bpe['merges'][:10]}
    report['examples']=[{'text':text,'scope':'illustrative, not evaluation',
                         'tokens':{n:t.segment(text)['tokens'] for n,t in tokenizers.items()}}
                        for text in ('the handwriting','minimum evidence','better letter','the, # & (')]
    report['control_policy']=policy
    report['text_candidates_by_length']=dict(sorted(Counter(map(len,candidates)).items()))
    report['training_provenance']=training_provenance
    report['evaluation_evidence_sha256']=source_expected
    report['hw_reference_sha256']=reference_hashes
    report['text_model_sha256']={n:sha256_file(out/f'text-{n}.json') for n in ('bpe','unigram')}
    report['limitations']=['IAM train descriptive only; no independent visual or recognition accuracy.',
                           'Variable token lengths and different occurrences confound raw span-support percentages.',
                           'Raw text count versus exact/usable evidence count means the entire HW policy is compared.',
                           'This is the same custom Unigram implementation, not SentencePiece.']
    write_json(out/'comparison.json',report)
    markdown=report_text(report)
    (out/'comparison.md').write_text(markdown,encoding='utf-8')
    # Offline page uses the same report text; no network, scripts, or remote assets.
    (out/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>IAM tokenizer controls</title>'
         '<style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px system-ui;color:#162c3d}pre{white-space:pre-wrap;line-height:1.6}</style>'
         '<h1>IAM handwriting-aware and text-only tokenizers</h1><pre>'+html.escape(markdown)+'</pre>',encoding='utf-8')
    write_json(out/'output_hashes.json',{p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='output_hashes.json'})
    print(json.dumps({'tokenizers':report['tokenizers'],'text_candidates_by_length':report['text_candidates_by_length'],
                      'shared_vocabulary_counts':{n:len(v['shared']) for n,v in report['vocabulary_overlap'].items()}},indent=2))


if __name__=='__main__':
    main()
