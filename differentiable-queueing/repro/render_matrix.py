"""
Render reproducibility_matrix.csv + status_summary.md from repro/status.json
"""
import json, csv, os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_PATH = os.path.join(THIS_DIR, 'status.json')
MATRIX_CSV = os.path.join(THIS_DIR, 'reproducibility_matrix.csv')
SUMMARY_MD = os.path.join(THIS_DIR, 'STATUS_SUMMARY.md')

STATUS_BADGE = {
    'pass': '✅',
    'pass_with_caveat': '✅⚠️',
    'pass_last_iterate': '✅',
    'noisy': '⚠️',
    'methodology_mismatch': '❌',
    'scheduled': '🔄',
    'running': '🔄',
    'blocked': '🚫',
}


def main():
    with open(STATUS_PATH) as f:
        d = json.load(f)
    arts = d['artifacts']

    # CSV
    with open(MATRIX_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['artifact_id', 'paper_section', 'status', 'driver', 'source', 'job_ids'])
        for a in arts:
            w.writerow([
                a['id'],
                a.get('paper_section', ''),
                a['status'],
                a.get('driver', ''),
                a.get('source', ''),
                ','.join(str(j) for j in a.get('job_ids', [])),
            ])

    # Summary
    counts = {}
    for a in arts:
        counts[a['status']] = counts.get(a['status'], 0) + 1

    lines = []
    lines.append('# Reproduction Status Summary')
    lines.append('')
    lines.append(f"Project: **{d['project']['name']}**  ·  Paper: `{d['project']['paper']['id']}`  ·  Updated: {d['last_updated']}")
    lines.append('')
    lines.append('## Status distribution')
    lines.append('')
    lines.append('| Status | Count | Meaning |')
    lines.append('|---|---:|---|')
    meaning = {
        'pass':                 'Numerical match within acceptance window',
        'pass_with_caveat':     'Pass but with documented limitation',
        'pass_last_iterate':    'Trend matches; uses last-iterate vs paper avg-iterate',
        'noisy':                'High variance, compute budget too small',
        'methodology_mismatch': 'Our test setup doesn\'t exercise the paper\'s exact claim',
        'scheduled':            'Set up, waiting to run',
        'running':              'Currently executing',
        'blocked':              'External blocker (hardware / access)',
    }
    for s, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f'| {STATUS_BADGE.get(s, "❓")} {s} | {n} | {meaning.get(s, "")} |')
    lines.append(f'| **TOTAL** | **{len(arts)}** | |')
    lines.append('')

    lines.append('## Per-artifact table')
    lines.append('')
    lines.append('| Artifact | §Section | Status | Result / Note |')
    lines.append('|---|---|---|---|')
    for a in arts:
        badge = STATUS_BADGE.get(a['status'], '❓')
        result = a.get('result', a.get('caveat', a.get('next_step', a.get('issue', ''))))
        if isinstance(result, dict):
            result_str = ', '.join(f'{k}={v}' for k, v in list(result.items())[:3])
        else:
            result_str = str(result)[:120]
        lines.append(f"| `{a['id']}` | {a.get('paper_section', '')} | {badge} {a['status']} | {result_str} |")
    lines.append('')

    if d.get('open_followups'):
        lines.append('## Open follow-ups')
        lines.append('')
        for f in d['open_followups']:
            lines.append(f"- **{f['id']}** — {f['note']}")
        lines.append('')

    if d.get('blockers_resolved'):
        lines.append('## Blockers resolved')
        lines.append('')
        for b in d['blockers_resolved']:
            lines.append(f"- **{b['id']}** ({b['resolved_on']}) — {b['note']}")
        lines.append('')

    with open(SUMMARY_MD, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Wrote: {MATRIX_CSV} ({len(arts)} rows)')
    print(f'Wrote: {SUMMARY_MD}')


if __name__ == '__main__':
    main()
