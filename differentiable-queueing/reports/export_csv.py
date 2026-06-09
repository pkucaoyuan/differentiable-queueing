# -*- coding: utf-8 -*-
"""Export same-name CSV files for every figure (paper-tier reproducibility:
each PDF figure has a CSV with the underlying numbers)."""
import json, csv, os, statistics
import numpy as np

OUTDIR = 'reports/figures'
os.makedirs(OUTDIR, exist_ok=True)


def write_csv(path, headers, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)
    print(f'  wrote {path}')


# ─── §5.2 CμRule papergrid ───
pw = json.load(open('results/reproduction/cmu_papergrid_pathwise.json'))
rf = json.load(open('results/reproduction/cmu_papergrid_reinforce.json'))
rows = []
for a in sorted(pw.keys(), key=float):
    for gp in sorted(pw[a].keys(), key=lambda x: -float(x)):
        pw_costs = [r['avg_cost'] for r in pw[a][gp]]
        rf_costs = [r['avg_cost'] for r in rf[a][gp]]
        rows.append([a, gp,
                     statistics.mean(pw_costs), statistics.stdev(pw_costs),
                     statistics.mean(rf_costs), statistics.stdev(rf_costs),
                     (statistics.mean(rf_costs) - statistics.mean(pw_costs)) / statistics.mean(pw_costs) * 100])
write_csv(f'{OUTDIR}/fig_section52_cmu_papergrid.csv',
          ['alpha', 'gap', 'PW_mean', 'PW_std', 'RF_mean', 'RF_std', 'RF_PW_diff_pct'], rows)


# ─── §5.3 admission summary ───
summ = json.load(open('results/admission_control_summary.json'))
rows = []
for env in sorted(summ.keys()):
    row = [env]
    for m in ['PATHWISE_B1', 'SPSA_B10', 'SPSA_B100', 'SPSA_B1000']:
        v = summ[env].get(m, {})
        row += [v.get('mean'), v.get('std')]
    rows.append(row)
write_csv(f'{OUTDIR}/fig_section53_admission.csv',
          ['env', 'PW_B1_mean', 'PW_B1_std', 'SPSA_B10_mean', 'SPSA_B10_std',
           'SPSA_B100_mean', 'SPSA_B100_std', 'SPSA_B1000_mean', 'SPSA_B1000_std'], rows)


# ─── §6 WC vs Vanilla ───
wc = json.load(open('loss/criss_cross_bh_ppg_softmax.json'))
van = json.load(open('loss/criss_cross_bh_ppg_vanilla.json'))
rows = []
for r in wc:
    rows.append([r['epoch'], 'WC',      r['test_loss'], r['train_loss'], r['test_loss_std']])
for r in van:
    rows.append([r['epoch'], 'Vanilla', r['test_loss'], r['train_loss'], r['test_loss_std']])
write_csv(f'{OUTDIR}/fig_section6_wc_vs_vanilla.csv',
          ['epoch', 'variant', 'test_loss', 'train_loss', 'test_loss_std'], rows)


# ─── §7 STE training curves ───
envs = ['criss_cross_bh', 'reentrant_2', 'reentrant_3', 'reentrant_4', 'reentrant_5',
        'reentrant_6', 'reentrant_7', 'reentrant_8', 'reentrant_9', 'reentrant_10']
rows = []
for env in envs:
    d = json.load(open(f'loss/{env}_ppg_softmax.json'))
    for r in d:
        rows.append([env, r['epoch'], r['test_loss'], r['train_loss'], r['test_loss_std']])
write_csv(f'{OUTDIR}/fig_section7_training_curves.csv',
          ['env', 'epoch', 'test_loss', 'train_loss', 'test_loss_std'], rows)


# ─── §7 STE vs cμ ───
d = json.load(open('results/reproduction/ste_vs_cmu_benchmark.json'))
rows = []
for env in d.keys():
    r = d[env]
    rows.append([env, r['cmu_mean'], r['cmu_std'], r['ste_mean'], r['ste_std'],
                 r['improvement_pct'], r['eval_batch'], r['eval_T']])
write_csv(f'{OUTDIR}/fig_section7_ste_vs_cmu.csv',
          ['env', 'cmu_mean', 'cmu_std', 'ste_mean', 'ste_std',
           'improvement_pct', 'eval_batch', 'eval_T'], rows)


# ─── §4.3.1 GPU benchmark ───
d_small = json.load(open('results/reproduction/gpu_benchmark.json'))
d_large = json.load(open('results/reproduction/gpu_benchmark_large.json'))
rows = []
for r in d_small['results']:
    rows.append([r['device'], r['batch_size'], r['median_time_s'], r['throughput_events_per_s']])
for r in d_large['results']:
    rows.append(['cuda', r['batch_size'], r['median_time_s'], r['throughput_events_per_s']])
write_csv(f'{OUTDIR}/fig_section4_3_1_gpu_benchmark.csv',
          ['device', 'batch_size', 'median_time_s', 'throughput_events_per_s'], rows)


# ─── E5 heavy traffic ───
import re
text = open('logs/direct/E5_heavy_traffic.out').read()
sections = re.split(r'--- (Gap = [\d.]+) ---', text)
rows = []
for i in range(1, len(sections), 2):
    label = sections[i]
    body = sections[i+1]
    if 'PATHWISE' not in body:
        continue
    gap = float(label.replace('Gap = ', ''))
    for m in re.finditer(r'PATHWISE rho=([\d.]+): cost=([\d.]+)±([\d.]+)', body):
        rows.append([gap, float(m.group(1)), float(m.group(2)), float(m.group(3))])
write_csv(f'{OUTDIR}/fig_E5_heavy_traffic.csv',
          ['gap', 'rho', 'cost_mean', 'cost_std'], rows)


# ─── §8 Theorem 2 ───
d = json.load(open('results/reproduction/theorem2_validation_v2.json'))
rows = []
for k in sorted(d.keys()):
    if k == 'fit':
        continue
    r = d[k]
    rows.append([r['rho'], r['gap'], r['pw_var'], r['rf_var'], r['ratio']])
write_csv(f'{OUTDIR}/fig_section8_theorem2.csv',
          ['rho', 'gap_1_minus_rho', 'pw_variance', 'rf_variance', 'rf_over_pw_ratio'], rows)


# ─── §7 Polyak vs Last ───
d = json.load(open('results/reproduction/polyak_eval.json'))
rows = []
for env in d.keys():
    r = d[env]
    rows.append([env, r['last_iterate_cost'], r['polyak_avg_cost'], r['diff_pct'],
                 r['polyak_window_epochs']])
write_csv(f'{OUTDIR}/fig_section7_polyak_vs_last.csv',
          ['env', 'last_iterate_cost', 'polyak_avg_cost', 'diff_pct', 'polyak_window'], rows)


print('\nAll figure CSVs exported.')
