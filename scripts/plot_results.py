#!/usr/bin/env python3
"""Generate figures from audited CSV files, never from hard-coded benchmark values."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def main(results=None):
    root=Path(results or ROOT/'results'); dest=root/'figures';dest.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'budgetce-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    def rows(name):
        with (root/name).open(newline='') as f:return list(csv.DictReader(f))
    data=rows('matched_training.csv')
    labels=['Native','PyTorch\nchunk 1,024','CUDA\nchunk 1,024']
    vals=[float(x['tokens_per_s'])/1000 for x in data]
    lo=[v-float(x['min_tokens_per_s'])/1000 for v,x in zip(vals,data)]
    hi=[float(x['max_tokens_per_s'])/1000-v for v,x in zip(vals,data)]
    fig,ax=plt.subplots(figsize=(7.4,3.9))
    bars=ax.bar(labels,vals,yerr=[lo,hi],capsize=5)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=8)
    ax.set(ylabel='Training throughput (thousand tokens/s)',ylim=(0,56),title='Matched-chunk training confirmation')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.012,'Tesla T4 | S=1,024 | 3 repeats × 100 measured updates\nWhiskers: observed repeat min–max, not confidence intervals.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(dest/'matched_training.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,3.7))
    vals=[float(x['peak_allocated_mib']) for x in data]
    bars=ax.bar(labels,vals)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=5)
    ax.set(ylabel='Peak PyTorch allocated memory (MiB)',ylim=(0,1010),title='Whole-training memory footprint')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.016,'Maximum across three repeats; not whole-device usage or reserved memory.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(dest/'matched_memory.png',dpi=180);plt.close(fig)
    op=rows('operator_comparisons.csv')
    labs=[f"N={int(x['n']):,}\nV={int(x['v']):,}" for x in op]
    vals=[float(x['cuda1024_vs_native_speedup']) for x in op]
    fig,ax=plt.subplots(figsize=(8.0,3.65))
    bars=ax.bar(np.arange(len(vals)),vals)
    ax.bar_label(bars,labels=[f'{v:.2f}×' for v in vals],padding=4)
    ax.axhline(1,linestyle='--',linewidth=1)
    ax.set(xticks=np.arange(len(vals)),xticklabels=labs,ylabel='Throughput relative to native',ylim=(0,1.62),title='CUDA chunk 1,024: linear cross-entropy forward and backward')
    ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.014,'Hidden width 512; pooled three-block timings. Chunk 1,024 is not feasible under every memory allowance.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(dest/'operator_throughput.png',dpi=180);plt.close(fig)
    print('Generated 3 figures from audited records:',dest)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path);main(p.parse_args().results)
