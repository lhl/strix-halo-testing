#!/usr/bin/env python3
"""llama_cpp_bencher.py – hot‑patch

* **Fix**: default `--build-root` → `/home/lhl/llama.cpp`.
* **Fix**: use `df['mode']` (column) instead of attribute; eliminates `KeyError: False`.
"""
from __future__ import annotations

import argparse, datetime as dt, json, shlex, subprocess as sp, sys, textwrap, time, re
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# ---------------------------- BUILD DISCOVERY ---------------------------- #

def discover_builds(root: Path)->Dict[str,str]:
    return {p.parts[-4]: str(p) for p in root.glob('llama.cpp-*/build/bin/llama-bench')}

# ----------------------------- MEM MONITOR ------------------------------ #

def _first_int(s:str)->int:
    m=re.search(r"(\d+)",s);return int(m.group(1)) if m else 0

def _monitor(cmd:List[str],parse,field:str,stop:Event,peaks:Dict[str,int],intv:float):
    try:init=parse(sp.check_output(cmd,text=True))
    except Exception:init=0
    peak=init
    while not stop.is_set():
        try:cur=parse(sp.check_output(cmd,text=True));peak=max(peak,cur)
        except Exception:pass
        time.sleep(intv)
    peaks[field]=peak;peaks[field.replace('_peak_','_delta_')]=peak-init

def parse_rocm(out:str)->int:
    try:return int(out.splitlines()[1].split(',')[2])//(1024*1024)
    except:return 0

def parse_gtt(out:str)->int:
    return _first_int(next((l for l in out.splitlines() if re.match(r"^\s*GTT",l)),""))

def parse_meminfo(out:str)->int:
    tot=avail=0
    for l in out.splitlines():
        if l.startswith('MemTotal'):tot=_first_int(l)
        elif l.startswith('MemAvailable'):avail=_first_int(l)
    return (tot-avail)//1024

# ------------------------------ BENCH RUN ------------------------------- #

def run_bench(bin:str,model:str,flags:str,mode:str,val:int,gpu:bool,raw_sink,intv:float):
    if val==0:return{}
    bench_args=f"-p {val} -n 0" if mode=='pp' else f"-p 0 -n {val}"
    cmd=f"{shlex.quote(bin)} -m {shlex.quote(model)} {bench_args} {flags} -o jsonl"
    print("\n[RUN]",cmd)
    stop=Event();peaks={}
    ths=[Thread(target=_monitor,args=(['cat','/proc/meminfo'],parse_meminfo,'system_ram_peak_mib',stop,peaks,intv),daemon=True)]
    ths[0].start()
    if gpu:
        for c,p,f in ([['rocm-smi','--showmeminfo','vram','--csv'],parse_rocm,'vram_peak_mib'],
                      [['amdgpu_top','-d'],parse_gtt,'gtt_peak_mib']):
            t=Thread(target=_monitor,args=(c,p,f,stop,peaks,intv),daemon=True);t.start();ths.append(t)

    last=None
    proc=sp.Popen(shlex.split(cmd),stdout=sp.PIPE,stderr=sp.STDOUT,text=True)
    for line in proc.stdout:
        try:j=json.loads(line);last=j;raw_sink.write(line)
        except json.JSONDecodeError:sys.stdout.write(line)
    proc.wait();stop.set();[t.join() for t in ths]
    tps=last.get('avg_ts') if last else None
    return {'timestamp':dt.datetime.now().isoformat(),'mode':mode,'value':val,'tokens_per_sec':tps,'bench_raw':last,**peaks}

# ----------------------------- PLOTTING ---------------------------------- #

def comb_plot(df,metric,label,mode,out):
    fig,ax=plt.subplots(figsize=(8,5))
    subset=df[df['mode']==mode]
    for build,grp in subset.groupby('build'):
        ax.plot(grp['value'],grp[metric],marker='o',label=build)
    if ax.lines:
        ax.set_title(f"{label} – {mode.upper()} sweep");ax.set_xlabel('Tokens');ax.set_ylabel(label);ax.legend();ax.grid(True,alpha=.3,linestyle=':');fig.tight_layout();fig.savefig(out/f"{mode}_{metric}.png",dpi=150)
    plt.close(fig)

# ------------------------------ MAIN ------------------------------------- #

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,description='llama.cpp benchmark helper')
    ap.add_argument('-m','--model',required=True);ap.add_argument('-o','--outdir')
    ap.add_argument('-b','--builds',nargs='*');ap.add_argument('--build-root',type=Path,default=Path('/home/lhl/llama.cpp'))
    ap.add_argument('-p','--pp',nargs='*',type=int);ap.add_argument('-n','--tg',nargs='*',type=int);ap.add_argument('--flags',default='-fa 1')
    ap.add_argument('--skip-gpu-mon',action='store_true');ap.add_argument('--list-builds',action='store_true');ap.add_argument('--interval',type=float,default=.2)
    args=ap.parse_args()

    found=discover_builds(args.build_root)
    if args.list_builds:
        print('\n'.join(f"{k}: {v}" for k,v in found.items()));return
    sel=args.builds or list(found.keys());builds={n:found.get(n,n) for n in sel}
    out=Path(args.outdir) if args.outdir else Path(Path(args.model).stem);out.mkdir(exist_ok=True)
    pp=args.pp or [2**i for i in range(14)];tg=args.tg or pp

    raw=(out/'raw_runs.jsonl').open('w');records=[]
    for b,bin in builds.items():
        for mode,vals in (('pp',pp),('tg',tg)):
            for v in vals:
                rec=run_bench(bin,args.model,args.flags,mode,v,not args.skip_gpu_mon,raw,args.interval)
                if rec:rec['build']=b;records.append(rec)
    raw.close()
    if not records:print('No data');return

    df=pd.DataFrame(records);df.to_json(out/'results.jsonl',orient='records',lines=True)
    for met,lbl in [('tokens_per_sec','tokens/s'),('vram_peak_mib','Peak VRAM (MiB)')]:
        for mode in ('pp','tg'):comb_plot(df,met,lbl,mode,out)
    tab=tabulate(df[['build','mode','value','tokens_per_sec','vram_peak_mib','vram_delta_mib','gtt_peak_mib','system_ram_peak_mib']],headers='keys',tablefmt='github');(out/'summary.md').write_text(tab)
    print('Done – artifacts in',out)

if __name__=='__main__':main()

