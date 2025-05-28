#!/usr/bin/env python3
"""llama_cpp_bencher.py

End‑to‑end benchmark runner for multiple **llama.cpp** variants.

New in this revision (JSONL ingestion)
======================================
* Every `llama-bench` call is now forced to `-o jsonl` and the _full_ JSON
  record returned by the binary is streamed to **raw_runs.jsonl**.
* `tokens_per_sec` is taken from `avg_ts` in that JSON object ­– no more fragile
  table parsing.
* The last JSON object of each run is embedded in the high‑level results as
  `bench_raw` for easy programmatic re‑use.
"""
from __future__ import annotations

import argparse, datetime as dt, json, shlex, subprocess as sp, sys, textwrap, time
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List
import re

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

###############################################################################
# ---------------------------- BUILD DISCOVERY ------------------------------ #
###############################################################################

def discover_builds(root: Path) -> Dict[str,str]:
    return {p.parts[-4]: str(p) for p in root.glob('llama.cpp-*/build/bin/llama-bench')}

###############################################################################
# ----------------------------- MEM MONITORS -------------------------------- #
###############################################################################

def _first_int(txt:str)->int:
    m=re.search(r"(\d+)",txt);return int(m.group(1)) if m else 0

def _monitor(cmd:List[str],parse,field:str,stop:Event,peaks:Dict[str,int]):
    peak=0
    while not stop.is_set():
        try:
            used=parse(sp.check_output(cmd,text=True))
            peak=max(peak,used)
        except Exception:
            pass
        time.sleep(1)
    if peak:peaks[field]=peak

def parse_rocm(out:str)->int:
    try:return int(out.splitlines()[1].split(',')[2])//(1024*1024)
    except:return 0

def parse_gtt(out:str)->int:
    line=next((l for l in out.splitlines() if re.match(r"^\s*GTT",l)),"");return _first_int(line)

def parse_meminfo(out:str)->int:
    t=a=0
    for l in out.splitlines():
        if l.startswith('MemTotal'):t=_first_int(l)
        elif l.startswith('MemAvailable'):a=_first_int(l)
    return (t-a)//1024

###############################################################################
# ------------------------------- BENCH RUN --------------------------------- #
###############################################################################

def run_bench(bin:str,model:str,flags:str,mode:str,val:int,gpu:bool,raw_sink)->Dict:
    if val==0:return{}
    bench_args=f"-p {val} -n 0" if mode=='pp' else f"-p 0 -n {val}"
    cmd=f"{shlex.quote(bin)} -m {shlex.quote(model)} {bench_args} {flags} -o jsonl"
    print("\n[RUN]",cmd)
    stop=Event();peaks={}
    sysmon=Thread(target=_monitor,args=(['cat','/proc/meminfo'],parse_meminfo,'system_ram_peak_mib',stop,peaks),daemon=True);sysmon.start()
    mons=[sysmon]
    if gpu:
        mons+=[Thread(target=_monitor,args=(['rocm-smi','--showmeminfo','vram','--csv'],parse_rocm,'vram_peak_mib',stop,peaks),daemon=True),
               Thread(target=_monitor,args=(['amdgpu_top','-d'],parse_gtt,'gtt_peak_mib',stop,peaks),daemon=True)]
        [m.start() for m in mons[1:]]

    last_json=None
    start=time.time()
    proc=sp.Popen(shlex.split(cmd),stdout=sp.PIPE,stderr=sp.STDOUT,text=True,bufsize=1)
    try:
        for line in proc.stdout:
            try:
                rec=json.loads(line)
                last_json=rec
                raw_sink.write(line)
            except json.JSONDecodeError:
                sys.stdout.write(line)
    finally:
        proc.wait();stop.set();[m.join() for m in mons]

    tok_s= last_json.get('avg_ts') if last_json else None
    return {'timestamp':dt.datetime.now().astimezone().isoformat(),'mode':mode,'value':val,'tokens_per_sec':tok_s,'bench_raw':last_json,**peaks}

###############################################################################
# -------------------------------- PLOT ------------------------------------- #
###############################################################################

def plot(df,build,mode,out):
    d=df[(df.build==build)&(df.mode==mode)].sort_values('value')
    if d.empty:return
    fig,ax1=plt.subplots(figsize=(8,5));ax1.set_title(f"{build} – {mode.upper()} sweep");ax1.set_xlabel('Tokens');ax1.set_ylabel('tokens/s');ax1.plot(d.value,d.tokens_per_sec,marker='o');ax1.grid(True,alpha=.3,linestyle=':')
    if 'vram_peak_mib'in d.columns and d.vram_peak_mib.notna().any():
        ax2=ax1.twinx();ax2.set_ylabel('VRAM MiB');ax2.plot(d.value,d.vram_peak_mib,marker='x',linestyle='--')
    fig.tight_layout();fig.savefig(out/f"{build}_{mode}.png",dpi=150);plt.close(fig)

###############################################################################
# -------------------------------- MAIN ------------------------------------- #
###############################################################################

def main():
    parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,description='llama.cpp benchmark helper')
    parser.add_argument('-m','--model',required=True);parser.add_argument('-o','--outdir');parser.add_argument('-b','--builds',nargs='*');parser.add_argument('--build-root',default=Path('/home/lhl/llama.cpp'),type=Path);parser.add_argument('-p','--pp',nargs='*',type=int);parser.add_argument('-n','--tg',nargs='*',type=int);parser.add_argument('--flags',default='-fa 1');parser.add_argument('--skip-gpu-mon',action='store_true');parser.add_argument('--list-builds',action='store_true')
    args=parser.parse_args()

    found=discover_builds(args.build_root)
    if args.list_builds:
        print('\n'.join(f"{k}: {v}" for k,v in found.items()));return
    sel=args.builds or list(found.keys())
    builds={name:(found.get(name,name)) for name in sel}

    out=Path(args.outdir) if args.outdir else Path(Path(args.model).stem);out.mkdir(parents=True,exist_ok=True)
    pp=args.pp or [2**i for i in range(14)];tg=args.tg or pp

    raw_file=(out/'raw_runs.jsonl').open('w')
    records=[]
    for b,bin in builds.items():
        for mode,vals in(('pp',pp),('tg',tg)):
            for v in vals:
                rec=run_bench(bin,args.model,args.flags,mode,v,not args.skip_gpu_mon,raw_file)
                if rec:
                    rec['build']=b;records.append(rec)
    raw_file.close()

    if not records:
        print('No data');return
    df=pd.DataFrame(records);df.to_json(out/'results.jsonl',orient='records',lines=True)
    for build in df.build.unique():
        for mode in('pp','tg'):plot(df,build,mode,out)
    tab=tabulate(df[['build','mode','value','tokens_per_sec','vram_peak_mib','gtt_peak_mib','system_ram_peak_mib']],headers='keys',tablefmt='github');(out/'summary.md').write_text(tab)
    print('Done: artifacts in',out)

if __name__=='__main__':
    main()
