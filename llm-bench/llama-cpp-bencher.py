#!/usr/bin/env python3
"""llama_cpp_bencher.py – hot‑patch

* **Fix**: default `--build-root` → `/home/lhl/llama.cpp`.
* **Fix**: use `df['mode']` (column) instead of attribute; eliminates `KeyError: False`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess as sp
import sys
import textwrap
import time
import re
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List
import os
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# ----------------------------- SYSTEM INFO ------------------------------ #

def run_cmd(cmd: str) -> str:
    try:
        return sp.check_output(cmd, shell=True, text=True).strip()
    except sp.CalledProcessError as e:
        return e.output.strip()


def gather_system_info(executable: str) -> Dict[str, str]:
    info = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": platform.node(),
        "kernel": run_cmd("uname -r"),
        "os": platform.platform(),
        "cpu": run_cmd("lscpu | grep 'Model name' | awk -F: '{print $2}'"),
        "rocm_version": run_cmd(
            "rocminfo | grep -m1 'Runtime Version' | awk '{print $3}' || true"
        ),
        "hip_version": run_cmd("hipcc --version 2>/dev/null | head -n1 || true"),
        "vulkan_version": run_cmd(
            "vulkaninfo --summary 2>/dev/null | awk -F: '/Vulkan Instance Version/{print $2;exit}' || true"
        ),
        "amdgpu_driver": run_cmd("modinfo -F version amdgpu 2>/dev/null || true"),
    }

    try:
        exe_path = Path(executable).resolve()
        repo_root = exe_path.parents[2]
        commit = run_cmd(f"git -C {repo_root} rev-parse HEAD 2>/dev/null || true")
        if commit:
            info["llama_cpp_commit"] = commit
    except Exception:
        pass

    gpu_name = run_cmd("rocm-smi --showproductname --json || true")
    if gpu_name:
        info["gpu"] = gpu_name

    return info

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
    peaks[field]=peak
    peaks[field.replace('_peak_','_delta_')]=peak-init
    peaks[field.replace('_peak_','_start_')]=init

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

def run_bench(bin:str,model:str,flags:str,mode:str,val:int,gpu:bool,raw_sink,intv:float,env:Dict[str,str]|None=None,extra:Dict|None=None):
    if val==0:
        return {}
    bench_args=f"-p {val} -n 0" if mode=='pp' else f"-p 0 -n {val}"
    cmd=f"{shlex.quote(bin)} -m {shlex.quote(model)} {bench_args} {flags} -o jsonl"
    print("\n[RUN]",cmd)
    run_start=dt.datetime.now(dt.timezone.utc)
    stop=Event();peaks={}
    ths=[Thread(target=_monitor,args=(['cat','/proc/meminfo'],parse_meminfo,'system_ram_peak_mib',stop,peaks,intv),daemon=True)]
    ths[0].start()
    if gpu:
        for c,p,f in ([['rocm-smi','--showmeminfo','vram','--csv'],parse_rocm,'vram_peak_mib'],
                      [['amdgpu_top','-d'],parse_gtt,'gtt_peak_mib']):
            t=Thread(target=_monitor,args=(c,p,f,stop,peaks,intv),daemon=True);t.start();ths.append(t)

    last=None
    proc=sp.Popen(shlex.split(cmd),stdout=sp.PIPE,stderr=sp.STDOUT,text=True,env={**os.environ,**(env or {})})
    for line in proc.stdout:
        try:j=json.loads(line);last=j;raw_sink.write(line)
        except json.JSONDecodeError:sys.stdout.write(line)
    proc.wait();run_end=dt.datetime.now(dt.timezone.utc)
    stop.set();[t.join() for t in ths]
    duration=(run_end - run_start).total_seconds()
    tps=last.get('avg_ts') if last else None
    msg=f"[DONE] {mode} {val} → {duration:.1f}s"
    if tps is not None:
        msg+=f", {tps:.2f} tok/s"
    print(msg)
    return {
        'start_time': run_start.isoformat().replace('+00:00','Z'),
        'end_time': run_end.isoformat().replace('+00:00','Z'),
        'duration_s': (run_end - run_start).total_seconds(),
        'mode': mode,
        'value': val,
        'tokens_per_sec': tps,
        'bench_raw': last,
        **peaks,
        **(extra or {}),
    }

# ----------------------------- PLOTTING ---------------------------------- #

def comb_plot(df,metric,label,mode,out):
    fig,ax=plt.subplots(figsize=(8,5))
    subset=df[df['mode']==mode]
    for build,grp in subset.groupby('build'):
        ax.plot(grp['value'],grp[metric],marker='o',label=build)
    if ax.lines:
        ax.set_title(f"{label} – {mode.upper()} sweep");ax.set_xlabel('Tokens');ax.set_ylabel(label);ax.legend();ax.grid(True,alpha=.3,linestyle=':');fig.tight_layout();fig.savefig(out/f"{mode}_{metric}.png",dpi=150)
    plt.close(fig)

def sweep_table(df,mode):
    vals=sorted(df[df['mode']==mode]['value'].unique())
    headers=['backend','hipblaslt','-fa','-b']+[str(v) for v in vals]
    rows=[]
    for (b,fa,bf,hiplt),grp in df.groupby(['build','fa','b','hipblaslt']):
        row=[b,str(hiplt),fa,bf]
        for v in vals:
            sub=grp[(grp['mode']==mode)&(grp['value']==v)]
            row.append(sub['tokens_per_sec'].iloc[0] if not sub.empty else None)
        rows.append(row)

    # determine the best value for each concurrency column
    best={v:max((r[i+4] or 0 for r in rows),default=0) for i,v in enumerate(vals)}

    table=[]
    for row in rows:
        formatted=row[:4]
        for i,v in enumerate(vals):
            val=row[i+4]
            if val is None:
                formatted.append('-')
            else:
                formatted.append(f"**{val}**" if val==best[v] else val)
        table.append(formatted)

    return tabulate(table,headers=headers,tablefmt='github')

def write_summary(df,out):
    for met,lbl in [('tokens_per_sec','tokens/s'),('vram_peak_mib','Peak VRAM (MiB)')]:
        for mode in ('pp','tg'):
            comb_plot(df,met,lbl,mode,out)

    rows=[]
    for (b,fa,bf,hiplt),grp in df.groupby(['build','fa','b','hipblaslt']):
        def _get(mode,val):
            sub=grp[(grp['mode']==mode)&(grp['value']==val)]
            return sub['tokens_per_sec'].iloc[0] if not sub.empty else None
        pp512=_get('pp',512)
        tg128=_get('tg',128)
        mem=(grp['vram_peak_mib'] + grp['gtt_peak_mib']).max()
        rows.append({'backend':b,'hipblaslt':hiplt,'fa':fa,'b':bf,'pp512':pp512,'tg128':tg128,'mem':mem})

    best_pp=max((r['pp512'] or 0 for r in rows),default=0)
    best_tg=max((r['tg128'] or 0 for r in rows),default=0)
    best_mem=min((r['mem'] for r in rows),default=0)

    table=[]
    for r in rows:
        pp = f"**{r['pp512']}**" if r['pp512']==best_pp and r['pp512'] is not None else (r['pp512'] if r['pp512'] is not None else '-')
        tg = f"**{r['tg128']}**" if r['tg128']==best_tg and r['tg128'] is not None else (r['tg128'] if r['tg128'] is not None else '-')
        mem = f"**{r['mem']}**" if r['mem']==best_mem else r['mem']
        table.append([r['backend'],r.get('hipblaslt',''),r['fa'],r['b'],pp,tg,mem])

    headers=['backend','hipblaslt','-fa','-b','pp512','tg128','max_mem']
    top_tab=tabulate(table,headers=headers,tablefmt='github')
    pp_tab=sweep_table(df,'pp')
    tg_tab=sweep_table(df,'tg')
    md='\n\n'.join([top_tab,'\n### PP sweep\n',pp_tab,'\n### TG sweep\n',tg_tab])
    (out/'summary.md').write_text(md)

# ------------------------------ MAIN ------------------------------------- #

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,description='llama.cpp benchmark helper')
    ap.add_argument('-m','--model',required=True);ap.add_argument('-o','--outdir')
    ap.add_argument('-b','--builds',nargs='*');ap.add_argument('--build-root',type=Path,default=Path('/home/lhl/llama.cpp'))
    ap.add_argument('-p','--pp',nargs='*',type=int);ap.add_argument('-n','--tg',nargs='*',type=int)
    ap.add_argument('--flags',default='')
    ap.add_argument('--moe',action='store_true')
    ap.add_argument('--skip-gpu-mon',action='store_true');ap.add_argument('--list-builds',action='store_true');ap.add_argument('--interval',type=float,default=.2)
    ap.add_argument('--resummarize',action='store_true',help='recompute summary from results.jsonl')
    ap.add_argument('--rerun',action='store_true',help='force sweep rerun even if results exist')
    args=ap.parse_args()

    found=discover_builds(args.build_root)
    if args.list_builds:
        print('\n'.join(f"{k}: {v}" for k,v in found.items()));return
    sel=args.builds or list(found.keys());builds={n:found.get(n,n) for n in sel}
    out=Path(args.outdir) if args.outdir else Path(Path(args.model).stem);out.mkdir(exist_ok=True)

    # capture system information once per run
    any_bin=next(iter(builds.values())) if builds else None
    if any_bin:
        sys_info=gather_system_info(any_bin)
        (out/'system_info.json').write_text(json.dumps(sys_info,indent=2))

    run_start=dt.datetime.now(dt.timezone.utc)

    pp=args.pp or [2**i for i in range(14)];tg=args.tg or pp

    if args.resummarize:
        res=out/'results.jsonl'
        if not res.exists():
            print('No results.jsonl in',out)
            return
        df=pd.read_json(res,orient='records',lines=True)
        write_summary(df,out)
        print('Summary updated from existing results')
        return

    raw=(out/'raw_runs.jsonl').open('a');records=[]
    res_file=out/'results.jsonl'
    existing_df=pd.read_json(res_file,orient='records',lines=True) if res_file.exists() else pd.DataFrame()
    base_flags=args.flags.strip()
    for b,bin in builds.items():
        env_opts=[{}]
        if b in ('hip','rocwmma'):
            env_opts.append({'ROCBLAS_USE_HIPBLASLT':'1'})
        b_opts=['']
        if b=='vulkan' and args.moe:
            b_opts.append('-b 256')
        for env in env_opts:
            for fa in ('','-fa 1'):
                for bf in b_opts:
                    flags=" ".join(f for f in (base_flags,fa,bf) if f).strip()
                    info={'build':b,'fa':fa.strip(), 'b':bf.strip(), 'hipblaslt':env.get('ROCBLAS_USE_HIPBLASLT','')}
                    for mode,vals in (('pp',pp),('tg',tg)):
                        for v in vals:
                            if not args.rerun and not existing_df.empty:
                                mask=(existing_df['build']==info['build'])&(existing_df['fa']==info['fa'])&(existing_df['b']==info['b'])&(existing_df['hipblaslt']==info['hipblaslt'])&(existing_df['mode']==mode)&(existing_df['value']==v)
                                if mask.any():
                                    print(f"[SKIP] {mode} {v} for build {b} (use --rerun to force)")
                                    continue
                            rec=run_bench(bin,args.model,flags,mode,v,not args.skip_gpu_mon,raw,args.interval,env,info)
                            if rec:records.append(rec)
    raw.close()
    if not records and existing_df.empty:
        print('No data');return

    new_df=pd.DataFrame(records)
    df=pd.concat([existing_df,new_df],ignore_index=True)
    df.to_json(out/'results.jsonl',orient='records',lines=True)
    if not new_df.empty:
        write_summary(df,out)

    run_end=dt.datetime.now(dt.timezone.utc)
    run_info={
        'start_time': run_start.isoformat().replace('+00:00','Z'),
        'end_time': run_end.isoformat().replace('+00:00','Z'),
        'duration_s': (run_end - run_start).total_seconds(),
    }
    (out/'run_info.json').write_text(json.dumps(run_info,indent=2))

    print('Done – artifacts in',out)

if __name__=='__main__':main()

