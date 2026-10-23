#!/usr/bin/env python3
"""Parallel read-only solver workers; one validated atomic writer per output."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import random
import subprocess
import time
from pathlib import Path

from improve import ROOT,save,transform,untransform
from validate import read_cases,read_programs,simulate,validate


def worker(case,maximum,config,budget,binary="lookahead"):
    radius,pattern,required=case
    width,variant,seed,symmetry,cap,look=config
    pattern,permutation,reflection=transform(radius,pattern,symmetry)
    if binary=="optimize":
        header=f"{radius} {required} {width} {maximum-1} {seed} {budget} {variant}\n"
    else:
        header=f"{radius} {required} {width} {maximum-1} {variant} {cap} {look} {seed} {budget}\n"
    try:
        result=subprocess.run([str(ROOT/binary)],input=header+"\n".join(pattern)+"\n",text=True,capture_output=True,timeout=budget+5)
        if result.returncode!=0:return None
        program=[line.split() for line in result.stdout.splitlines()]
        if not program or any(len(line)!=6 or any(op not in {"F","B","T","-"} for op in line) for line in program):return None
        return untransform(program,permutation,reflection)
    except subprocess.TimeoutExpired:return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds",type=float,default=180)
    parser.add_argument("--case-seconds",type=float,default=2)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--width",type=int,default=32)
    parser.add_argument("--variant",type=int,default=0)
    parser.add_argument("--cap",type=int,default=8)
    parser.add_argument("--look",type=int,default=4)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--passes",type=int,default=1)
    parser.add_argument("--file")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--binary",default="lookahead")
    parser.add_argument("--random-variants",action="store_true")
    parser.add_argument("--snapshots",action="store_true",help="Publish checkpoints after each 100-line improvement")
    parser.add_argument("--output-dir",type=Path,help="Separate candidate outputs for safe independent campaigns")
    parser.add_argument("--order",choices=["length","random","regret"],default="length")
    parser.add_argument("--min-length",type=int,default=0)
    args=parser.parse_args();rng=random.Random(args.seed);deadline=time.monotonic()+args.seconds
    datasets=[];jobs=[]
    for source in sorted(ROOT.glob(args.file or "in_*.txt")):
        if "example" in source.name:continue
        output=source.with_name(source.name.replace("in_","out_",1));validate(source,output)
        cases=read_cases(source);programs=read_programs(output,len(cases));datasets.append((source,output,cases,programs))
        if args.output_dir:
            args.output_dir.mkdir(exist_ok=True)
            output=args.output_dir/output.name
            if output.exists():
                existing=read_programs(output,len(cases))
                for i in range(len(cases)):
                    r,p,c=cases[i]
                    if len(existing[i])<len(programs[i]) and simulate(r,existing[i],p)>=c:programs[i]=existing[i]
            save(output,programs)
            datasets[-1]=(source,output,cases,programs)
        for i,case in enumerate(cases):
            if len(programs[i])<args.min_length:continue
            for trial in range(args.passes):
                priority=len(programs[i])
                if args.order=="random":priority=1
                if args.order=="regret":
                    r,pat,c=case
                    size=sum(row.count("M") for row in pat)
                    priority=len(programs[i])/(size*c)**.5
                jobs.append((priority*rng.uniform(.6,1.4),len(datasets)-1,i,trial))
    jobs.sort(reverse=True)
    if args.limit:jobs=jobs[:args.limit]
    initial=sum(sum(map(len,data[3])) for data in datasets)
    last_published=initial
    iterator=iter(jobs);pending={};completed=improved=0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        def submit():
            if time.monotonic()>=deadline:return False
            try:_,d,i,trial=next(iterator)
            except StopIteration:return False
            source,output,cases,programs=datasets[d]
            variant=rng.randrange(4) if args.random_variants else args.variant
            seed=rng.randrange(40000) if args.binary=="optimize" else rng.getrandbits(32)
            config=(args.width,variant,seed,rng.randrange(12),args.cap,args.look)
            future=executor.submit(worker,cases[i],len(programs[i]),config,min(args.case_seconds,max(.01,deadline-time.monotonic())),args.binary)
            pending[future]=(d,i)
            return True
        for _ in range(args.workers):submit()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                d,i=pending.pop(future);completed+=1
                source,output,cases,programs=datasets[d];program=future.result()
                if program is not None and len(program)<len(programs[i]):
                    r,pattern,required=cases[i]
                    if simulate(r,program,pattern)>=required:
                        before=len(programs[i]);programs[i]=program;save(output,programs);improved+=1
                        total=sum(sum(map(len,data[3])) for data in datasets)
                        print(f"{source.name} case {i+1}: {before} -> {len(program)}; total {total}",flush=True)
                        if args.snapshots and len(datasets)==4 and last_published-total>=100:
                            subprocess.run(["python3",str(ROOT/"publish.py")],check=True)
                            last_published=total
                submit()
    for source,output,cases,programs in datasets:
        report=validate(source,output);output.with_suffix(".optimization.json").write_text(json.dumps(report,indent=2)+"\n")
        print(f"{output.name}: {report['instruction_lines']}",flush=True)
    total=sum(sum(map(len,data[3])) for data in datasets)
    print(f"Campaign: {initial} -> {total}; {completed} attempts, {improved} improvements",flush=True)


if __name__=="__main__":main()
