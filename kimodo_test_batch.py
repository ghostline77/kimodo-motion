"""Generate independent test motions with one bounded text/motion model load."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def child(plan):
    """Use the existing offload bootstrap and unmodified upstream generator."""
    from kimodo.scripts import generate
    import runtime_bootstrap
    loading = importlib.import_module("kimodo.model.load_model")
    original_inputs, original_main = generate.get_generation_inputs, generate.main
    original_load = loading.load_model
    loaded = None

    def load_once(*args, **kwargs):
        nonlocal loaded
        if loaded is None:
            loaded = original_load(*args, **kwargs)
        return loaded

    def all_prompts(args, fps):
        values = original_inputs(args, fps)
        values["texts"] = [job["prompt"] for job in plan["jobs"]]
        return values

    def independent_jobs():
        generate.get_generation_inputs = original_inputs
        for index, job in enumerate(plan["jobs"]):
            output = Path(job["output"])
            sys.argv = [__file__, *job["arguments"]]
            progress = {"stage": "generating", "index": index, "job": job["id"],
                        "completed": [item["id"] for item in plan["jobs"][:index]]}
            Path(plan["progress"]).write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(json.dumps(progress), flush=True)
            started = time.time()
            with (output / "generation.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    original_main()
            (output / "timing.json").write_text(json.dumps({"generation_seconds": time.time()-started}), encoding="utf-8")
        Path(plan["progress"]).write_text(json.dumps({"stage":"generated", "completed":[j["id"] for j in plan["jobs"]]}, indent=2), encoding="utf-8")

    loading.load_model = load_once
    generate.get_generation_inputs = all_prompts
    generate.main = independent_jobs
    sys.argv = [__file__, *plan["jobs"][0]["arguments"]]
    runtime_bootstrap.main()


def parent(args):
    from kimodo_motion import inspect_motion
    jobs = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(jobs,list) or not 1 <= len(jobs) <= 16:
        raise ValueError("Expected 1..16 independent jobs")
    ids = set()
    for job in jobs:
        if not job["id"].replace("-", "").isalnum() or job["id"] in ids:
            raise ValueError("Invalid or duplicate job ID")
        ids.add(job["id"])
        prompt = job["prompt"]
        if not prompt.strip() or "." in prompt.rstrip(".") or not 2 <= job["seconds"] <= 10:
            raise ValueError("Each independent job needs one sentence and 2..10 seconds")
    upstream = args.upstream.resolve(strict=True)
    revision = subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(upstream),"status","--porcelain"],text=True).strip():
        raise ValueError("Upstream checkout must be unmodified")
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    source = Path(__file__).parent
    for name in [Path(__file__).name,"runtime_bootstrap.py","kimodo_motion.py"]:
        shutil.copy2(source/name,root/name)
    checkpoint = args.checkpoint_root.resolve(strict=True)/"Kimodo-SOMA-RP-v1.1"
    hashes = {name:digest(checkpoint/name) for name in ["config.yaml","model.safetensors"]}
    for job in jobs:
        output = root/job["id"]
        output.mkdir()
        job["output"] = str(output)
        job["arguments"] = [job["prompt"],"--model","Kimodo-SOMA-RP-v1.1","--duration",str(job["seconds"]),
                            "--num_samples","1","--seed",str(job["seed"]),"--diffusion_steps","50",
                            "--bvh","--bvh_standard_tpose","--output",str(output/"motion")]
    plan = {"jobs":jobs,"progress":str(root/"progress.json")}
    (root/"jobs.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    command = [sys.executable,str(root/Path(__file__).name),"--child","--plan",str(root/"jobs.json")]
    env = dict(os.environ,PYTHONPATH=os.pathsep.join([str(upstream),str(source/"work/native-deps")]),
               CHECKPOINT_DIR=str(checkpoint.parent),TEXT_ENCODERS_DIR=str(checkpoint.parent.parent),
               TEXT_ENCODER_MODE="local",TEXT_ENCODER_DEVICE="cpu",GHOSTLINE_TEXT_MODE="sequential",
               GHOSTLINE_BLAS_THREADS="1",GHOSTLINE_TEXT_REPORT=str(root/"text-memory.json"),
               HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    common = {"model":"Kimodo-SOMA-RP-v1.1","steps":50,"engine":"upstream-pytorch-batch",
              "upstream_revision":revision,"checkpoint_hashes":hashes,"command":command,
              "text_mode":"sequential","blas_threads":1,"working_directory":str(checkpoint.parent.parent),
              "runner_sha256":digest(root/Path(__file__).name),"runtime_bootstrap_sha256":digest(root/"runtime_bootstrap.py"),
              "packages":{name:importlib.metadata.version(name) for name in ["torch","transformers","peft","accelerate","numpy","safetensors"]}}
    (root/"request.json").write_text(json.dumps(common,indent=2),encoding="utf-8")
    for job in jobs:
        output=Path(job["output"])
        request={**common,"prompt":job["prompt"],"seed":job["seed"],"frames":round(job["seconds"]*30),
                 "upstream_arguments":job["arguments"],"batch_evidence":str(root)}
        (output/"request.json").write_text(json.dumps(request,indent=2),encoding="utf-8")
        shutil.copy2(root/Path(__file__).name,output/"runner.py")
        shutil.copy2(root/"runtime_bootstrap.py",output/"runtime_bootstrap.py")
    with (root/"batch.log").open("w",encoding="utf-8") as log:
        result=subprocess.run(command,cwd=checkpoint.parent.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
    outcomes=[]
    for job in jobs:
        output=Path(job["output"])
        try:
            inspected=inspect_motion(output/"motion.npz",round(job["seconds"]*30))
            for name in ["motion.bvh","generation.log"]:
                if not (output/name).is_file(): raise ValueError(f"Missing {name}")
            request=json.loads((output/"request.json").read_text(encoding="utf-8"))
            package={"format":"ghostline-generated-motion","version":1,**request,**inspected,
                     "review_status":"unreviewed","retargeted":False,"runtime_verified":False,
                     "files":{name:digest(output/name) for name in ["motion.npz","motion.bvh","generation.log"]}}
            (output/"motion.json").write_text(json.dumps(package,indent=2),encoding="utf-8")
            outcomes.append({"id":job["id"],"state":"complete","manifest":str(output/"motion.json")})
        except Exception as error:
            outcomes.append({"id":job["id"],"state":"failed","error":str(error)})
    receipt={"exit_code":result.returncode,"outcomes":outcomes}
    (root/"completion.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt,indent=2))
    if result.returncode or any(o["state"]!="complete" for o in outcomes):
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--upstream",type=Path)
    parser.add_argument("--checkpoint-root",type=Path)
    parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    if args.child:
        child(json.loads(args.plan.read_text(encoding="utf-8")))
    else:
        if not all([args.output,args.upstream,args.checkpoint_root]): parser.error("Parent needs output, upstream and checkpoint-root")
        parent(args)


if __name__=="__main__":
    main()
