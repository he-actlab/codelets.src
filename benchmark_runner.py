import sys, os, argparse, random, pathlib, subprocess, shutil, json
import numpy as np

CFG_BASE = pathlib.Path(f"{pathlib.Path(__file__).parent}/codelets/examples/genesys/configs")

MODELS =["resnet18","resnet50","gpt2-trimmed-opt",
         "efficientnet-lite4-opt-no-softmax","mobilenetv2-opt",
         "yolov3-opt-static","bert-base-cased-transpose-opt-trimmed-ort","vgg16"]

FACTORS = {}
# Factors = oc, oh, ow
FACTORS[216] = [
    (216, 1, 1),
    (108, 2, 1),
    (54, 4, 1),
    (54, 2, 2),
    (27, 4, 2),
    (27, 8, 1),
    (9, 24, 1),
    (9, 12, 2),
    (9, 8, 3),
    (3, 72, 1),
    (3, 36, 2),
    (3, 12, 6),
    (3, 9, 8),
    (2, 108, 1),
    (2, 54, 2),
    (2, 27, 4),

    (1, 216, 1),
    (1, 108, 2),
    (1, 54, 4),
    (1, 27, 8),
]

def get_factors(n):
    factors = {1}
    max_p  = int(n**0.5)
    p,inc = 2,1
    while p <= max_p:
        while n%p==0:
            factors.update([f*p for f in factors])
            n //= p
            max_p = int(n**0.5)
        p,inc = p+inc,2
    if n>1:
        factors.update([f*n for f in factors])
    return sorted(factors)

def copy_file(src, dst, verbose=False):
    if verbose: print(f"Copying {src} --> {dst}")
    shutil.copy(src, dst)

def try_subprocess_exec(cmd, verbose=False, fail=False):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err_msg,out_msg, err_code = r.stderr, r.stdout, r.returncode
    if err_code != 0 and fail:
        cmd_str = " ".join(cmd)
        raise RuntimeError(f"Failed to executed command {cmd_str}:\n{err_msg}\n{out_msg}")
    elif verbose and fail:
        print(f"{err_msg}\n{out_msg}")
    return err_msg, out_msg, err_code

def generate_cfg(base_cfg_name, oc_split, ow_split, oh_split, fused: bool):
    base_cfg_path = pathlib.Path(f"{CFG_BASE}/{base_cfg_name}")
    with open(base_cfg_path, "r") as f:
        base_cfg = json.load(f)

    assert "GPU_SCALING" in base_cfg
    assert base_cfg["GPU_SCALING"]['TOTAL'] == oc_split*ow_split*oh_split
    base_cfg["GPU_SCALING"]['OC'] = oc_split
    base_cfg["GPU_SCALING"]['OH'] = oh_split
    base_cfg["GPU_SCALING"]['OW'] = ow_split
    base_cfg["FUSE_LAYERS"] = fused

    new_cfg_name = base_cfg_name.split(".")[0] + f"_oc{oc_split}_oh{oh_split}_ow{ow_split}"
    if fused:
        new_cfg_name += f"_fused"

    new_cfg_path = pathlib.Path(f"{CFG_BASE}/{new_cfg_name}.json")

    with open(new_cfg_path, "w") as f:
        json.dump(base_cfg, f, indent=4)

    return new_cfg_name

def compile_benchmark(cfg, model, ext):
    cmd = ["python", "tools/benchmark_compilation.py",
           "--model", model,
           "--config", cfg,
           "--extension", ext,
           "--verbose"
           ]
    return try_subprocess_exec(cmd, verbose=True)

def get_scale_factors(scale_factor):
    root_factors = get_factors(scale_factor)
    all_factors = []
    for rf in root_factors:
        oc = int(scale_factor/rf)
        fac_fac = get_factors(rf)
        oh_factors = []
        for oh in fac_fac:
            ow = int(rf/oh)
            if ow not in oh_factors:
                all_factors.append((oc, oh, ow))
                oh_factors.append(oh)
            else:
                break
    assert all([np.prod(v) == scale_factor for v in all_factors])
    return all_factors


def append_log_msg(fname, msg):
    print(f"{msg}")
    with open(fname, "a") as f:
        f.write(f"{msg}\n")

def run_benchmarks_for_cfg(scale_factor):
    fname=f"compilation_results_sf{scale_factor}_v2.txt"
    with open(fname, "w") as f:
        f.write(f"Compilation results for {scale_factor}\n")
    fails = []
    failure_outputs = []
    success = []
    cfg_name = f"gpu_scaling_{scale_factor}.json"
    scale_factors = get_scale_factors(scale_factor)
    for sf in scale_factors:
        append_log_msg(fname, f"Running compilation for oc_split={sf[0]}, oh_split={sf[1]}, ow_split={sf[2]}, fusion=True")
        sf_cfg_name = generate_cfg(cfg_name, sf[0], sf[1], sf[2], True)
        sf_cfg_name += ".json"
        ext = f"_oc{sf[0]}_oh{sf[1]}_ow{sf[2]}_fused"
        for m in MODELS:
            bench_str = f"model={m}, oc_split={sf[0]}, oh_split={sf[1]}, ow_split={sf[2]}, fused=True"
            append_log_msg(fname, f"Compiling {bench_str}...")
            err_msg, out_msg, err_code = compile_benchmark(sf_cfg_name, m, ext)
            if err_code != 0:
                fails.append(bench_str)
                err_msg = f"{bench_str} failure output:\n{err_msg}\n{out_msg}"
                failure_outputs.append(err_msg)
                append_log_msg(fname, f"Failed compilation for {bench_str}\n{err_msg}")
            else:
                success.append(bench_str)
                append_log_msg(fname, f"Successful compilation for {bench_str}")

        append_log_msg(fname, f"Running compilation for oc_split={sf[0]}, oh_split={sf[1]}, ow_split={sf[2]}, fusion=False")
        sf_cfg_name = generate_cfg(cfg_name, sf[0], sf[1], sf[2], False)
        sf_cfg_name += ".json"
        ext = f"_oc{sf[0]}_oh{sf[1]}_ow{sf[2]}_unfused"
        for m in MODELS:
            bench_str = f"model={m}, oc_split={sf[0]}, oh_split={sf[1]}, ow_split={sf[2]}, fused=False"
            append_log_msg(fname, f"Compiling {bench_str}...")
            err_msg, out_msg, err_code = compile_benchmark(sf_cfg_name, m, ext)
            if err_code != 0:
                fails.append(bench_str)
                err_msg = f"{bench_str} failure output:\n{err_msg}\n{out_msg}"
                failure_outputs.append(err_msg)
                append_log_msg(fname, f"Failed compilation for {bench_str}\n{err_msg}")
            else:
                success.append(bench_str)
                append_log_msg(fname, f"Successful compilation for {bench_str}")


    output_str = f"Successes" + "-"*40 + "\n"
    output_str += "\n".join(success)
    output_str += f"Failures" + "-"*40 + "\n"
    output_str += "\n".join(fails)
    output_str += f"Failure outputs" + "-"*40 + "\n"
    output_str += "\n".join(failure_outputs)

    with open(f"compilation_results_sf{scale_factor}_finalv2.txt", "w") as f:
        f.write(output_str)

# Fixes
# fused clip-dw has incorrect number of loops
# fused dw conv includes an additional loop with incorrect number of iterations
# Fix request size iterations number

# TPU


if __name__ == "__main__":
    model="resnet18"
    # model="resnet50"
    # model="gpt2-trimmed-opt"
    # model="efficientnet-lite4-opt-no-softmax"
    # model="mobilenetv2-opt"
    # model="yolov3-opt-static"
    # model="bert-base-cased-transpose-opt-trimmed-ort"
    # model="vgg16"
    # factors = get_scale_factors(216)
    # cfg = "gpu_scaling_test.json"
    # cmd = ["python", "tools/benchmark_compilation.py",
    #        "--model", model,
    #        "--config", cfg
    #        ]
    #
    # verbose = True
    # #
    # if verbose:
    #     cmd.append("--verbose")
    # try_subprocess_exec(cmd, verbose=verbose)
    run_benchmarks_for_cfg(216)

