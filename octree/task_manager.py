#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""
Multi GPU parallel octree conversion pipeline for running hyper search.
Make a file tasks.json describing tasks e.g.
{
"data_root": "/home/sxyu/data",
"train_root": "/home/sxyu/proj/jaxnerf/jaxnerf/train/SH16",
"tasks": [{
        "octree_name": "oct_chair_bb1_2",
        "train_dir": "chair",
        "data_dir": "nerf_synthetic/chair",
        "config": "sh",
        "extr_flags": ["--bbox_from_data", "--bbox_scale", "1.2"],
        "opt_flags": [],
        "eval_flags": []
    },
    ...]
}

Then,
python dispatch.py tasks.json --gpus='space delimited list of gpus to use'

For each task, final octree is saved to
<data_root>/<data_dir>/octrees/<octree_name>/tree.npz
If you specify --keep_raw, the above is raw tree and the optimized tree is saved to
<data_root>/<data_dir>/octrees/<octree_name>/tree_opt.npz

Capacity, raw eval PSNR/SSIM/LPIPS, optimized eval PSNR/SSIM/LPIPS are saved to
<data_root>/<data_dir>/octrees/<octree_name>/results.txt
"""
import argparse
import sys
import os
import os.path as osp
import subprocess
import concurrent.futures
import json
from multiprocessing import Process, Queue

parser = argparse.ArgumentParser()
parser.add_argument("task_json", type=str)
parser.add_argument("--gpus", type=str, required=True,
                    help="space delimited GPU id list (pre CUDA_VISIBLE_DEVICES)")
parser.add_argument("--keep_raw", action='store_true',
        help="do not overwrite raw octree (takes extra disk space)")
args = parser.parse_args()

def convert_one(env, train_dir, data_dir, config, octree_name,
                extr_flags, opt_flags=[], eval_flags=[]):
    octree_store_dir = osp.join(train_dir, 'octrees', octree_name)
    octree_file = osp.join(octree_store_dir, "tree.npz")
    octree_opt_file = osp.join(octree_store_dir,
            "tree_opt.npz") if args.keep_raw else octree_file
    config_name = f"{config}"
    os.makedirs(octree_store_dir, exist_ok=True)
    extr_base_cmd = [
        "python", "-u", "-m", "octree.extraction",
        "--train_dir", train_dir,
        "--config", config_name, "--is_jaxnerf_ckpt",
        "--output ", octree_file,
        "--data_dir", data_dir
    ]
    opt_base_cmd = [
        "python", "-u", "-m", "octree.optimization",
        "--config", config_name, "--input", octree_file,
        "--output", octree_opt_file,
        "--data_dir", data_dir
    ]
    eval_base_cmd = [
        "python", "-u", "-m", "octree.evaluation",
        "--config", config_name, "--input ", octree_opt_file,
        "--data_dir", data_dir
    ]
    out_file_path = osp.join(octree_store_dir, 'results.txt')

    with open(out_file_path, 'w') as out_file:
        print('********************************************')
        print('! Extract', train_dir, octree_name)
        extr_cmd = ' '.join(extr_base_cmd + extr_flags)
        print(extr_cmd)
        extr_ret = subprocess.check_output(extr_cmd, shell=True, env=env).decode(
                sys.stdout.encoding)
        with open('pextract.txt', 'w') as f:
            f.write(extr_ret)

        extr_ret = extr_ret.split('\n')
        svox_str = extr_ret[-9]
        capacity = int(svox_str.split()[3].split(':')[1].split('/')[0])

        parse_metrics = lambda x: map(float, x.split()[2::2])
        psnr, ssim, lpips = parse_metrics(extr_ret[-2])
        print(': ', octree_name, 'RAW capacity',
              capacity, 'PSNR', psnr, 'SSIM', ssim, 'LPIPS', lpips)
        out_file.write(f'{capacity}\n{psnr:.10f} {ssim:.10f} {lpips:.10f}\n')

        print('! Optimize', train_dir, octree_name)
        opt_cmd = ' '.join(opt_base_cmd + opt_flags)
        print(opt_cmd)
        subprocess.call(opt_cmd, shell=True, env=env)

        if osp.exists(octree_opt_file):
            print('! Eval', train_dir, octree_name)
            eval_cmd = ' '.join(eval_base_cmd + eval_flags)
            print(eval_cmd)
            eval_ret = subprocess.check_output(eval_cmd, shell=True, env=env).decode(
                    sys.stdout.encoding)
            eval_ret = eval_ret.split('\n')

            epsnr, essim, elpips = parse_metrics(eval_ret[-2])
            print(':', octree_name, 'OPT capacity',
                  capacity, 'PSNR', epsnr, 'SSIM', essim, 'LPIPS', elpips)
            out_file.write(f'{epsnr:.10f} {essim:.10f} {elpips:.10f}\n')
        else:
            print('! Eval skipped')
            out_file.write(f'{psnr:.10f} {ssim:.10f} {lpips:.10f}\n')



def process_main(device, queue):
    # Set CUDA_VISIBLE_DEVICES programmatically
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    while True:
        task = queue.get()
        if len(task) == 0:
            break
        convert_one(env, **task)

if __name__=='__main__':
    with open(args.task_json, 'r') as f:
        tasks_file = json.load(f)
    all_tasks = tasks_file.get('tasks', [])
    data_root = tasks_file['data_root']
    train_root = tasks_file['train_root']
    pqueue = Queue()
    # Scene_tasks generated per scene (use {%} to mean scene name)
    if 'scene_tasks' in tasks_file:
        symb = '{%}'
        scenes = tasks_file['scenes']
        for scene_task in tasks_file['scene_tasks']:
            for scene in scenes:
                task = scene_task.copy()
                task['data_dir'] = scene_task['data_dir'].replace(symb, scene)
                task['train_dir'] = scene_task['train_dir'].replace(symb, scene)
                task['octree_name'] = scene_task['octree_name'].replace(symb, scene)
                all_tasks.append(task)

    print(len(all_tasks), 'total tasks')

    for task in all_tasks:
        task['train_dir'] = osp.join(train_root, task['train_dir'])
        task['data_dir'] = osp.join(data_root, task['data_dir'])
        octrees_dir = osp.join(task['data_dir'], 'octrees')
        os.makedirs(octrees_dir, exist_ok=True)
        # santity check
        assert os.path.exists(task['train_dir']), task['train_dir']
        assert os.path.exists(task['data_dir']), task['data_dir']

    for task in all_tasks:
        pqueue.put(task)
    pqueue.put({})

    args.gpus = list(map(int, args.gpus.split()))
    print('GPUS:', args.gpus)

    all_procs = []
    for i, gpu in enumerate(args.gpus):
        process = Process(target=process_main, args=(gpu, pqueue))
        process.daemon = True
        process.start()
        all_procs.append(process)

    for i, gpu in enumerate(args.gpus):
        all_procs[i].join()
