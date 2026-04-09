import os
import subprocess
import multiprocessing as mp
import argparse
import re
import numpy as np
from pathlib import Path
import shutil
import math
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Runner for Vimeo Test")
    parser.add_argument('-opt', type=str, required=True, help='Path to option file')
    parser.add_argument('-staa_opt', type=str, required=True, help='Path to staa option file')
    parser.add_argument('-path', type=str, required=True, help='Data path')
    parser.add_argument('-mode', type=str, required=True, help='Running mode')
    parser.add_argument('-model', type=str, required=True, help='Model name')
    parser.add_argument('-qp', type=int, required=True, help='QP value')
    parser.add_argument('--codec_type', type=str, required=True, help='Codec type')
    parser.add_argument('--checkpoints', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None, choices=['vimeo90k', 'ucf101', 'snufilm'])
    parser.add_argument('--script_name', type=str, default="benchmark_vimeo_septuplet_compress.py", help='Lower level script name')
    parser.add_argument('--total_slices', type=int, default=8, help='Total number of slices')
    parser.add_argument('--gpu_list', type=str, default="0,1,2,3,4,5,6,7", help='Comma separated GPU IDs')
    parser.add_argument('--output_log_dir', type=str, default="/output/parallel_logs", help='Log directory')
    parser.add_argument('--dataset_type', type=str, default='septuplet', choices=['septuplet', '65frames'], help='use small model')
    return parser.parse_args()

def run_task(args, slice_id, gpu_id):
    log_file = os.path.join(args.output_log_dir, f"log_slice_{slice_id}_gpu_{gpu_id}.txt")
    cmd = [
        "python", 
        os.path.join("/code/codes", args.script_name),
        "-opt", args.opt,
        "-staa_opt", args.staa_opt,
        "-path", args.path,
        "-mode", args.mode,
        "-model", args.model,
        "-qp", str(args.qp),
        "--codec_type", args.codec_type,
        "--slice_id", str(slice_id),
        "--total_slices", str(args.total_slices),
        "--gpu_id", str(gpu_id),
        "--dataset_type", str(args.dataset_type)
    ]
    if args.checkpoints is not None:
        cmd.append("--checkpoints")
        cmd.append(args.checkpoints)
    print(f"[Scheduler] Starting Slice {slice_id} on GPU {gpu_id} ... {log_file}")
    os.makedirs(args.output_log_dir, exist_ok=True)
    print(' '.join(cmd))
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        process.wait()
    if process.returncode == 0:
        return log_file
    else:
        print(f"[Scheduler] Slice {slice_id} (GPU {gpu_id}) FAILED.")
        return None

def parse_results(log_files, qp_value, model_name):
    metrics_config = [
        ("psnr", r"psnr:([\d.-]+)", True),
        ("psnr_avg_mse", r"psnr_avg_mse:([\d.-]+)", False),
        # ("mse", r"mse:([\d.-]+)", False),
        ("ssim", r"ssim:([\d.-]+)", True),
        ("psnr_LFR_lq", r"psnr_LFR_lq:([\d.-]+)", False),
        ("ssim_LFR_lq", r"ssim_LFR_lq:([\d.-]+)", False),
        ("psnr_LFR_hq", r"psnr_LFR_hq:([\d.-]+)", False),
        ("sigma", r"sigma:([\d.-]+)", False),
        ("psnr inter", r"psnr inter:([\d.-]+)", False),
        ("ssim inter", r"ssim inter:([\d.-]+)", False),
        ("lpips", r"lpips:([\d.-]+)", False),
        ("dists", r"dists:([\d.-]+)", False),
        ("tlpips", r"tlpips\(1e3\):([\d.-]+)", False),
        ("tof", r"tof\(1e1\):([\d.-]+)", False),
        ("warpping_psnr", r"warpping_psnr:([\d.-]+)", False),
        ("ave_img_bpp", r"ave_img_bpp:([\d.-]+)", False),
        ("vmaf", r"vmaf:([\d.-]+)", True),
    ]
    aggregated_data = {key: [] for key, _, _ in metrics_config}
    total_samples_processed = 0
    for log_file in log_files:
        if not log_file or not os.path.exists(log_file):
            continue
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                file_metrics = {key: [] for key, _, _ in metrics_config}
                for key, pattern, is_primary in metrics_config:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    clean_matches = []
                    for m in matches:
                        if m.lower() == 'nan':
                            clean_matches.append(np.nan)
                        else:
                            try:
                                clean_matches.append(float(m))
                            except ValueError:
                                print(f'................. wrong {key}')
                                clean_matches.append(np.nan)
                    file_metrics[key] = clean_matches
                count_psnr = len(file_metrics['psnr'])
                count_ssim = len(file_metrics['ssim'])
                row_count = min(count_psnr, count_ssim)
                if row_count > 0:
                    for i in range(row_count):
                        for key, _, _ in metrics_config:
                            val = file_metrics[key][i] if i < len(file_metrics[key]) else np.nan
                            aggregated_data[key].append(val)
                    total_samples_processed += row_count
                    print(f"  - Parsed {os.path.basename(log_file)}: {row_count} entries.")
                else:
                    print(f"  - Warning: No valid data pairs found in {log_file}")
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
    if total_samples_processed == 0:
        print("\n[ERROR] Failed to aggregate results. No data found.")
        return
    final_results = {}
    for key, _, _ in metrics_config:
        values = aggregated_data[key]
        if not values:
            final_results[key] = np.nan
        else:
            arr = np.array(values)
            mean_val = np.nanmean(arr)
            final_results[key] = mean_val
    def fmt(val, precision):
        if np.isnan(val):
            return "nan"
        return f"{val:.{precision}f}"
    output_str = (
        f"{args.dataset} dataset,"
        f"QP:{qp_value},"
        f"Model:{model_name},"
        f"psnr:{fmt(final_results['psnr'], 2)},"
        # f"psnr_avg_mse:{fmt(final_results['psnr_avg_mse'], 4)},"
        f"psnr_avg_mse:{fmt(-10 * math.log10(final_results['psnr_avg_mse']), 6)},"
        f"ssim:{fmt(final_results['ssim'], 4)},"
        f"psnr_LFR_lq:{fmt(final_results['psnr_LFR_lq'], 2)},"
        f"ssim_LFR_lq:{fmt(final_results['ssim_LFR_lq'], 4)},"
        f"psnr_LFR_hq:{fmt(final_results['psnr_LFR_hq'], 2)},"
        f"sigma:{fmt(final_results['sigma'], 4)},"
        f"psnr inter:{fmt(final_results['psnr inter'], 2)},"
        f"ssim inter:{fmt(final_results['ssim inter'], 4)},"
        f"lpips:{fmt(final_results['lpips'], 4)},"
        f"dists:{fmt(final_results['dists'], 4)},"
        f"tlpips(1e3):{fmt(final_results['tlpips'], 2)},"
        f"tof(1e1):{fmt(final_results['tof'], 4)},"
        f"warpping_psnr:{fmt(final_results['warpping_psnr'], 2)},"
        f"ave_img_bpp:{fmt(final_results['ave_img_bpp'], 6)},"
        f"vmaf:{fmt(final_results['vmaf'], 4)}"
    )
    print("\n" + "="*60)
    print("GLOBAL RESULTS AGGREGATION")
    print(f"Total Samples Processed: {total_samples_processed}")
    print("="*60)
    print(output_str)
    print("="*60)


    try:
        output_dir = "/output"
        result_filename = f"results_{args.dataset}_{model_name}.txt"
        result_path = os.path.join(output_dir, result_filename)
        
        with open(result_path, "a") as f:
            f.write(output_str + "\n")
        print(f"[INFO] Results saved to: {result_path}")
    except:
        pass


if __name__ == "__main__":
    args = parse_args()
    gpu_ids = [int(x) for x in args.gpu_list.split(',')]
    total_tasks = args.total_slices
    print(f"=== Parallel Scheduler Started ===")
    print(f"Target QP: {args.qp}, Model: {args.model}")
    BASE_OUTPUT_DIR = "/data/fengxm/vimeo90k/tvrn_revision"
    output_folder = os.path.join(
        BASE_OUTPUT_DIR, 
        args.codec_type,
        args.dataset,
        args.model,
        f"QP{args.qp}"
    )
    
    processes = []
    for i in range(total_tasks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        # 使用 Process 而不是 Pool，以便更好地控制启动时间
        p = mp.Process(target=run_task, args=(args, i, gpu_id))
        p.start()
        processes.append(p)
        
        # 如果不是最后一个任务，则等待10秒再提交下一个
        # if i < total_tasks - 1:
        print(f"[Scheduler] Waiting 10 seconds before launching next task...")
        # time.sleep(30)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 收集结果 (这里简单地假设所有进程都成功生成了log文件)
    results = []
    for i in range(total_tasks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        log_file = os.path.join(args.output_log_dir, f"log_slice_{i}_gpu_{gpu_id}.txt")
        if os.path.exists(log_file):
             results.append(log_file)
        else:
             results.append(None)

    print(f'remove the output folder: {output_folder}')
    print("=== All Tasks Finished. Aggregating... ===")
    parse_results(results, args.qp, args.model)
