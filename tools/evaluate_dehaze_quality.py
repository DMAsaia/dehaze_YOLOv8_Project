import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import DehazeFeatureFuse, DehazeFeatureFuseSkip, DehazeFeatureFuseSkipResidual, DehazeHead
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.utils import yaml_load


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate dehazing quality on paired hazy/clean images.')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('datasets/VOC_hazy/VOC_hazy.yaml'))
    parser.add_argument('--split', type=str, default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--num', type=int, default=0, help='Number of images to evaluate. 0 means all paired images.')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--out-dir', type=Path, default=Path('runs/dehaze_quality/eval'))
    parser.add_argument('--include-padding', action='store_true',
                        help='Include letterbox padding in metrics. Default evaluates only the resized image area.')
    parser.add_argument('--print-every', type=int, default=100)
    return parser.parse_args()


def resolve_split_paths(data_yaml, split):
    data = yaml_load(data_yaml)
    root = Path(data.get('path', data_yaml.parent))
    image_dir = Path(data[split])
    clean_dir = Path(data.get('clean', 'clean'))
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    if not clean_dir.is_absolute():
        clean_dir = root / clean_dir
    clean_split_dir = clean_dir / split
    image_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    pairs = [(p, clean_split_dir / p.name) for p in image_files if (clean_split_dir / p.name).exists()]
    if not pairs:
        raise FileNotFoundError(f'No hazy/clean pairs found for split={split}: {image_dir} -> {clean_split_dir}')
    return pairs


def letterbox_pair_with_mask(hazy, clean, new_shape=640):
    shape = hazy.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        hazy = cv2.resize(hazy, new_unpad, interpolation=cv2.INTER_LINEAR)
        clean = cv2.resize(clean, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    hazy = cv2.copyMakeBorder(hazy, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    clean = cv2.copyMakeBorder(clean, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    mask = np.zeros(hazy.shape[:2], dtype=bool)
    mask[top:top + new_unpad[1], left:left + new_unpad[0]] = True
    crop = (top, left, top + new_unpad[1], left + new_unpad[0])
    return hazy, clean, mask, crop


def bgr_to_rgb_float(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def tensor_to_rgb_float(tensor):
    img = tensor.detach().float().clamp(0, 1).cpu().numpy()
    return img.transpose(1, 2, 0).astype(np.float32)


def psnr_from_mse(mse):
    return float('inf') if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def ssim_rgb(a, b):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for ch in range(3):
        x = a[..., ch].astype(np.float32)
        y = b[..., ch].astype(np.float32)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
        scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


def gradient_l1(a, b):
    scores = []
    for ch in range(3):
        x = a[..., ch].astype(np.float32)
        y = b[..., ch].astype(np.float32)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        tx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        ty = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        scores.append(float(np.mean(np.abs(gx - tx) + np.abs(gy - ty))))
    return float(np.mean(scores))


def compute_metrics(pred, target):
    diff = pred - target
    l1 = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    return {
        'l1': l1,
        'mse': mse,
        'psnr': psnr_from_mse(mse),
        'ssim': ssim_rgb(pred, target),
        'grad_l1': gradient_l1(pred, target),
    }


def crop_valid(img, crop):
    top, left, bottom, right = crop
    return img[top:bottom, left:right]


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.device}' if args.device and args.device != 'cpu' and torch.cuda.is_available()
                          else 'cpu')
    model, _ = attempt_load_one_weight(str(args.weights), device=device)
    if not any(isinstance(m, (DehazeFeatureFuse, DehazeFeatureFuseSkip, DehazeFeatureFuseSkipResidual, DehazeHead))
               for m in model.modules()):
        raise RuntimeError(f'{args.weights} does not contain a dehaze output module.')

    # Keep child modules in eval mode, but enable BaseModel's dehaze return gate.
    model.eval()
    model.training = True
    model.return_dehaze = True

    pairs = resolve_split_paths(args.data, args.split)
    if args.num > 0:
        pairs = pairs[:args.num]

    rows = []
    for i, (hazy_path, clean_path) in enumerate(pairs):
        hazy = cv2.imread(str(hazy_path))
        clean = cv2.imread(str(clean_path))
        if hazy is None or clean is None:
            continue

        hazy_lb, clean_lb, mask, crop = letterbox_pair_with_mask(hazy, clean, args.imgsz)
        x = np.ascontiguousarray(hazy_lb.transpose(2, 0, 1)[::-1])
        x = torch.from_numpy(x).to(device).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            out = model(x)
        if not (isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[1])):
            raise RuntimeError('Model did not return a dehaze image. Check model.return_dehaze and model.training.')

        dehaze_rgb = tensor_to_rgb_float(out[1][0])
        if dehaze_rgb.shape[:2] != hazy_lb.shape[:2]:
            dehaze_rgb = cv2.resize(dehaze_rgb, (hazy_lb.shape[1], hazy_lb.shape[0]), interpolation=cv2.INTER_LINEAR)

        hazy_rgb = bgr_to_rgb_float(hazy_lb)
        clean_rgb = bgr_to_rgb_float(clean_lb)

        if not args.include_padding:
            hazy_rgb = crop_valid(hazy_rgb, crop)
            clean_rgb = crop_valid(clean_rgb, crop)
            dehaze_rgb = crop_valid(dehaze_rgb, crop)

        hazy_metrics = compute_metrics(hazy_rgb, clean_rgb)
        dehaze_metrics = compute_metrics(dehaze_rgb, clean_rgb)

        row = {
            'index': i,
            'image': str(hazy_path),
            'clean': str(clean_path),
            'valid_pixels': int(mask.sum()) if not args.include_padding else int(mask.size),
        }
        for key, value in hazy_metrics.items():
            row[f'hazy_{key}'] = value
        for key, value in dehaze_metrics.items():
            row[f'dehaze_{key}'] = value
        row['delta_l1'] = row['dehaze_l1'] - row['hazy_l1']
        row['delta_mse'] = row['dehaze_mse'] - row['hazy_mse']
        row['delta_psnr'] = row['dehaze_psnr'] - row['hazy_psnr']
        row['delta_ssim'] = row['dehaze_ssim'] - row['hazy_ssim']
        rows.append(row)

        if args.print_every > 0 and (i + 1) % args.print_every == 0:
            print(f'Evaluated {i + 1}/{len(pairs)} images')

    if not rows:
        raise RuntimeError('No images were evaluated.')

    fieldnames = list(rows[0].keys())
    per_image_csv = args.out_dir / 'per_image_metrics.csv'
    with per_image_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [k for k in rows[0] if k not in {'index', 'image', 'clean'}]
    summary = {
        'weights': str(args.weights),
        'data': str(args.data),
        'split': args.split,
        'imgsz': args.imgsz,
        'num_images': len(rows),
        'include_padding': args.include_padding,
    }
    for key in metric_keys:
        values = np.array([r[key] for r in rows], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        summary[f'{key}_mean'] = float(finite_values.mean()) if finite_values.size else float('inf')
        summary[f'{key}_std'] = float(finite_values.std()) if finite_values.size else 0.0

    summary_json = args.out_dir / 'summary.json'
    summary_csv = args.out_dir / 'summary.csv'
    summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f'Evaluated {len(rows)} images')
    print(f'Per-image metrics: {per_image_csv.resolve()}')
    print(f'Summary JSON: {summary_json.resolve()}')
    print(f'Summary CSV: {summary_csv.resolve()}')
    print('Mean metrics:')
    for prefix in ('hazy', 'dehaze', 'delta'):
        keys = [k for k in summary if k.startswith(prefix) and k.endswith('_mean')]
        for key in keys:
            print(f'  {key}: {summary[key]:.6f}')


if __name__ == '__main__':
    main()
