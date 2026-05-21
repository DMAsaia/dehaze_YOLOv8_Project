import argparse
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
    parser = argparse.ArgumentParser(description='Save hazy/dehaze/clean triplets for YOLOv8-dehaze.')
    parser.add_argument('--weights', type=Path, default=Path('runs/detect/train6/weights/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('datasets/VOC_hazy/VOC_hazy.yaml'))
    parser.add_argument('--split', type=str, default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--num', type=int, default=12)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--out-dir', type=Path, default=Path('runs/dehaze_vis/train6'))
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


def to_display_img(tensor):
    img = tensor.detach().float().clamp(0, 1).cpu().numpy()
    img = (img.transpose(1, 2, 0) * 255).round().astype(np.uint8)  # RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def add_title(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def letterbox_pair(hazy, clean, new_shape=640):
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
    return hazy, clean


def save_triplet(index, hazy_bgr, dehaze_bgr, clean_bgr, out_dir, stem):
    panels = [
        add_title(hazy_bgr, 'Hazy input'),
        add_title(dehaze_bgr, 'Dehaze output'),
        add_title(clean_bgr, 'Clean target'),
    ]
    triplet = np.concatenate(panels, axis=1)
    out_file = out_dir / f'{index:02d}_{stem}_triplet.jpg'
    cv2.imwrite(str(out_file), triplet)
    return out_file


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

    pairs = resolve_split_paths(args.data, args.split)[:args.num]
    saved = []

    for i, (hazy_path, clean_path) in enumerate(pairs):
        hazy = cv2.imread(str(hazy_path))
        clean = cv2.imread(str(clean_path))
        if hazy is None or clean is None:
            continue

        hazy_lb, clean_lb = letterbox_pair(hazy, clean, args.imgsz)

        x = np.ascontiguousarray(hazy_lb.transpose(2, 0, 1)[::-1])  # BGR HWC -> RGB CHW
        x = torch.from_numpy(x).to(device).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            out = model(x)
        if not (isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[1])):
            raise RuntimeError('Model did not return a dehaze image. Check model.return_dehaze and model.training.')

        dehaze = out[1][0]
        dehaze_bgr = to_display_img(dehaze)
        if dehaze_bgr.shape[:2] != hazy_lb.shape[:2]:
            dehaze_bgr = cv2.resize(dehaze_bgr, (hazy_lb.shape[1], hazy_lb.shape[0]), interpolation=cv2.INTER_LINEAR)

        saved.append(save_triplet(i, hazy_lb, dehaze_bgr, clean_lb, args.out_dir, hazy_path.stem))

    print(f'Saved {len(saved)} triplets to {args.out_dir.resolve()}')
    for p in saved:
        print(p)


if __name__ == '__main__':
    main()
