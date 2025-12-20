# train_room_local.py
import os
import glob
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ─────────────────────────────────────────
# 0. 경로/기본 설정 (❗여기만 너 환경에 맞게 확인)
# ─────────────────────────────────────────
# 데이터 루트 (train/val/test 안에 images/labels 있는 최상위)
BASE = Path(r"T:\03_Platform\03.Floorplans\Train\Room\data")

# room_data.yaml 이 저장될 위치
DATA_YAML_PATH = BASE / "room_data.yaml"

# LabelMe json이 같은 구조로 BASE 아래에 있다고 가정
SPLITS = ["train", "val", "test"]

# 🔥 여기에는 "fine-tune에 사용할 가중치" 경로 넣어줘
# 1) 콜랩에서 학습한 best.pt 를 로컬로 가져왔다면:
# PRETRAINED_WEIGHTS = Path(r"C:\Users\yujee\OneDrive\문서\GitHub\ai_train\models\room_junction_v11_best.pt")
# 2) 처음부터 yolo11m-seg.pt로 시작하고 싶으면:
# PRETRAINED_WEIGHTS = "yolo11m-seg.pt"
PRETRAINED_WEIGHTS = Path(r"T:\03_Platform\02.AI\03_Room\01_Area\v0.0.1\weights.pt")

# ─────────────────────────────────────────
# 1. 클래스 정의 (콜랩 코드 그대로)
# ─────────────────────────────────────────
CLASSES = [
    "r1", "r2", "r3", "r4",
    "r5", "r6", "r7", "r7-1",
    "r7-2", "r7-3", "r8", "r9",
    "r10", "r10-1", "r11"
]
NUM_CLASSES = len(CLASSES)

SYNONYMS = {
    "r1":    ["침실"],
    "r2":    ["화장실"],
    "r3":    ["거실"],
    "r4":    ["발코니"],
    "r5":    ["duck"],
    "r6":    ["현관"],
    "r7":    ["회색"],
    "r7-1":  ["창고"],
    "r7-2":  ["다용도실"],
    "r7-3":  ["실외기"],
    "r8":    ["테라스"],
    "r9":    ["주방 및 식당"],
    "r10":   ["드레스룸"],
    "r10-1": ["파우더룸"],
    "r11":   ["복도"],
}

CLASS_TO_ID = {n: i for i, n in enumerate(CLASSES)}


# ─────────────────────────────────────────
# 2. LabelMe → YOLO Seg 변환 함수 (콜랩 코드 로컬용)
# ─────────────────────────────────────────
def normalize_label(s: str) -> str:
    s0 = (s or "").strip()
    s1 = s0.replace(" ", "")
    s2 = s1.lower()
    if s2 in {f"r{i}" for i in range(1, 11)}:
        return s2
    for k, arr in SYNONYMS.items():
        for a in arr:
            if s1.lower() == a.lower():
                return k
    return s2

def rectangle_to_polygon(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_yolo_seg_line(cls_id, pts, W, H):
    nums = []
    for (x, y) in pts:
        xn = clamp(x, 0, W - 1) / W
        yn = clamp(y, 0, H - 1) / H
        nums.extend([f"{xn:.6f}", f"{yn:.6f}"])
    return f"{cls_id} " + " ".join(nums)

def find_json_for_image(img_path: str, split_root: str):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    candidates = [
        os.path.join(os.path.dirname(img_path), f"{stem}.json"),
        os.path.join(os.path.dirname(img_path), f"{stem}.JSON"),
    ]
    for folder in ["labels", "annotations", "json", "labelme", "labels_json"]:
        candidates += [
            os.path.join(split_root, folder, f"{stem}.json"),
            os.path.join(split_root, folder, f"{stem}.JSON"),
        ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def read_size_and_json(json_path, img_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        W = data.get("imageWidth")
        H = data.get("imageHeight")
        if isinstance(W, int) and isinstance(H, int):
            return W, H, data
        im = cv2.imread(img_path)
        if im is not None:
            H, W = im.shape[:2]
            return W, H, data
    except Exception:
        pass
    return None, None, None

def convert_labelme_to_yolo_seg(dataset_root: Path, splits):
    """
    BASE/train/images, BASE/val/images, BASE/test/images 아래에
    LabelMe json이 있을 때 YOLO seg txt로 변환.
    """
    print("🔁 LabelMe → YOLO Seg 변환 시작...")
    for split in splits:
        split_root = dataset_root / split
        img_dir = split_root / "images"
        out_dir = split_root / "labels"
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = []
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
                ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF")
        for ext in exts:
            imgs.extend(glob.glob(str(img_dir / f"*{ext}")))
        imgs.sort()

        for img_path in imgs:
            js = find_json_for_image(img_path, str(split_root))
            if js is None:
                continue
            W, H, data = read_size_and_json(js, img_path)
            if not (isinstance(W, int) and isinstance(H, int)):
                continue

            lines = []
            for sh in data.get("shapes", []):
                st = sh.get("shape_type", "polygon")
                if st not in ("polygon", "rectangle"):
                    continue
                norm = normalize_label(sh.get("label", ""))
                if norm not in CLASS_TO_ID:
                    continue
                cid = CLASS_TO_ID[norm]
                pts = sh.get("points", [])
                if st == "rectangle" and len(pts) == 2:
                    pts = rectangle_to_polygon(pts[0], pts[1])
                if len(pts) < 3:
                    continue
                lines.append(to_yolo_seg_line(cid, pts, W, H))

            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_txt = out_dir / f"{stem}.txt"
            if lines:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
            else:
                if out_txt.exists():
                    out_txt.unlink()
    print("✅ LabelMe → YOLO Seg 변환 완료\n")


# ─────────────────────────────────────────
# 3. room_data.yaml 생성 (콜랩 data.yaml 로컬 버전)
# ─────────────────────────────────────────
def make_data_yaml(yaml_path: Path, base: Path):
    img_dirs = {
        "train": str(base / "train" / "images"),
        "val":   str(base / "val" / "images"),
        "test":  str(base / "test" / "images"),
    }

    yaml_content = f"""
train: {img_dirs['train']}
val: {img_dirs['val']}
test: {img_dirs['test']}

nc: {NUM_CLASSES}
names: {CLASSES}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print("✅ room_data.yaml 저장 완료 →", yaml_path, "\n")


# ─────────────────────────────────────────
# 4. 학습(fine-tune) 실행
# ─────────────────────────────────────────
def main():
    # 1) (옵션) LabelMe → YOLO Seg 변환
    # 이미 txt로 다 변환되어 있다면 이 줄은 주석 처리해도 됨.
    convert_labelme_to_yolo_seg(BASE, SPLITS)

    # 2) data.yaml 생성
    make_data_yaml(DATA_YAML_PATH, BASE)

    # 3) 디바이스 설정
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 4) 모델 로드 (fine-tune)
    model = YOLO(str(PRETRAINED_WEIGHTS))  # yolo11m-seg.pt 또는 room_junction_v11 best.pt 등

    run_name = "room_junction_v11_ft_local"

    # 5) 학습
    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=100,
        imgsz=1280,          # 먼저 안전한 기준으로
        batch=2,
        device=device,
        project="runs_room_again/segment_local",
        name=run_name,
        save=True,
        patience=0,          # early stopping off
        optimizer="AdamW",   # 파인튜닝 안정적 (SGD 유지해도 되지만 AdamW가 무난)
        amp=(device != "cpu"),

        lr0=0.0003, lrf=0.1, momentum=0.9, weight_decay=0.0005,
        box=7.5, cls=0.3, dfl=1.5,

        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, shear=0.0, perspective=0.0,
        translate=0.01, scale=0.3,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0,

        workers=0,
        save_period=10,
    )

    print("🏁 학습 종료, 결과 폴더:", results.save_dir)


if __name__ == "__main__":
    main()
