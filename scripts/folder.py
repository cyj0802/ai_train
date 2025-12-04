import shutil
import random
from pathlib import Path
from typing import List, Tuple

def collect_paired_files(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """
    images_dir의 JPG와 labels_dir의 JSON을 stem(파일명) 기준으로 매칭.
    둘 다 있는 것만 반환.
    """
    image_files = sorted(images_dir.glob("*.jpg"))
    pairs: List[Tuple[Path, Path]] = []

    missing_labels = 0

    for img in image_files:
        stem = img.stem  # 예: 'xxx' from xxx.jpg
        label = labels_dir / f"{stem}.txt"
        if label.exists():
            pairs.append((img, label))
        else:
            missing_labels += 1
            print(f"⚠️ 라벨 없음 (스킵): {img.name}")

    print(f"\n✅ 매칭된 이미지-라벨 쌍: {len(pairs)}개")
    if missing_labels > 0:
        print(f"⚠️ 라벨 없는 이미지: {missing_labels}개 (위에 로그 표시됨)")
    return pairs

def split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.2):
    """
    전체 n개를 train/val/test로 나누는 인덱스 범위를 계산.
    test_ratio = 1 - train_ratio - val_ratio
    """
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test

def copy_pairs(pairs: List[Tuple[Path, Path]], out_root: Path, split_name: str):
    """
    pairs 목록을 out_root/split_name/images, out_root/split_name/labels 에 복사.
    """
    img_out_dir = out_root / split_name / "images"
    lbl_out_dir = out_root / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for img, lbl in pairs:
        shutil.copy2(img, img_out_dir / img.name)
        shutil.copy2(lbl, lbl_out_dir / lbl.name)

    print(f"📁 {split_name}: {len(pairs)}개 복사 완료")

if __name__ == "__main__":
    # ───── 경로 설정 ─────
    source = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\data")
    images_dir = source / "images"
    labels_dir = source / "labels"

    out_root = source  # train/val/test를 data 아래에 생성

    # ───── 1) 이미지-라벨 매칭 ─────
    pairs = collect_paired_files(images_dir, labels_dir)
    n = len(pairs)
    if n == 0:
        raise SystemExit("❌ 매칭된 이미지-라벨 쌍이 없습니다. 경로/확장자를 확인하세요.")

    # ───── 2) 셔플 + train/val/test 분할 ─────
    random.seed(42)  # 재현 가능하게 고정
    random.shuffle(pairs)

    n_train, n_val, n_test = split_indices(n, train_ratio=0.7, val_ratio=0.2)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"\n총 {n}개 → train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    # ───── 3) 복사 ─────
    copy_pairs(train_pairs, out_root, "train")
    copy_pairs(val_pairs, out_root, "val")
    copy_pairs(test_pairs, out_root, "test")

    print("\n✅ 데이터셋 분할 완료")
