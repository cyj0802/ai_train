import os
import json
import glob
from pathlib import Path
import cv2


def convert_json_to_txt(json_path: str, out_path: str):
    """하나의 JSON 파일을 YOLO Segmentation TXT로 변환"""
    stem = os.path.splitext(os.path.basename(json_path))[0]

    # JSON 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 이미지 파일 찾기
    img_path = None
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
        candidate = os.path.join(os.path.dirname(json_path), stem + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"[WARN] 이미지 파일을 찾을 수 없음: {json_path}")
        return

    # 이미지 크기 읽기
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 이미지 로드 실패: {img_path}")
        return

    H, W = img.shape[:2]

    # shapes가 없으면 스킵
    shapes = data.get("shapes", [])
    if not shapes:
        print(f"[WARN] shapes 없음 → skip: {json_path}")
        return

    lines = []
    for sh in shapes:
        label = sh.get("label")
        points = sh.get("points")

        if not label or not points or len(points) < 3:
            continue

        # class_id는 그냥 0으로 통일 (필요하면 직접 변경)
        cls_id = 0

        yo = [str(cls_id)]
        for (x, y) in points:
            xn = max(0, min(x, W - 1)) / W
            yn = max(0, min(y, H - 1)) / H
            yo.append(f"{xn:.6f}")
            yo.append(f"{yn:.6f}")

        lines.append(" ".join(yo))

    # txt 저장
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"→ 변환 완료: {out_path}")


def convert_folder(json_folder: str, out_folder: str):
    """json 폴더 전체를 txt로 변환"""
    os.makedirs(out_folder, exist_ok=True)

    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    json_files.sort()

    for js in json_files:
        stem = os.path.splitext(os.path.basename(js))[0]
        out_txt = os.path.join(out_folder, stem + ".txt")
        convert_json_to_txt(js, out_txt)


if __name__ == "__main__":
    # 예시: json 폴더에서 txt로 변환
    INPUT_FOLDER = Path(r"T:\03_Platform\03.Floorplans\Train\Furniture\symbol\data")
    OUTPUT_FOLDER = Path(r"T:\03_Platform\03.Floorplans\Train\Furniture\symbol\labels")

    convert_folder(INPUT_FOLDER, OUTPUT_FOLDER)
