from ultralytics import YOLO
from pathlib import Path
import os

def main():
    # -------------------------------------------------
    # 1) 경로 설정 (로컬용)
    # -------------------------------------------------
    BASE = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\data")

    SPLITS = ["train", "val", "test"]
    IMG_DIRS = {s: BASE / s / "images" for s in SPLITS}
    TXT_DIRS = {s: BASE / s / "labels" for s in SPLITS}

    for s in SPLITS:
        os.makedirs(IMG_DIRS[s], exist_ok=True)
        os.makedirs(TXT_DIRS[s], exist_ok=True)

    # -------------------------------------------------
    # 2) 클래스 정의
    # -------------------------------------------------
    CLASSES = [
        "w1-1","w1-2","w1-3","w1-4",
        "w2-1","w2-2","w2-3","w2-4",
        "w3-1","w3-2","w3-3","w3-4",
        "w4",
        "wt1-1","wt1-2","wt1-3","wt1-4",
    ]
    NUM_CLASSES = len(CLASSES)
    print("NUM_CLASSES:", NUM_CLASSES)

    # -------------------------------------------------
    # 3) data.yaml 생성
    # -------------------------------------------------
    yaml_content = f"""
train: {IMG_DIRS['train'].as_posix()}
val: {IMG_DIRS['val'].as_posix()}
test: {IMG_DIRS['test'].as_posix()}

nc: {NUM_CLASSES}
names: {CLASSES}
"""

    data_yaml_path = BASE / "room_data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("data.yaml 저장 완료 →", data_yaml_path)

    # -------------------------------------------------
    # 4) Fine-tune 모델 로드
    # -------------------------------------------------
    MODEL_PATH = Path(r"T:\03_Platform\02.AI\01_Wall\01_Corner\v0.0.1\weights.pt")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    # -------------------------------------------------
    # 5) 학습 설정
    # -------------------------------------------------
    run_name = "room_junction_v11_ft"

    results = model.train(
        data=str(data_yaml_path),
        epochs=300,
        imgsz=(1600, 1280),
        batch=4,
        device=0,
        project=str(BASE / "runs_wall_corners" / "detect"),
        name=run_name,
        save=True,
        patience=100,
        optimizer="SGD",
        amp=True,
        lr0=0.003, lrf=0.1, momentum=0.9, weight_decay=0.0005,
        box=7.5, cls=0.3, dfl=1.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, shear=0.0, perspective=0.0,
        translate=0.02, scale=0.95,
        flipud=0.0, fliplr=0.0,
        mosaic=0.1, mixup=0.0,
    )

    print("학습 완료!")

# 반드시 필요!!
if __name__ == "__main__":
    main()
