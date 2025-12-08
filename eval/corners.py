from ultralytics import YOLO
from pathlib import Path

def main():
    BASE = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\data")
    YAML = Path(r"T:\03_Platform\02.AI\01_Wall\01_Corner\v0.0.2\data")
    MODEL_PATH = Path(r"T:\03_Platform\02.AI\01_Wall\01_Corner\v0.0.2\data\runs_wall_corners\detect\room_junction_v11_ft6\weights\best.pt")
    
    DATA_YAML = YAML / "room_data.yaml"  # 이미 네 코드에서 생성된 yaml

    model = YOLO(str(MODEL_PATH))

    # YOLO val() 호출 (test 세트 평가)
    results = model.val(
        data=str(DATA_YAML),
        split="test",          # ← test 폴더 사용
        imgsz=(1600, 1280),
        batch=4,
        device=0
    )

    metrics = results.metrics

    print("\n===== Evaluation Results =====")
    print(f"Precision     : {metrics.precision:.4f}")
    print(f"Recall        : {metrics.recall:.4f}")
    print(f"mAP50         : {metrics.map50:.4f}")
    print(f"mAP50-95      : {metrics.map:.4f}")
    print("==============================\n")

if __name__ == "__main__":
    main()
