from ultralytics import YOLO
from pathlib import Path

def main():
    BASE = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\data")
    YAML = Path(r"T:\03_Platform\02.AI\01_Wall\01_Corner\v0.0.2")
    MODEL_PATH = Path(r"T:\03_Platform\02.AI\01_Wall\01_Corner\v0.0.2\runs_wall_corners\detect\room_junction_v11_ft6\weights\best.pt")
    
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

    # Overall metrics
    print("Mean Precision (mP):", results.box.mp)
    print("Mean Recall (mR):", results.box.mr)
    print("mAP50:", results.box.map50)
    print("mAP50-95:", results.box.map)

    # Per-class metrics
    print("Precision per class:", results.box.p)
    print("Recall per class:", results.box.r)
    print("AP50 per class:", results.box.ap50)
    print("AP50-95 per class:", results.box.ap)
    print("mAP50_per_class:", results.box.maps)

    print("=================================\n")


if __name__ == "__main__":
    main()
