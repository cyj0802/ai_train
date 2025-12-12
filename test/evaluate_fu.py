# evaluate_room.py
import torch
from ultralytics import YOLO
from pathlib import Path

# ----------------------------
# 0. 경로 설정
# ----------------------------
# 네가 학습에서 사용했던 BASE 경로 그대로
BASE = Path(r"T:\03_Platform\03.Floorplans\Train\Furniture\symbol\data")

DATA_YAML_PATH = BASE / "room_data.yaml"

# 평가할 가중치 (학습이 끝나면 runs_room/.../weights/best.pt 로 생성됨)
WEIGHTS = "runs_room/segment_local/room_junction_v11_ft_local2/weights/best.pt"


# ----------------------------
# 1. mAP 평가
# ----------------------------
def main():
    # GPU or CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 모델 로드
    model = YOLO(str(WEIGHTS))

    # 평가 (test split)
    results = model.val(
        data=str(DATA_YAML_PATH),
        split="test",      # test images 사용
        imgsz=1600,
        device=device,
        save_json=False,
        conf=0.001,
        iou=0.5  # mAP50만 보고 싶으면 0.5로 평가
    )

    print("\n===== Evaluation Results =====")

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
