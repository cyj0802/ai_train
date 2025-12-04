from ultralytics import YOLO
from pathlib import Path
import os

# -------------------------------------------------
# 1) 경로 설정 (로컬용으로 수정)
# -------------------------------------------------
# ★ 여기를 네 로컬 경로에 맞게 바꿔줘
# 예: BASE = Path(r"C:\Users\yujee\Documents\ai_dataset\room_corners")
BASE = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\data")

SPLITS = ["train", "val", "test"]
IMG_DIRS = {s: BASE / s / "images" for s in SPLITS}
TXT_DIRS = {s: BASE / s / "labels" for s in SPLITS}  # 변환된 YOLO TXT가 들어가는 곳

# 폴더 생성 (없으면 자동으로 생성)
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
# 3) data.yaml 생성 (로컬 경로 기준)
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
# 4) Fine-tune용 모델 로드
# -------------------------------------------------
# ★ 여기를 네가 가지고 있는 .pt 파일 경로로 수정
# 예: MODEL_PATH = Path(r"C:\Users\yujee\Documents\models\room_junction_v11.pt")
MODEL_PATH = Path(r"T:\03_Platform\02.AI\01_Wall\01_Corner\v0.0.1\weights.pt")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

# 이미 학습된 .pt를 불러와서 그대로 이어서 학습(Fine-tune)
model = YOLO(str(MODEL_PATH))

# -------------------------------------------------
# 5) 학습 설정 (로컬용 Fine-tuning)
# -------------------------------------------------
run_name = "room_junction_v11_ft"  # 결과 폴더 이름 (fine-tune 버전이라 ft 붙임)

results = model.train(
    data=str(data_yaml_path),
    epochs=300,
    # 이미지 실제 크기: 1600 x 1280 → (H, W) = (1280, 1600)
    imgsz=(1280, 1600),
    batch=2,                 # GPU 여유되면 4, 8로 올려도 됨
    device=0,                # GPU 0번 사용 / GPU 없으면 "cpu"로 변경

    project=str(BASE / "runs_wall_corners" / "detect"),
    name=run_name,
    save=True,
    patience=100,
    optimizer="SGD",         # 필요시 "AdamW" 등으로 변경 가능
    amp=True,

    # 하이퍼파라미터 (기존 값 유지)
    lr0=0.003, lrf=0.1, momentum=0.9, weight_decay=0.0005,
    box=7.5, cls=0.3, dfl=1.5,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, shear=0.0, perspective=0.0,
    translate=0.02, scale=0.95,
    flipud=0.0, fliplr=0.0,
    mosaic=0.1, mixup=0.0,
)

print("학습 완료!")
