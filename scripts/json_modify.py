import os
import json
import shutil

# 경로 설정
label_dir = r"T:\03_Platform\03.Floorplans\Train\Furniture\symbol\labels"
output_dir = r"T:\03_Platform\03.Floorplans\Train\Furniture\symbol\images"

# 출력 폴더 없으면 생성
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(label_dir):
    if filename.lower().endswith(".json"):
        json_path = os.path.join(label_dir, filename)

        # JSON 파일 읽기
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # imagePath를 json 파일명에서 확장자만 jpg로 변경
        base_name = os.path.splitext(filename)[0]  # 파일명(확장자 제거)
        new_image_name = base_name + ".jpg"

        data["imagePath"] = new_image_name
        data["imageData"] = None  # null 로 저장됨

        # 저장할 경로
        new_json_path = os.path.join(output_dir, filename)

        # 수정된 JSON 저장
        with open(new_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

print("처리 완료!")
