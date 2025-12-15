# import json
# from pathlib import Path
# from typing import Dict

# def replace_labels_and_save_new(json_path: Path, replacements: dict, save_folder: Path):
#     if not json_path.exists():
#         print("❌ 파일 없음:", json_path)
#         return

#     # JSON 로드
#     with json_path.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     # label 치환
#     changed = False
#     for shape in data.get("shapes", []):
#         old_label = shape.get("label")
#         if old_label in replacements:
#             shape["label"] = replacements[old_label]
#             changed = True

#     if not changed:
#         print("⚠️ 변경 라벨 없음:", json_path.name)
#         return

#     # 저장 폴더가 없으면 생성
#     save_folder.mkdir(parents=True, exist_ok=True)

#     # 치환된 JSON 저장
#     save_path = save_folder / json_path.name
#     with save_path.open("w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

#     print(f"✅ 치환 저장 완료: {save_path.name}")

# def run_folder_replace(source_folder: Path, save_folder: Path, replacements: Dict[str,str]):
#     json_files = source_folder.rglob("*.json")
#     count = 0
#     for jf in json_files:
#         replace_labels_and_save_new(jf, replacements, save_folder)
#         count += 1

#     print(f"\n📁 전체 검사 완료 ✅ 총 {count}개 JSON 검사함")

# if __name__ == "__main__":
#     source = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\corner\diag\test\labels")
#     target = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\corner\diag\test\label")
#     replace_map = {
#         "17": "13",
#     }
#     run_folder_replace(source, target, replace_map)

from pathlib import Path
from typing import Dict

def replace_txt_labels_and_save_new(
    txt_path: Path,
    replacements: Dict[str, str],
    save_folder: Path,
):
    if not txt_path.exists():
        print("❌ 파일 없음:", txt_path)
        return

    with txt_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    changed = False

    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append(line)
            continue

        parts = line.split()
        cls = parts[0]

        if cls in replacements:
            parts[0] = replacements[cls]
            changed = True

        new_lines.append(" ".join(parts))

    if not changed:
        print("⚠️ 변경 라벨 없음:", txt_path.name)
        return

    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / txt_path.name

    with save_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"✅ 치환 저장 완료: {save_path.name}")


def run_folder_replace_txt(
    source_folder: Path,
    save_folder: Path,
    replacements: Dict[str, str],
):
    txt_files = source_folder.rglob("*.txt")
    count = 0

    for tf in txt_files:
        replace_txt_labels_and_save_new(tf, replacements, save_folder)
        count += 1

    print(f"\n📁 전체 검사 완료 ✅ 총 {count}개 TXT 검사함")


if __name__ == "__main__":
    source = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\corner\diag\val\labels")
    target = Path(r"T:\03_Platform\03.Floorplans\Train\Wall\corner\diag\val\label")

    replace_map = {
        "17": "13",
    }

    run_folder_replace_txt(source, target, replace_map)
