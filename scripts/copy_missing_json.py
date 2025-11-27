import shutil
from pathlib import Path

def copy_missing_jsons(source_folder: Path, target_folder: Path):
    # target 폴더 없으면 생성
    target_folder.mkdir(parents=True, exist_ok=True)

    src_files = list(source_folder.rglob("*.json"))
    missing = []

    for sf in src_files:
        tf = target_folder / sf.name
        if not tf.exists():  # ✅ replace에 없으면 복사 대상
            missing.append((sf, tf))

    if not missing:
        print("🎉 복사할 신규 JSON 없음 (모두 replace에 존재)")
        return

    for sf, tf in missing:
        shutil.copy2(sf, tf)
        print(f"📄 JSON 복사: {sf.name} → {tf}")

    print(f"\n📁 복사 완료 ✅ 총 {len(missing)}개 JSON 파일이 새로 추가됨")

# 실행
if __name__ == "__main__":
    source = Path(r"T:\03_Platform\03.Floorplans\Train\Room\Room")
    target = Path(r"T:\03_Platform\03.Floorplans\Train\Room\replace")

    copy_missing_jsons(source, target)
