import shutil
from pathlib import Path

def copy_jpgs_to_images(source_folder: Path, images_folder: Path):
    """
    source_folder 안의 모든 jpg를 images_folder로 복사합니다.
    """
    images_folder.mkdir(parents=True, exist_ok=True)

    jpg_files = list(source_folder.rglob("*.jpg"))
    if not jpg_files:
        print("⚠️ 복사할 JPG 파일이 없습니다.")
        return

    for jp in jpg_files:
        dest = images_folder / jp.name
        shutil.copy2(jp, dest)
        print(f"🖼 이미지 복사: {jp.name} → {dest}")

    print(f"\n📁 총 {len(jpg_files)}개 JPG 파일 복사 완료 ✅")

if __name__ == "__main__":
    room_folder = Path(r"T:\03_Platform\03.Floorplans\Train\Room\Room")
    images_folder = Path(r"T:\03_Platform\03.Floorplans\Train\Room\images")

    copy_jpgs_to_images(room_folder, images_folder)
