from pathlib import Path
import shutil


def collect_preview_images(base_dir: Path) -> None:
    """
    base_dir 하위의 각 폴더에서 preview.png를 찾아
    images/{폴더명}.png 형태로 복사한다.
    """
    images_dir = base_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue

        if subdir.name == "images":
            continue

        preview_img = subdir / "preview_furniture_rotation.png"
        if not preview_img.exists():
            continue

        target_img = images_dir / f"{subdir.name}.png"
        shutil.copy2(preview_img, target_img)

        print(f"[OK] {subdir.name} → {target_img.name}")


if __name__ == "__main__":
    BASE_DIR = Path(r"T:\05_AI\furn\json_images\10300_Dong-Gaepo")
    collect_preview_images(BASE_DIR)
