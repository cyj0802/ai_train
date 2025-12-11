import cv2
import numpy as np

def resize_with_padding(img, size=(1600, 1280), pad_color=(0,0,0)):
    target_w, target_h = size  # width, height
    h, w = img.shape[:2]

    # 원본 비율 유지하며 스케일 계산
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 리사이즈
    resized = cv2.resize(img, (new_w, new_h))

    # 패딩 계산
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    # 패딩 추가
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded


def resize_and_save(source_path: str, target_path: str, size=(1600, 1280), pad_color=(0,0,0)):
    img = cv2.imread(source_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {source_path}")

    out = resize_with_padding(img, size=size, pad_color=pad_color)
    cv2.imwrite(target_path, out)
    return target_path


if __name__ == "__main__":
    resize_and_save(
        source_path="images/1.png",
        target_path="output.png",
        size=(1600, 1280),
    )
