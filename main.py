from __future__ import annotations
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict

# 기본 실행
def main():
    print("Project initialized 🚀")
    print("Python is ready for backend + JSON processing!")

    # 테스트: sample.json 읽기 예시
    test_file = Path("sample.json")
    if test_file.exists():
        with test_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print("✅ sample.json 로드 성공")
        print("버전:", data.get("version"))
    else:
        print("⚠️ sample.json 없음 (정상, 테스트용 예시입니다)")

# ─────────────────────────────────────────
# FastAPI 서버 실행 (선택)
# ─────────────────────────────────────────
def start_server():
    print("Starting FastAPI server on localhost...")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# FastAPI app 템플릿
from fastapi import FastAPI, UploadFile, File
app = FastAPI(
    title="AI Medical Device Backend",
    description="Floor-plan JSON processing + Ultrasound AI backend (template)",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "server is running ✅"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    img_bytes = await file.read()
    # 이미지 저장해보기
    save_path = Path("uploads")
    save_path.mkdir(exist_ok=True)
    output_file = save_path / file.filename
    with output_file.open("wb") as f:
        f.write(img_bytes)
    return {"filename": file.filename, "status": "uploaded ✅"}

# 실행 선택
if __name__ == "__main__":
    main()
    # FastAPI 서버도 바로 띄우려면 주석 해제 👇
    # start_server()
