"""Prompt Reader — PNG 프롬프트 메타데이터 뷰어 (로컬 웹 서버)."""

from __future__ import annotations

import socket
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

import pngmeta

BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 22000
URL = f"http://{HOST}:{PORT}"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

app = FastAPI(title="Prompt Reader")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (최대 64MB).")
    try:
        info = pngmeta.extract_prompt_info(data)
    except pngmeta.NotPngError:
        raise HTTPException(status_code=400, detail="PNG 파일이 아닙니다. PNG 이미지를 올려주세요.")
    info["filename"] = file.filename or "image.png"
    info["pretty_chunks"] = {
        key: pretty
        for key, value in info["chunks"].items()
        if (pretty := pngmeta.pretty_json(value)) is not None
    }
    return info


def _open_browser_when_ready(url: str, port: int, timeout: float = 15.0) -> None:
    """포트가 연결을 받아들이는 시점을 기다렸다가 기본 브라우저로 연다.

    그냥 연결만 하고 끊으면 서버 쪽에 asyncio 예외 로그가 남아서(Windows
    Proactor) 정상적인 HEAD 요청으로 완결한다.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5) as sock:
                sock.settimeout(2.0)
                sock.sendall(f"GET / HTTP/1.0\r\nHost: {HOST}\r\n\r\n".encode())
                sock.recv(64)
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.2)


def serve() -> None:
    threading.Thread(target=_open_browser_when_ready, args=(URL, PORT), daemon=True).start()
    print(f"Prompt Reader 시작: {URL} (잠시 후 브라우저가 열립니다)")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    serve()
