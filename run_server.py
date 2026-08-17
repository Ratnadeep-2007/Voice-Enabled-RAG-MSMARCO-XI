#!/usr/bin/env python3
"""
VoiceRAG Production Server Launcher
Handles smart port allocation, instance detection, browser launching, and startup logging.
"""

import os
import sys
import time
import socket
import webbrowser
import threading
import urllib.request

# Suppress HuggingFace symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def is_voicerag_already_running(port: int, host: str = "127.0.0.1") -> bool:
    try:
        req = urllib.request.Request(f"http://{host}:{port}/", headers={"User-Agent": "VoiceRAG-Launcher"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            content = resp.read(500).decode("utf-8", errors="ignore")
            return "VoiceRAG" in content or resp.status == 200
    except Exception:
        return False


def find_best_port(default_port: int = 8000, host: str = "127.0.0.1") -> int:
    if not is_port_in_use(default_port, host):
        return default_port
    
    # If default port is in use and already running VoiceRAG
    if is_voicerag_already_running(default_port, host):
        return default_port

    # Otherwise scan for next available port
    for p in range(default_port + 1, default_port + 20):
        if not is_port_in_use(p, host):
            return p
    return default_port


def open_browser_when_ready(url: str, check_url: str, timeout: float = 30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.8)
        try:
            req = urllib.request.Request(check_url, headers={"User-Agent": "VoiceRAG-Prober"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status in (200, 404):
                    print(f"[*] VoiceRAG is ready! Opening {url} in browser...")
                    webbrowser.open(url)
                    return
        except Exception:
            continue
    # Fallback open if timeout
    webbrowser.open(url)


def main():
    host = os.getenv("HOST", "127.0.0.1")
    desired_port = int(os.getenv("PORT", "8000"))

    print("=" * 68)
    print("  VoiceRAG: Low-Latency Multilingual Voice RAG System (MSMARCO-XI)")
    print("  Reasoning Engine : Groq LPU / OpenRouter (Qwen 2.5 7B / Llama 3.3)")
    print("  Vector Storage   : FAISS IVF-PQ + LMDB Zero-Copy")
    print("  Target Latency   : Sub-200ms Retrieval Bound")
    print("=" * 68)
    print()

    # Check if already running on desired port
    if is_port_in_use(desired_port, host):
        if is_voicerag_already_running(desired_port, host):
            url = f"http://{host}:{desired_port}"
            print(f"[!] VoiceRAG server is ALREADY running and healthy at: {url}")
            print(f"[*] Opening {url} in your default browser now...")
            webbrowser.open(url)
            print()
            print("[*] To restart the server, close the existing running instance first.")
            print("[*] Press Enter to exit this launcher.")
            try:
                input()
            except Exception:
                pass
            return

    target_port = find_best_port(desired_port, host)
    url = f"http://{host}:{target_port}"

    print(f"[*] Starting VoiceRAG server on: {url}")
    print(f"[*] Dashboard:   {url}/")
    print(f"[*] API Docs:    {url}/docs")
    print(f"[*] Arch View:   {url}/architecture")
    print()

    # Launch browser prober in background thread
    t = threading.Thread(
        target=open_browser_when_ready,
        args=(url, f"{url}/"),
        daemon=True,
    )
    t.start()

    import uvicorn
    uvicorn.run("api.main:app", host=host, port=target_port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
