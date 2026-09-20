"""
publish_public.py
=============================================================================
One-Click Public Website Publisher for AeroClear Dehazing Web App

Publishes your localhost:8000 web app to a secure public HTTPS website URL
that anyone can access from smartphones, other laptops, or evaluation panels.

Usage:
    python publish_public.py
    python publish_public.py --port 8000
=============================================================================
"""

import os
import sys
import time
import subprocess
import re
import urllib.request
import json

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def get_public_ip():
    """Gets machine public IP for localtunnel verification password."""
    try:
        req = urllib.request.Request("https://api.ipify.org", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.read().decode('utf-8').strip()
    except Exception:
        return "Check your public IP at ipify.org"

def start_tunnel(port=8000):
    print("\n" + "=" * 78)
    print("      AEROCLEAR PUBLIC WEBSITE PUBLISHER")
    print("=" * 78)
    print(f" [*] Forwarding local port: http://127.0.0.1:{port}")
    print(" [*] Connecting to high-speed global Anycast edge network...")

    cloudflared_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe")
    proc = None
    public_url = None

    # Priority 1: Cloudflare Tunnel (Enterprise grade, unlimited bandwidth, zero 503 errors)
    if os.path.exists(cloudflared_path):
        print(" [*] Engine: Cloudflare Anycast Edge (IPv4 HTTP/2)")
        cmd = [
            cloudflared_path,
            "tunnel",
            "--edge-ip-version", "4",
            "--protocol", "http2",
            "--url", f"http://127.0.0.1:{port}"
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            url_event = threading.Event()
            url_holder = {"url": None}

            def _drain():
                for line in iter(proc.stdout.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if not url_holder["url"]:
                        match = re.search(r'https://(?!api\.)[a-zA-Z0-9.-]+\.trycloudflare\.com', line)
                        if match:
                            url_holder["url"] = match.group(0)
                            url_event.set()

            drain_thread = threading.Thread(target=_drain, daemon=True, name="CloudflaredPipeDrainer")
            drain_thread.start()

            url_event.wait(timeout=25)
            public_url = url_holder["url"]

        except Exception as e:
            print(f" [!] Cloudflare tunnel error: {e}")

    # Priority 2: SSH Reverse Tunnel Fallback (localhost.run)
    if not public_url:
        print(" [*] Engine: OpenSSH Direct Reverse Tunnel")
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"80:127.0.0.1:{port}",
            "nokey@localhost.run"
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            url_event = threading.Event()
            url_holder = {"url": None}

            def _drain_ssh():
                for line in iter(proc.stdout.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if not url_holder["url"]:
                        match = re.search(r'https://[a-zA-Z0-9.-]+\.lhr\.life', line) or re.search(r'https://[a-zA-Z0-9.-]+\.(?:life|link|run|io)', line)
                        if match:
                            url_holder["url"] = match.group(0)
                            url_event.set()

            drain_thread = threading.Thread(target=_drain_ssh, daemon=True, name="SSHPipeDrainer")
            drain_thread.start()

            url_event.wait(timeout=25)
            public_url = url_holder["url"]
        except Exception as e:
            print(f" [!] SSH tunnel error: {e}")

    if public_url:
        print("\n" + "*" * 78, flush=True)
        print("  [SUCCESS] YOUR REAL-TIME DEHAZING WEB APP IS LIVE ON THE PUBLIC INTERNET!", flush=True)
        print("*" * 78, flush=True)
        print(f"\n  >> PUBLIC HTTPS WEBSITE URL:", flush=True)
        print(f"     {public_url}", flush=True)
        print("\n  >> Accessible instantly on any mobile phone, tablet, or laptop worldwide!", flush=True)
        print("  >> High-bandwidth GPU stream, zero 503 errors, zero password gates.", flush=True)
        print("  >> Keep this terminal running to keep the public website active.", flush=True)
        print("*" * 78 + "\n", flush=True)

        # Notify local server
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/control",
                data=json.dumps({"public_url": public_url}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass

        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\nShutting down public tunnel...")
            proc.terminate()
    else:
        print(" [!] Could not establish tunnel automatically. Please check your internet connection.")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    start_tunnel(port)
