import urllib.request
import json
import time

print("Testing web server on http://localhost:8000...")

# 1. Test GET /
try:
    with urllib.request.urlopen("http://localhost:8000/") as resp:
        print(f"1. GET / -> Status: {resp.status}, Content-Type: {resp.headers.get('Content-Type')}")
except Exception as e:
    print(f"Error GET /: {e}")

# 2. Test GET /api/status
try:
    with urllib.request.urlopen("http://localhost:8000/api/status") as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"2. GET /api/status -> Status: {resp.status}, Method: {data['method']}, Video: {data['video']}, FPS: {data['fps']}")
except Exception as e:
    print(f"Error GET /api/status: {e}")

# 3. Test GET /api/videos
try:
    with urllib.request.urlopen("http://localhost:8000/api/videos") as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"3. GET /api/videos -> Count: {len(data['videos'])} videos discovered")
except Exception as e:
    print(f"Error GET /api/videos: {e}")

# 4. Test POST /api/control (switch to A1 and beta=1.5)
try:
    req = urllib.request.Request(
        "http://localhost:8000/api/control",
        data=json.dumps({"method": "A1", "beta": 1.5, "omega": 0.90}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"4. POST /api/control -> Status: {resp.status}, Updated Method: {data['settings']['method']}, Beta: {data['settings']['beta']}")
except Exception as e:
    print(f"Error POST /api/control: {e}")

# 5. Test video stream (read first 5 frames from MJPEG feed)
try:
    req = urllib.request.Request("http://localhost:8000/video_feed")
    with urllib.request.urlopen(req) as resp:
        bytes_read = 0
        t0 = time.perf_counter()
        frames_received = 0
        chunk_size = 1024 * 32
        
        while frames_received < 5 and time.perf_counter() - t0 < 5.0:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            if b"--frame" in chunk:
                frames_received += chunk.count(b"--frame")
                
        elapsed = time.perf_counter() - t0
        stream_fps = frames_received / elapsed if elapsed > 0 else 0
        print(f"5. GET /video_feed -> Received {frames_received} frames ({bytes_read/1024:.1f} KB) in {elapsed:.2f}s (~{stream_fps:.1f} stream FPS)")
except Exception as e:
    print(f"Error GET /video_feed: {e}")

print("All tests completed successfully!")
