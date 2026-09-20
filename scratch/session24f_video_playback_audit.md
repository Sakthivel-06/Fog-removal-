# Session 24F Video Playback Audit

## 1. Problem Identification
The user reported that the Streamlit UI shows an FPS of ~19, but the resulting output visually plays at an incorrect speed, making vehicles appear almost stationary.

## 2. Root Cause
Upon auditing `app/main.py`, it was discovered that the Streamlit application **does not encode or save an output video**. Instead, it displays the processed frames sequentially in a loop using `st.image()`.
- The update rate of the browser display is constrained by the processing speed (e.g., 19 FPS) and network/rendering latency.
- A 30 FPS source video displayed at 19 FPS natively forces it into slow motion, making vehicles look artificially slow.
- The pipeline itself is sound; the problem lies entirely within the Streamlit UI presentation logic.

## 3. Recommended Fixes
To resolve this while maintaining zero impact on the frozen algorithm (`novel_method/pipeline.py`), the following modifications to the UI (`app/main.py`) are required:

1.  **Extract Source Metadata:** Read the true FPS from the source video via `imageio.get_reader().get_meta_data()['fps']`.
2.  **Add Execution Modes:**
    *   **Mode A (Process and export complete video):** Process the requested frames and save them to a temporary `.mp4` file using `imageio.get_writer` set to the exact `source_fps`. Once processing is complete, display the video natively using `st.video()`.
    *   **Mode B (Live Preview - Frame-by-Frame):** Retain the current frame-by-frame `st.image()` updating for deep inspection.
3.  **Enhance Telemetry HUD:**
    *   Explicitly show **Source FPS**, **Output Playback FPS**, and **Processing Speed**.
    *   Provide clear disclaimers that live preview speed is bottlenecked by Streamlit and processing overhead, whereas the final exported video runs at the correct Native FPS.
4.  **Label Outputs:** Clearly differentiate "Frozen A4 output" and "High-Resolution A4" in the UI outputs.

## 4. Implementation Plan
I will modify `app/main.py` using `multi_replace_file_content` to inject the `imageio.get_writer` logic and the UI mode selectors, ensuring no dehazing logic is altered.
