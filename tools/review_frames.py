"""Contact sheet of annotated session-video frames at given times (seconds from start)."""
import sys

import cv2
import numpy as np


def sheet(video, times, out, cols=4, w=400):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, int(round(t * fps)))))
        ok, img = cap.read()
        if not ok:
            img = np.zeros((360, 640, 3), np.uint8)
        img = cv2.resize(img, (w, int(w * img.shape[0] / img.shape[1])))
        cv2.putText(img, f"{t:.1f}s", (5, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows))
    print(f"{out}: {len(times)} frames, video {n} frames @ {fps:.0f} fps = {n / fps:.0f}s")


if __name__ == "__main__":
    sheet(sys.argv[1], [float(x) for x in sys.argv[3].split(",")], sys.argv[2])
