import zlib
from pathlib import Path

import av
import numpy as np


def extract_intrinsic_features(video_path: str) -> np.ndarray:
    if not Path(video_path).exists():
        raise FileNotFoundError(video_path)

    container = av.open(video_path)
    stream = container.streams.video[0]

    codec_id_hash = float(zlib.crc32(stream.codec_context.name.encode()) % 1000)
    frame_count = 0
    pixel_diffs = []
    prev_frame = None

    for frame in container.decode(stream):
        array = frame.to_ndarray(format="gray")
        if prev_frame is not None:
            pixel_diffs.append(np.abs(array.astype(np.float32) - prev_frame.astype(np.float32)).std())
        prev_frame = array
        frame_count += 1

    container.close()

    mean_compression_qp = float(np.mean(pixel_diffs)) if pixel_diffs else 0.0
    noise_residual_std = float(np.std(pixel_diffs)) if pixel_diffs else 0.0
    frame_count_mod = float(frame_count % 100)

    return np.array([mean_compression_qp, noise_residual_std, codec_id_hash, frame_count_mod])
