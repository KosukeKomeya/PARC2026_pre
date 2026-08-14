#!/usr/bin/env python3
"""Compatibility shim for LeRobot v0.4.4 on Torchvision 0.26.

LeRobot v0.4.4's ``pyav`` backend uses ``torchvision.io.VideoReader``.
Torchvision 0.26, which accompanies the PARC PyTorch 2.11 stack, removed that
API.  This module restores only the small VideoReader interface LeRobot uses,
backed directly by the already-pinned PyAV dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


def install_video_reader_compat() -> bool:
    """Install a PyAV VideoReader replacement when Torchvision removed it.

    Returns ``True`` when the replacement was installed and ``False`` when
    the runtime still provides its native VideoReader implementation.
    """
    import av
    import torch
    import torchvision

    if hasattr(torchvision.io, "VideoReader"):
        return False

    class PyAVVideoReaderCompat:
        """Subset of torchvision.io.VideoReader required by LeRobot 0.4.4."""

        def __init__(self, source: str | Path, stream: str = "video") -> None:
            if stream != "video":
                raise ValueError(f"Only the video stream is supported, got {stream!r}")
            self.container = av.open(str(source))
            if not self.container.streams.video:
                self.container.close()
                raise RuntimeError(f"No video stream found in {source}")
            self.stream = self.container.streams.video[0]
            self.stream.thread_type = "AUTO"

        def seek(
            self,
            time_s: float,
            keyframes_only: bool = False,
        ) -> "PyAVVideoReaderCompat":
            time_base = self.stream.time_base
            if time_base is None:
                raise RuntimeError("Video stream has no time base")
            offset = max(0, int(float(time_s) / float(time_base)))
            self.container.seek(
                offset,
                stream=self.stream,
                backward=True,
                any_frame=not keyframes_only,
            )
            return self

        def __iter__(self) -> Iterator[dict[str, Any]]:
            for frame in self.container.decode(self.stream):
                if frame.pts is None:
                    continue
                frame_time_base = frame.time_base or self.stream.time_base
                if frame_time_base is None:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                yield {
                    "data": torch.from_numpy(rgb).permute(2, 0, 1),
                    "pts": float(frame.pts * frame_time_base),
                }

    torchvision.io.VideoReader = PyAVVideoReaderCompat
    print(
        "PI05_VIDEO_COMPAT installed PyAV reader for Torchvision 0.26",
        flush=True,
    )
    return True
