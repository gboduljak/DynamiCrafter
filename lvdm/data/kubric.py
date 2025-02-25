from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Kubric(Dataset):
    def __init__(self, root_dir: Union[Path, str], max_dataset_size: Optional[int] = None):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.videos = list(sorted(root_dir.iterdir()))
        if max_dataset_size is not None:
            self.videos = self.videos[:max_dataset_size]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int, torch.Tensor]]:
        path = self.videos[idx]
        cap = cv2.VideoCapture(path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

        cap.release()

        frames = (
            torch.tensor(np.stack(frames))
            .permute(3, 0, 1, 2)
            .float()
        )
        frames = (frames / 255 - 0.5) * 2
        # [t,h,w,c] -> [c,t,h,w]

        return {
            'video': frames,
            'caption': "",
            'path': str(path),
            'fps': 12,
            'frame_stride': 1
        }