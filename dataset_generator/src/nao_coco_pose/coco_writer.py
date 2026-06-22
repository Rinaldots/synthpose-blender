"""Acumula amostras e gera o JSON no formato COCO-pose. Python puro."""
from __future__ import annotations

import json
from pathlib import Path

from .keypoints import COCO_KEYPOINTS, COCO_SKELETON
from .visibility import V_ABSENT


class CocoDatasetBuilder:
    def __init__(self, description: str = "NAO synthetic COCO-pose (Webots)"):
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self._img_id = 0
        self._ann_id = 0
        self.info = {"description": description}
        self.categories = [{
            "id": 1,
            "name": "person",          # 'person' mantém compatibilidade com COCO
            "supercategory": "person",
            "keypoints": COCO_KEYPOINTS,
            "skeleton": COCO_SKELETON,
        }]

    def add_sample(self, file_name, width, height, keypoints_xyv, bbox, area) -> int:
        """Adiciona uma imagem + sua anotação.

        keypoints_xyv: lista de 17 tuplas (x, y, v) na ORDEM COCO.
        """
        self._img_id += 1
        self._ann_id += 1
        self.images.append({
            "id": self._img_id,
            "file_name": str(file_name),
            "width": int(width),
            "height": int(height),
        })
        flat: list[float] = []
        n_labeled = 0
        for (x, y, v) in keypoints_xyv:
            flat.extend([float(x), float(y), int(v)])
            if v != V_ABSENT:
                n_labeled += 1
        self.annotations.append({
            "id": self._ann_id,
            "image_id": self._img_id,
            "category_id": 1,
            "iscrowd": 0,
            "keypoints": flat,
            "num_keypoints": n_labeled,
            "bbox": [float(b) for b in bbox],
            "area": float(area),
        })
        return self._img_id

    def to_dict(self) -> dict:
        return {
            "info": self.info,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
