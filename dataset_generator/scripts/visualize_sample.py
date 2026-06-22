"""Sobrepõe keypoints + esqueleto + bbox numa imagem para validar a projeção.

É o teste mais importante do pipeline: se os pontos não caem nas juntas, o erro
está na projeção (projection.py, incluindo AXIS_REMAP) ou no mapeamento
(keypoints.py, em especial os offsets faciais).

Uso:
    python scripts/visualize_sample.py \
        output/annotations/person_keypoints.json output/images 0 overlay.png
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(ann_path: str, img_dir: str, index: str, out_path: str) -> None:
    import cv2
    from pycocotools.coco import COCO

    coco = COCO(ann_path)
    img_ids = coco.getImgIds()
    img_info = coco.loadImgs(img_ids[int(index)])[0]
    ann = coco.loadAnns(coco.getAnnIds(imgIds=img_info["id"]))[0]
    skel = coco.loadCats(coco.getCatIds())[0]["skeleton"]

    img = cv2.imread(str(Path(img_dir) / img_info["file_name"]))
    if img is None:
        print(f"imagem não encontrada: {img_info['file_name']}")
        return

    kp = ann["keypoints"]
    pts = [(kp[i * 3], kp[i * 3 + 1], kp[i * 3 + 2]) for i in range(len(kp) // 3)]

    for a, b in skel:  # arestas em índices 1-based
        xa, ya, va = pts[a - 1]
        xb, yb, vb = pts[b - 1]
        if va > 0 and vb > 0:
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

    for x, y, v in pts:
        if v > 0:
            color = (0, 0, 255) if v == 2 else (0, 165, 255)  # vermelho=visível, laranja=ocluso
            cv2.circle(img, (int(x), int(y)), 3, color, -1)

    x, y, w, h = (int(v) for v in ann["bbox"])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imwrite(out_path, img)
    print(f"[ok] overlay salvo em {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
    else:
        main(*sys.argv[1:5])
