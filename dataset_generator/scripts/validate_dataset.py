"""Validação e QA do dataset COCO-pose.

Carrega o JSON com pycocotools (já valida a estrutura) e imprime estatísticas:
contagem de imagens/anotações e a distribuição de visibilidade por keypoint.

Uso:
    python scripts/validate_dataset.py output/annotations/person_keypoints.json
"""
from __future__ import annotations

import sys
from collections import Counter


def main(ann_path: str) -> None:
    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("pycocotools não instalado — veja requirements.txt")
        return

    coco = COCO(ann_path)  # carregar já valida o formato básico
    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds()
    cat = coco.loadCats(coco.getCatIds())[0]
    kp_names = cat["keypoints"]

    print(f"imagens:    {len(img_ids)}")
    print(f"anotações:  {len(ann_ids)}")
    print(f"keypoints:  {len(kp_names)}  (esperado: 17)")
    print(f"skeleton:   {len(cat.get('skeleton', []))} arestas")

    vis = [Counter() for _ in kp_names]
    for a in coco.loadAnns(ann_ids):
        flat = a["keypoints"]
        for i in range(len(kp_names)):
            vis[i][flat[i * 3 + 2]] += 1

    print("\nvisibilidade por keypoint (0=ausente  1=ocluso  2=visível):")
    for name, counter in zip(kp_names, vis):
        ordered = {k: counter.get(k, 0) for k in (0, 1, 2)}
        print(f"  {name:>16}: {ordered}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        main(sys.argv[1])
