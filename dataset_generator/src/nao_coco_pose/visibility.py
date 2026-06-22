"""Flags de visibilidade COCO, oclusão por profundidade e bounding box.

NumPy puro (a parte de profundidade depende de uma imagem que o controlador
passa). Mantém as três flags COCO:
    0 = não rotulado (fora do quadro / não resolvido)
    1 = rotulado, porém ocluso
    2 = rotulado e visível
"""
from __future__ import annotations

import numpy as np

V_ABSENT = 0
V_OCCLUDED = 1
V_VISIBLE = 2


def in_frame(uv, width: int, height: int) -> bool:
    u, v = float(uv[0]), float(uv[1])
    return 0.0 <= u < width and 0.0 <= v < height


def occluded_by_depth(uv, point_depth: float, depth_image, tol: float = 0.02) -> bool:
    """True se a cena tiver geometria à frente do keypoint no pixel (u, v).

    depth_image: (H, W) com distância por pixel (RangeFinder do Webots).
    TODO: confirme as UNIDADES/convenção do seu RangeFinder (metros vs.
    normalizado) e ajuste `tol`. Sem depth_image, assume não-ocluso.
    """
    if depth_image is None:
        return False
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = depth_image.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return True
    scene_depth = float(depth_image[v, u])
    return scene_depth < (point_depth - tol)


def visibility_flag(uv, depth, width, height, depth_image=None) -> int:
    if depth <= 0 or not in_frame(uv, width, height):
        return V_ABSENT
    if occluded_by_depth(uv, depth, depth_image):
        return V_OCCLUDED
    return V_VISIBLE


def bbox_from_keypoints(uv_list, flags, width, height, margin: float = 0.12):
    """BBox COCO [x, y, w, h] + area a partir dos keypoints rotulados.

    Aproximação: envolve os pontos rotulados + margem, recortando ao quadro.
    Para máxima precisão, derive a bbox da máscara de segmentação do Webots
    (Camera com recognitionSegmentation) — ver docs/pipeline.md.
    """
    pts = [uv for uv, f in zip(uv_list, flags) if f != V_ABSENT]
    if not pts:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    x0 = max(0.0, x0 - mw)
    y0 = max(0.0, y0 - mh)
    x1 = min(float(width), x1 + mw)
    y1 = min(float(height), y1 + mh)
    w, h = x1 - x0, y1 - y0
    return [float(x0), float(y0), float(w), float(h)], float(w * h)
