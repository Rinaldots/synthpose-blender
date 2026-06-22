"""Projeção 3D -> 2D (modelo pinhole) e geometria de câmera.

Tudo aqui é NumPy puro, independente do Webots, portanto testável isolado.
Esta é a etapa mais propensa a bugs do pipeline: sempre confirme o resultado
sobrepondo os pontos na imagem (scripts/visualize_sample.py).
"""
from __future__ import annotations

import numpy as np

# Converte o referencial da câmera do Webots para o padrão de visão
# computacional (x: direita, y: baixo, z: para frente).
# PRESSUPOSTO: a câmera do Webots olha ao longo de -Z (x à direita, y p/ cima).
# Se o overlay sair espelhado/de cabeça p/ baixo, ajuste os sinais aqui.
AXIS_REMAP = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])


def build_intrinsics(width: int, height: int, fov_h_rad: float) -> np.ndarray:
    """Matriz intrínseca K a partir do FOV horizontal (pixels quadrados)."""
    f = (width / 2.0) / np.tan(fov_h_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0.0, cx],
                     [0.0, f, cy],
                     [0.0, 0.0, 1.0]])


def pose_list_to_matrix(pose16) -> np.ndarray:
    """Converte os 16 valores (row-major) de Node.getPose() em 4x4 (câmera->mundo)."""
    return np.asarray(pose16, dtype=float).reshape(4, 4)


def world_to_pixels(points_world, cam_to_world, K, axis_remap=AXIS_REMAP):
    """Projeta pontos do mundo para pixels.

    Args:
        points_world: (N, 3) coordenadas no referencial do mundo.
        cam_to_world: 4x4, pose da câmera no mundo.
        K: 3x3, intrínseca.
    Returns:
        (uv, depth): uv (N, 2) em pixels; depth (N,) em metros no eixo óptico.
        depth <= 0 indica ponto atrás da câmera (deve ser descartado).
    """
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    world_to_cam = np.linalg.inv(np.asarray(cam_to_world, dtype=float))
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    cam = (world_to_cam @ hom.T).T[:, :3]        # referencial da câmera (Webots)
    cam_cv = (np.asarray(axis_remap) @ cam.T).T  # referencial CV
    depth = cam_cv[:, 2].copy()
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    proj = (np.asarray(K) @ (cam_cv / safe[:, None]).T).T
    return proj[:, :2], depth


def look_at_rotation(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotação 3x3 (câmera->mundo) que faz a câmera (olhando p/ -Z) mirar o alvo."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    z = eye - target                       # +Z aponta p/ longe do alvo (olha -Z)
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:           # up paralelo a z -> escolhe outro up
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rotation_to_axis_angle(R) -> list[float]:
    """Converte rotação 3x3 em [x, y, z, angle] (formato 'rotation' do Webots)."""
    R = np.asarray(R, dtype=float)
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return [0.0, 0.0, 1.0, 0.0]
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:                           # angle ~ pi: extrai eixo da diagonal
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        return [float(axis[0]), float(axis[1]), float(axis[2]), angle]
    axis /= n
    return [float(axis[0]), float(axis[1]), float(axis[2]), angle]
