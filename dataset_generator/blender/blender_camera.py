"""Utilitários de câmera Blender para projeção pinhole (CV convention).

Blender camera space:  X=direita, Y=cima,  –Z=frente
CV (OpenCV) space:     X=direita, Y=baixo, +Z=frente

BLENDER_AXIS_REMAP converte pontos de Blender-cam-space → CV-space.
Compatível com nao_coco_pose.projection.world_to_pixels().
"""
import numpy as np

# Matriz de remapeamento de eixos: Blender cam → CV
# x_cv =  x_ble  (direita não muda)
# y_cv = -y_ble  (cima → baixo)
# z_cv = -z_ble  (atrás → frente)
BLENDER_AXIS_REMAP = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0, -1.0],
], dtype=float)


def build_K(cam_data, width: int, height: int) -> np.ndarray:
    """Matriz intrínseca 3×3 a partir dos parâmetros fundamentais do Blender.

    Usa lens (focal em mm) e sensor_width (mm) — mesmos valores que o Blender
    usa internamente. Assume pixels quadrados.
    """
    f_px = cam_data.lens / cam_data.sensor_width * width
    cx, cy = width / 2.0, height / 2.0
    return np.array([
        [f_px,  0.0,  cx],
        [ 0.0, f_px,  cy],
        [ 0.0,  0.0, 1.0],
    ])


def cam_to_world(cam_obj) -> np.ndarray:
    """Matriz câmera→mundo 4×4 (numpy float64) do objeto Camera.

    Deve ser lida APÓS bpy.context.view_layer.update() para refletir
    a pose atual da cena.
    """
    return np.array(cam_obj.matrix_world, dtype=float)


def fov_from_K(K: np.ndarray, width: int, height: int) -> tuple:
    """Retorna (fov_h_deg, fov_v_deg) a partir de K e resolução."""
    import math
    fov_h = 2 * math.degrees(math.atan(width  / (2 * K[0, 0])))
    fov_v = 2 * math.degrees(math.atan(height / (2 * K[1, 1])))
    return fov_h, fov_v
