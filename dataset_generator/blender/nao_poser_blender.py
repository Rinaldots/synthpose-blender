"""Equivalente Blender de nao_landmarks.py: posa o NAO e lê keypoints.

Este módulo é importado DENTRO do Blender (bpy disponível).
Não importa bpy no topo para permitir testes de lógica pura fora do Blender.

Convenção de eixos:
  NAO/Webots: X=frente, Y=esquerda, Z=cima
  Blender:    X=direita, Y=frente, Z=cima
  Transformação: x_ble = -y_nao, y_ble = x_nao, z_ble = z_nao  (rotação -90° em Z)

  O armature foi criado com essa convenção em create_nao_armature.py.
  Não alterar sem atualizar os dois arquivos.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

# Offsets faciais no referencial da cabeça (HeadPitch bone), em metros.
# Mesmos valores de keypoints.py — validar pelo overlay.
# Convenção: X=frente, Y=esquerda, Z=cima (NAO frame), convertidos abaixo.
_HEAD_OFFSETS_NAO = {
    # Calibrados pelo depsgraph: offset_z=0.055 minimiza divergência com o mesh real
    # em diferentes ângulos de HeadPitch (face Y_max=0.058, bone_Z=0.4605).
    # Referencial NAO: X=frente, Y=esquerda, Z=cima.
    "nose":       (0.058,  0.000,  0.055),
    "left_eye":   (0.053,  0.030,  0.070),
    "right_eye":  (0.053, -0.030,  0.070),
    "left_ear":   (0.010,  0.066,  0.062),
    "right_ear":  (0.010, -0.066,  0.062),
}

# Mapeamento keypoint COCO → joint do armature
COCO_TO_JOINT = {
    "nose":            "HeadPitch",
    "left_eye":        "HeadPitch",
    "right_eye":       "HeadPitch",
    "left_ear":        "HeadPitch",
    "right_ear":       "HeadPitch",
    "left_shoulder":   "LShoulderPitch",
    "right_shoulder":  "RShoulderPitch",
    "left_elbow":      "LElbowYaw",
    "right_elbow":     "RElbowYaw",
    "left_wrist":      "LWristYaw",
    "right_wrist":     "RWristYaw",
    "left_hip":        "LHipPitch",
    "right_hip":       "RHipPitch",
    "left_knee":       "LKneePitch",
    "right_knee":      "RKneePitch",
    "left_ankle":      "LAnklePitch",
    "right_ankle":     "RAnklePitch",
}

COCO_KEYPOINTS_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def _nao_vec_to_blender(v):
    """Converte vetor de direção NAO → Blender (sem translação)."""
    return (-v[1], v[0], v[2])


def _parse_urdf_axes(urdf_path: Path) -> dict:
    """Retorna {joint_name: (ax, ay, az)} em espaço NAO."""
    root = ET.parse(urdf_path).getroot()
    axes = {}
    for j in root.findall("joint"):
        name = j.get("name")
        ax = j.find("axis")
        if ax is not None:
            xyz = tuple(float(v) for v in ax.get("xyz", "0 0 1").split())
        else:
            xyz = (0.0, 0.0, 1.0)
        axes[name] = xyz
    return axes


def _bone_chain_order(arm_data) -> list[str]:
    """Retorna bones na ordem parent-before-child (DFS)."""
    ordered = []
    visited = set()

    def visit(bone):
        if bone.name in visited:
            return
        if bone.parent:
            visit(bone.parent)
        visited.add(bone.name)
        ordered.append(bone.name)

    for bone in arm_data.bones:
        visit(bone)
    return ordered


class NaoPoserBlender:
    """Posa o armature do NAO e lê posições 3D dos keypoints COCO.

    Uso:
        import bpy
        arm = bpy.data.objects['NAO_Armature']
        poser = NaoPoserBlender(arm, urdf_path)
        poser.apply_pose({'LShoulderPitch': 0.5, 'LElbowYaw': -0.3})
        kps = poser.get_keypoints_world()
        # kps['left_shoulder'] → (x, y, z) em espaço mundo Blender
    """

    def __init__(self, arm_obj, urdf_path: Path):
        self.arm = arm_obj
        self._nao_axes = _parse_urdf_axes(urdf_path)
        self._chain_order = _bone_chain_order(arm_obj.data)
        self._last_angles: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Posing
    # ------------------------------------------------------------------

    def reset_pose(self) -> None:
        import bpy
        from mathutils import Quaternion
        for pb in self.arm.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion()
        bpy.context.view_layer.update()

    def apply_pose(self, joint_angles: dict[str, float]) -> None:
        """Aplica ângulos de junta (rad) ao armature.

        Aplica na ordem parent→child para que as matrizes acumuladas
        estejam corretas ao computar o espaço local de cada bone.
        """
        import bpy
        from mathutils import Quaternion, Vector

        self._last_angles = dict(joint_angles)
        self.reset_pose()

        arm_inv3 = self.arm.matrix_world.to_3x3().inverted()

        for bone_name in self._chain_order:
            angle = joint_angles.get(bone_name)
            if angle is None or abs(angle) < 1e-9:
                continue
            if bone_name not in self.arm.pose.bones:
                continue

            nao_axis = self._nao_axes.get(bone_name, (0.0, 0.0, 1.0))
            ble_axis = Vector(_nao_vec_to_blender(nao_axis))

            pb = self.arm.pose.bones[bone_name]
            pb.rotation_mode = "QUATERNION"

            # Atualiza cena para que parent.matrix reflita poses já aplicadas
            bpy.context.view_layer.update()

            if pb.parent:
                parent_world = self.arm.matrix_world @ pb.parent.matrix
                local_axis = parent_world.to_3x3().inverted() @ ble_axis
            else:
                local_axis = arm_inv3 @ ble_axis

            pb.rotation_quaternion = Quaternion(local_axis.normalized(), angle)

        bpy.context.view_layer.update()

    # ------------------------------------------------------------------
    # Keypoint reading
    # ------------------------------------------------------------------

    def _bone_head_world(self, bone_name: str):
        """Posição world do HEAD do bone após pose."""
        import bpy
        from mathutils import Vector
        bpy.context.view_layer.update()
        pb = self.arm.pose.bones[bone_name]
        return self.arm.matrix_world @ Vector(pb.head)

    def _head_offset_world(self, off_nao: tuple) -> "Vector":
        """Rotaciona um offset NAO-frame para world space usando a matriz do bone.

        Em vez de reconstruir o frame da cabeça manualmente com ângulos, pega
        a rotação acumulada do bone HeadPitch (FK do Blender) e a aplica ao
        offset. Correto para qualquer combinação de HeadYaw + HeadPitch.

        off_nao: (x, y, z) em frame NAO (X=frente, Y=esq, Z=cima)
        """
        from mathutils import Vector

        pb = self.arm.pose.bones["HeadPitch"]
        rb = self.arm.data.bones["HeadPitch"]

        # Rotação do bone em rest e em pose, ambas em world space
        R_rest = (self.arm.matrix_world @ rb.matrix_local).to_3x3()
        R_pose = (self.arm.matrix_world @ pb.matrix).to_3x3()
        # Delta = rotação aplicada pela pose sobre o rest
        R_delta = R_pose @ R_rest.inverted()

        # Offset NAO → Blender world em rest (NAO_X→+Y, NAO_Y→-X, NAO_Z→+Z)
        off_rest = Vector((-off_nao[1], off_nao[0], off_nao[2]))
        return R_delta @ off_rest

    def get_keypoints_world(self) -> dict[str, tuple | None]:
        """{nome_coco: (x, y, z)} em espaço mundo Blender. None se falhar."""
        from mathutils import Vector

        out = {}

        for kp in COCO_KEYPOINTS_ORDER:
            joint = COCO_TO_JOINT[kp]
            try:
                if kp in _HEAD_OFFSETS_NAO:
                    head_pos = Vector(self._bone_head_world("HeadPitch"))
                    off_world = self._head_offset_world(_HEAD_OFFSETS_NAO[kp])
                    out[kp] = tuple(head_pos + off_world)
                else:
                    pos = self._bone_head_world(joint)
                    out[kp] = tuple(pos)
            except Exception as exc:
                print(f"[NaoPoser] erro keypoint '{kp}': {exc}")
                out[kp] = None

        return out
