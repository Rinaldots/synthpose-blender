"""Testa NaoPoserBlender: carrega nao_rig.blend, aplica pose, imprime keypoints.

Uso:
    blender --background nao_rig.blend --python test_poser.py
"""
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from nao_poser_blender import NaoPoserBlender, COCO_KEYPOINTS_ORDER

URDF_PATH = HERE.parent / "protos" / "nao_exported.urdf"

arm = bpy.data.objects.get("NAO_Armature")
if arm is None:
    sys.exit("[ERRO] NAO_Armature não encontrado no .blend")

poser = NaoPoserBlender(arm, URDF_PATH)

# --- Pose zero (rest) ---
print("\n=== POSE ZERO ===")
poser.reset_pose()
kps = poser.get_keypoints_world()
print(f"{'Keypoint':<20} {'X':>8} {'Y':>8} {'Z':>8}")
print("-" * 50)
for kp in COCO_KEYPOINTS_ORDER:
    pos = kps[kp]
    if pos:
        print(f"{kp:<20} {pos[0]:8.4f} {pos[1]:8.4f} {pos[2]:8.4f}")
    else:
        print(f"{kp:<20}  NONE")

# --- Pose com braço levantado ---
print("\n=== POSE: OMBRO ESQUERDO LEVANTADO 90° ===")
poser.apply_pose({
    "LShoulderPitch": -1.5708,   # braço horizontal para frente
    "RShoulderPitch": -0.5,
    "LElbowYaw": 0.3,
    "LHipPitch": -0.4,
    "LKneePitch": 0.8,
    "LAnklePitch": -0.4,
})
kps = poser.get_keypoints_world()
print(f"{'Keypoint':<20} {'X':>8} {'Y':>8} {'Z':>8}")
print("-" * 50)
for kp in COCO_KEYPOINTS_ORDER:
    pos = kps[kp]
    if pos:
        print(f"{kp:<20} {pos[0]:8.4f} {pos[1]:8.4f} {pos[2]:8.4f}")
    else:
        print(f"{kp:<20}  NONE")

print("\n[OK] Teste concluído.")
