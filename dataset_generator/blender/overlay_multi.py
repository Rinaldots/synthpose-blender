"""Gera overlays de keypoints em múltiplas poses e ângulos de câmera.

Uso:
    blender --background nao_full.blend --python overlay_multi.py
Saídas:
    overlay_NOME.png  para cada combinação pose × câmera definida abaixo.
"""
import sys
import numpy as np
from pathlib import Path

import bpy
import mathutils

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from nao_poser_blender import NaoPoserBlender, COCO_KEYPOINTS_ORDER
from blender_camera import build_K, cam_to_world, BLENDER_AXIS_REMAP
from nao_coco_pose.projection import world_to_pixels
from nao_coco_pose.visibility import visibility_flag, V_ABSENT

URDF_PATH = HERE.parent / "protos" / "nao_exported.urdf"
W, H = 640, 480
OUT_DIR = HERE / "yaw_test"
OUT_DIR.mkdir(exist_ok=True)

import math as _math

def _face_cam(yaw_rad, dist=2.0, height=0.55, aim_z=0.45):
    """Câmera posicionada na direção para a qual a face aponta após HeadYaw."""
    # NAO X (frente) em Blender = (-sin(yaw), cos(yaw), 0)
    bx = -_math.sin(yaw_rad)
    by =  _math.cos(yaw_rad)
    return (bx * dist, by * dist, height), (0.0, 0.0, aim_z)

# ---------------------------------------------------------------------------
# Poses × câmera que segue a face (1 câmera por pose = 4 imagens)
# ---------------------------------------------------------------------------
POSES_CAMERAS = {
    "yaw_neg_080": ({"HeadYaw": -0.8}, _face_cam(-0.8)),
    "yaw_neg_040": ({"HeadYaw": -0.4}, _face_cam(-0.4)),
    "yaw_pos_040": ({"HeadYaw":  0.4}, _face_cam( 0.4)),
    "yaw_pos_080": ({"HeadYaw":  0.8}, _face_cam( 0.8)),
}
POSES   = {k: v[0] for k, v in POSES_CAMERAS.items()}
CAMERAS = {k: v[1] for k, v in POSES_CAMERAS.items()}

_COLORS = [
    (1.0, 0.15, 0.15, 1),  # nose
    (1.0, 0.55, 0.00, 1),  # left_eye
    (1.0, 0.55, 0.00, 1),  # right_eye
    (1.0, 1.00, 0.00, 1),  # left_ear
    (1.0, 1.00, 0.00, 1),  # right_ear
    (0.0, 1.00, 0.00, 1),  # left_shoulder
    (0.0, 0.75, 0.00, 1),  # right_shoulder
    (0.0, 0.50, 1.00, 1),  # left_elbow
    (0.0, 0.30, 0.80, 1),  # right_elbow
    (0.0, 1.00, 1.00, 1),  # left_wrist
    (0.0, 0.70, 0.70, 1),  # right_wrist
    (1.0, 0.00, 1.00, 1),  # left_hip
    (0.7, 0.00, 0.70, 1),  # right_hip
    (1.0, 0.00, 0.40, 1),  # left_knee
    (0.7, 0.00, 0.30, 1),  # right_knee
    (0.5, 0.00, 1.00, 1),  # left_ankle
    (0.3, 0.00, 0.80, 1),  # right_ankle
]


def draw_keypoints(pix, uv_list, depths):
    R = 6
    for i, (uv, d) in enumerate(zip(uv_list, depths)):
        if visibility_flag(uv, d, W, H) == V_ABSENT:
            continue
        col = np.array(_COLORS[i], dtype=np.float32)
        cx_i, cy_i = int(round(float(uv[0]))), int(round(float(uv[1])))
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dx * dx + dy * dy <= R * R:
                    px, py = cx_i + dx, cy_i + dy
                    if 0 <= px < W and 0 <= py < H:
                        pix[py, px] = col
        # anel branco
        for dy in range(-(R + 1), R + 2):
            for dx in range(-(R + 1), R + 2):
                r2 = dx * dx + dy * dy
                if R * R < r2 <= (R + 1) ** 2:
                    px, py = cx_i + dx, cy_i + dy
                    if 0 <= px < W and 0 <= py < H:
                        pix[py, px] = (1.0, 1.0, 1.0, 1.0)
    return pix


def setup_scene():
    # Ambiente mínimo (reutilizado entre renders)
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    ground = bpy.context.active_object
    mat_g = bpy.data.materials.new("Ground")
    mat_g.use_nodes = True
    mat_g.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
    ground.data.materials.append(mat_g)

    for name, etype, energy, loc, rot in [
        ("Sun",  "SUN",  3.0, (2, -1, 3),    (0.6, 0.2, -0.8)),
        ("Fill", "AREA", 70.0, (-1, -1, 1.5), (0, 0, 0)),
    ]:
        ld = bpy.data.lights.new(name, type=etype)
        ld.energy = energy
        lo = bpy.data.objects.new(name, ld)
        bpy.context.scene.collection.objects.link(lo)
        lo.location = loc
        if rot:
            lo.rotation_euler = rot

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bpy.context.scene.world = world
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.55, 0.60, 1)
    bg.inputs["Strength"].default_value = 0.5

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 64
    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.render.image_settings.file_format = "PNG"


# ---------------------------------------------------------------------------
arm = bpy.data.objects.get("NAO_Armature")
if arm is None:
    sys.exit("[MULTI] NAO_Armature não encontrado")

poser = NaoPoserBlender(arm, URDF_PATH)
setup_scene()

cam_data = bpy.data.cameras.new("MultiCam")
cam_data.lens = 50.0
cam_obj = bpy.data.objects.new("MultiCam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

sc = bpy.context.scene

total = len(POSES) * len(CAMERAS)
done = 0

for pose_name, angles in POSES.items():
    poser.reset_pose()
    poser.apply_pose(angles)
    bpy.context.view_layer.update()

    # câmera emparelhada com a pose
    cam_pos, aim_pt = CAMERAS[pose_name]
    cam_obj.location = mathutils.Vector(cam_pos)
    cam_obj.rotation_euler = (
        mathutils.Vector(aim_pt) - mathutils.Vector(cam_pos)
    ).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    if True:   # mantém indentação sem loop interno
        tag = f"{pose_name}"
        raw_path = OUT_DIR / f"raw_{tag}.png"
        out_path = OUT_DIR / f"overlay_{tag}.png"

        sc.render.filepath = str(raw_path)
        bpy.ops.render.render(write_still=True)

        K = build_K(cam_data, W, H)
        C2W = cam_to_world(cam_obj)
        kps_3d = poser.get_keypoints_world()
        pts = [kps_3d.get(kp) or (0.0, 0.0, 0.0) for kp in COCO_KEYPOINTS_ORDER]
        uv_list, depths = world_to_pixels(pts, C2W, K, axis_remap=BLENDER_AXIS_REMAP)

        img = bpy.data.images.load(str(raw_path), check_existing=False)
        pix = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)
        pix = np.flipud(pix)
        pix = draw_keypoints(pix, uv_list, depths)
        img.pixels[:] = np.flipud(pix).flatten().tolist()
        img.filepath_raw = str(out_path)
        img.file_format = "PNG"
        img.save()
        bpy.data.images.remove(img)

        done += 1
        print(f"[MULTI] {done}/{total} salvo: {out_path.name}")

print(f"\n[MULTI] Concluído. {total} overlays em {OUT_DIR}")
