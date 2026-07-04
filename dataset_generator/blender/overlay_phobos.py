"""Overlay de keypoints para o modelo NAO Phobos de alta qualidade.

Uso:
    blender --background nao_blender.blend --python overlay_phobos.py
Saídas:
    yaw_test/overlay_phobos_*.png  (render + 17 keypoints reprojetados)
"""
import sys
import math
import numpy as np
from pathlib import Path

import bpy
import mathutils

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from nao_poser_phobos import NaoPoserPhobos, hide_metadata_collections, COCO_KEYPOINTS_ORDER
from blender_camera import build_K, cam_to_world, BLENDER_AXIS_REMAP
from nao_coco_pose.projection import world_to_pixels
from nao_coco_pose.visibility import visibility_flag, V_ABSENT

W, H = 640, 480
OUT_DIR = HERE / "yaw_test"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Poses de teste (rad). Frame NAO: X=frente, Y=esq, Z=cima.
# ---------------------------------------------------------------------------
POSES = {
    "rest":       {},
    "arms_up":    {"LShoulderPitch": -1.2, "RShoulderPitch": -1.2},
    "wave":       {"RShoulderPitch": -1.4, "RShoulderRoll": -0.6, "RElbowRoll": 1.0,
                   "LShoulderRoll": 0.3},
    "sit":        {"LHipPitch": -1.2, "RHipPitch": -1.2, "LKneePitch": 1.6,
                   "RKneePitch": 1.6, "LShoulderPitch": 0.4, "RShoulderPitch": 0.4},
    "look_left":  {"HeadYaw": 0.8, "HeadPitch": 0.2},
}

_COLORS = [
    (1.0, 0.15, 0.15, 1), (1.0, 0.55, 0.0, 1), (1.0, 0.55, 0.0, 1),
    (1.0, 1.0, 0.0, 1), (1.0, 1.0, 0.0, 1),
    (0.0, 1.0, 0.0, 1), (0.0, 0.75, 0.0, 1),
    (0.0, 0.5, 1.0, 1), (0.0, 0.3, 0.8, 1),
    (0.0, 1.0, 1.0, 1), (0.0, 0.7, 0.7, 1),
    (1.0, 0.0, 1.0, 1), (0.7, 0.0, 0.7, 1),
    (1.0, 0.0, 0.4, 1), (0.7, 0.0, 0.3, 1),
    (0.5, 0.0, 1.0, 1), (0.3, 0.0, 0.8, 1),
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
        for dy in range(-(R + 1), R + 2):
            for dx in range(-(R + 1), R + 2):
                r2 = dx * dx + dy * dy
                if R * R < r2 <= (R + 1) ** 2:
                    px, py = cx_i + dx, cy_i + dy
                    if 0 <= px < W and 0 <= py < H:
                        pix[py, px] = (1.0, 1.0, 1.0, 1.0)
    return pix


def setup_scene():
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
    bg.inputs["Color"].default_value = (0.55, 0.55, 0.6, 1)
    bg.inputs["Strength"].default_value = 0.5

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 48
    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.render.image_settings.file_format = "PNG"


# ---------------------------------------------------------------------------
hide_metadata_collections()
poser = NaoPoserPhobos()
if not poser._joint_obj:
    sys.exit("[PHOBOS] nenhuma junta encontrada — rode com nao_blender.blend")
setup_scene()


def _ground_robot():
    """Levanta o NAO (via base_link) para que o ponto mais baixo fique em z=0."""
    poser.apply_pose({})
    bpy.context.view_layer.update()
    base = bpy.data.objects.get("base_link")
    min_z = 1e9
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.data.objects:
        if o.type != "MESH" or o.hide_render:
            continue
        ev = o.evaluated_get(dg)
        for corner in ev.bound_box:
            wz = (o.matrix_world @ mathutils.Vector(corner))[2]
            min_z = min(min_z, wz)
    base.location.z += -min_z
    bpy.context.view_layer.update()
    print(f"[PHOBOS] base_link levantado em {-min_z:.3f} m (min_z era {min_z:.3f})")

_ground_robot()

cam_data = bpy.data.cameras.new("PhobosCam")
cam_data.lens = 50.0
cam_obj = bpy.data.objects.new("PhobosCam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
sc = bpy.context.scene

# Câmera fixa à frente do NAO (frame NAO: +X = frente). Mira o tronco (~z=0.30).
CAM_POS = (2.2, 0.0, 0.55)   # à frente do NAO, na altura do tronco
AIM_PT = (0.0, 0.0, 0.40)

done = 0
for pose_name, angles in POSES.items():
    poser.apply_pose(angles)

    cam_obj.location = mathutils.Vector(CAM_POS)
    cam_obj.rotation_euler = (
        mathutils.Vector(AIM_PT) - mathutils.Vector(CAM_POS)
    ).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    raw_path = OUT_DIR / f"raw_phobos_{pose_name}.png"
    out_path = OUT_DIR / f"overlay_phobos_{pose_name}.png"

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
    print(f"[PHOBOS] {done}/{len(POSES)} salvo: {out_path.name}")

print(f"\n[PHOBOS] Concluído. {done} overlays em {OUT_DIR}")
