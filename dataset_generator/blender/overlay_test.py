"""Teste de projeção: renderiza 1 frame, projeta keypoints e salva overlay.

Se os círculos ficarem sobre os joints corretos na imagem, o AXIS_REMAP
e a extração da cam_to_world estão corretos — pré-condição para o dataset.

Uso:
    blender --background nao_full.blend --python overlay_test.py
Saídas:
    overlay_raw.png  — render limpo
    overlay_kp.png   — render com keypoints projetados desenhados
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
from blender_camera import build_K, cam_to_world, BLENDER_AXIS_REMAP, fov_from_K
from nao_coco_pose.projection import world_to_pixels
from nao_coco_pose.visibility import visibility_flag, V_ABSENT

URDF_PATH    = HERE.parent / "protos" / "nao_exported.urdf"
W, H         = 640, 480
RENDER_PATH  = HERE / "overlay_raw.png"
OVERLAY_PATH = HERE / "overlay_kp.png"

# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------
arm = bpy.data.objects.get("NAO_Armature")
if arm is None:
    sys.exit("[OVERLAY] NAO_Armature não encontrado")

poser = NaoPoserBlender(arm, URDF_PATH)
poser.apply_pose({
    "LShoulderPitch": -1.0,
    "RShoulderPitch":  1.2,
    "LElbowYaw":       0.5,
    "LHipPitch":      -0.4,
    "LKneePitch":      0.7,
    "LAnklePitch":    -0.35,
    "HeadYaw":         0.2,
})

# ---------------------------------------------------------------------------
# Câmera
# ---------------------------------------------------------------------------
cam_data = bpy.data.cameras.new("OverlayCam")
cam_data.lens = 50.0       # focal length em mm (sensor_width=36mm por padrão)
cam_obj  = bpy.data.objects.new("OverlayCam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

cam_pos = mathutils.Vector((0.9, 1.1, 0.55))
target  = mathutils.Vector((0.0, 0.0, 0.27))
cam_obj.location       = cam_pos
cam_obj.rotation_euler = (target - cam_pos).to_track_quat('-Z', 'Y').to_euler()

# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
ground = bpy.context.active_object
mat_g  = bpy.data.materials.new("Ground")
mat_g.use_nodes = True
mat_g.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
ground.data.materials.append(mat_g)

for name, etype, energy, loc, rot in [
    ("Sun",  'SUN',  3.0, (2, -1, 3),   (0.6, 0.2, -0.8)),
    ("Fill", 'AREA', 70.0, (-1, -1, 1.5), (0, 0, 0)),
]:
    ld  = bpy.data.lights.new(name, type=etype)
    ld.energy = energy
    lo  = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = loc
    if rot:
        lo.rotation_euler = rot

world = bpy.data.worlds.new("World")
world.use_nodes = True
bpy.context.scene.world = world
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value    = (0.55, 0.55, 0.60, 1)
bg.inputs["Strength"].default_value = 0.5

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
sc = bpy.context.scene
sc.render.engine               = 'CYCLES'
sc.cycles.samples              = 64
sc.render.resolution_x         = W
sc.render.resolution_y         = H
sc.render.filepath             = str(RENDER_PATH)
sc.render.image_settings.file_format = 'PNG'

bpy.ops.render.render(write_still=True)
print(f"[OVERLAY] Render base: {RENDER_PATH}")

# ---------------------------------------------------------------------------
# Projeção
# ---------------------------------------------------------------------------
bpy.context.view_layer.update()

K   = build_K(cam_data, W, H)
C2W = cam_to_world(cam_obj)
fov_h, fov_v = fov_from_K(K, W, H)
print(f"[OVERLAY] K  fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
print(f"[OVERLAY] FOV  h={fov_h:.1f}°  v={fov_v:.1f}°")

kps_3d = poser.get_keypoints_world()
pts    = [kps_3d.get(kp) or (0.0, 0.0, 0.0) for kp in COCO_KEYPOINTS_ORDER]
uv_list, depths = world_to_pixels(pts, C2W, K, axis_remap=BLENDER_AXIS_REMAP)

print(f"\n[OVERLAY] {'Keypoint':<20} {'u':>7} {'v':>7} {'depth':>8}")
print("[OVERLAY] " + "-" * 45)
for kp, uv, d in zip(COCO_KEYPOINTS_ORDER, uv_list, depths):
    flag = visibility_flag(uv, d, W, H)
    mark = "OK" if flag != V_ABSENT else "--"
    print(f"[OVERLAY] {kp:<20} {uv[0]:7.1f} {uv[1]:7.1f} {d:8.3f}m  {mark}")

# ---------------------------------------------------------------------------
# Desenho do overlay (numpy puro, sem OpenCV/PIL)
# Blender armazena pixels Y-up; flipamos para Y-down (= convenção UV).
# ---------------------------------------------------------------------------
_COLORS = [          # RGBA linear 0-1, um por keypoint COCO
    (1.0, 0.15, 0.15, 1),   # nose
    (1.0, 0.55, 0.00, 1),   # left_eye
    (1.0, 0.55, 0.00, 1),   # right_eye
    (1.0, 1.00, 0.00, 1),   # left_ear
    (1.0, 1.00, 0.00, 1),   # right_ear
    (0.0, 1.00, 0.00, 1),   # left_shoulder
    (0.0, 0.75, 0.00, 1),   # right_shoulder
    (0.0, 0.50, 1.00, 1),   # left_elbow
    (0.0, 0.30, 0.80, 1),   # right_elbow
    (0.0, 1.00, 1.00, 1),   # left_wrist
    (0.0, 0.70, 0.70, 1),   # right_wrist
    (1.0, 0.00, 1.00, 1),   # left_hip
    (0.7, 0.00, 0.70, 1),   # right_hip
    (1.0, 0.00, 0.40, 1),   # left_knee
    (0.7, 0.00, 0.30, 1),   # right_knee
    (0.5, 0.00, 1.00, 1),   # left_ankle
    (0.3, 0.00, 0.80, 1),   # right_ankle
]

img = bpy.data.images.load(str(RENDER_PATH), check_existing=False)
pix = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)
pix = np.flipud(pix)          # Y-up → Y-down

R = 7   # raio do círculo em pixels
for i, (uv, d) in enumerate(zip(uv_list, depths)):
    if visibility_flag(uv, d, W, H) == V_ABSENT:
        continue
    col = np.array(_COLORS[i], dtype=np.float32)
    cx_i, cy_i = int(round(float(uv[0]))), int(round(float(uv[1])))
    # círculo
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dx * dx + dy * dy <= R * R:
                px, py = cx_i + dx, cy_i + dy
                if 0 <= px < W and 0 <= py < H:
                    pix[py, px] = col
    # anel branco externo (1px)
    for dy in range(-(R+1), R + 2):
        for dx in range(-(R+1), R + 2):
            r2 = dx * dx + dy * dy
            if R * R < r2 <= (R + 1) ** 2:
                px, py = cx_i + dx, cy_i + dy
                if 0 <= px < W and 0 <= py < H:
                    pix[py, px] = (1.0, 1.0, 1.0, 1.0)

img.pixels[:] = np.flipud(pix).flatten().tolist()
img.filepath_raw = str(OVERLAY_PATH)
img.file_format  = 'PNG'
img.save()
bpy.data.images.remove(img)
print(f"\n[OVERLAY] Overlay salvo: {OVERLAY_PATH}")
