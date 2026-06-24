"""Render de teste: abre nao_full.blend, posa o NAO e salva render.png.

Uso:
    blender --background nao_full.blend --python render_test.py
"""
import sys
from pathlib import Path
import bpy
import mathutils

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from nao_poser_blender import NaoPoserBlender

URDF_PATH  = HERE.parent / "protos" / "nao_exported.urdf"
OUTPUT_IMG = HERE / "render_test.png"

# --- Armature ---
arm = bpy.data.objects.get("NAO_Armature")
if arm is None:
    sys.exit("[ERRO] NAO_Armature não encontrado")

poser = NaoPoserBlender(arm, URDF_PATH)
poser.apply_pose({
    "LShoulderPitch":  1.5,   # braço abaixado à esquerda
    "RShoulderPitch":  1.5,   # braço abaixado à direita
    "LShoulderRoll":   0.3,
    "RShoulderRoll":  -0.3,
    "LHipPitch":      -0.3,
    "LKneePitch":      0.6,
    "LAnklePitch":    -0.3,
})

# --- Câmera ---
cam_data = bpy.data.cameras.new("TestCam")
cam_data.lens = 35
cam_obj  = bpy.data.objects.new("TestCam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# NAO em world: pés ≈ Z=0.046, cabeça ≈ Z=0.48 → centro ≈ Z=0.28
nao_center = mathutils.Vector((0.0, 0.0, 0.26))
cam_pos    = mathutils.Vector((0.55, -0.55, 0.5))
direction  = nao_center - cam_pos
rot_quat   = direction.to_track_quat('-Z', 'Y')
cam_data.lens = 35
cam_obj.location = cam_pos
cam_obj.rotation_euler = rot_quat.to_euler()

# --- Fundo cinza (plano de chão) ---
bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0.0))
plane = bpy.context.active_object
plane.name = "Ground"
mat = bpy.data.materials.new("GroundMat")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
plane.data.materials.append(mat)

# --- Luz ---
light_data = bpy.data.lights.new("Sun", type='SUN')
light_data.energy = 4.0
light_obj  = bpy.data.objects.new("Sun", light_data)
bpy.context.scene.collection.objects.link(light_obj)
light_obj.location = (2, -1, 3)
light_obj.rotation_euler = (0.5, 0.3, -0.4)

# Luz de preenchimento
fill_data = bpy.data.lights.new("Fill", type='AREA')
fill_data.energy = 50.0
fill_obj  = bpy.data.objects.new("Fill", fill_data)
bpy.context.scene.collection.objects.link(fill_obj)
fill_obj.location = (-1, 1, 1.5)

# Fundo cinza claro (mundo)
world = bpy.data.worlds.new("World")
world.use_nodes = True
bpy.context.scene.world = world
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.6, 0.6, 0.6, 1)
bg.inputs["Strength"].default_value = 0.5

# --- Render config ---
sc = bpy.context.scene
sc.render.engine          = 'CYCLES'
sc.cycles.samples         = 128
sc.render.resolution_x    = 640
sc.render.resolution_y    = 640
sc.render.filepath        = str(OUTPUT_IMG)
sc.render.image_settings.file_format = 'PNG'

bpy.ops.render.render(write_still=True)
print(f"[RENDER] Salvo em {OUTPUT_IMG}")
