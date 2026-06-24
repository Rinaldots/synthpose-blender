"""Render de pré-visualização: T-pose, câmera frontal, 4 ângulos num grid."""
import bpy
import mathutils
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from nao_poser_blender import NaoPoserBlender

URDF_PATH  = HERE.parent / "protos" / "nao_exported.urdf"
OUTPUT_IMG = HERE / "render_preview.png"

arm = bpy.data.objects.get("NAO_Armature")
if arm is None:
    sys.exit("[ERRO] NAO_Armature não encontrado")

poser = NaoPoserBlender(arm, URDF_PATH)
poser.reset_pose()   # T-pose / rest

# Câmera frontal levemente elevada
cam_data = bpy.data.cameras.new("PreviewCam")
cam_obj  = bpy.data.objects.new("PreviewCam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

nao_center = mathutils.Vector((0.0, 0.0, 0.30))
cam_pos    = mathutils.Vector((0.0, 1.6, 0.50))
direction  = nao_center - cam_pos
cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
cam_obj.location = cam_pos
cam_data.lens = 50

# Chão
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = "Ground"
mat_g = bpy.data.materials.new("Ground")
mat_g.use_nodes = True
mat_g.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.25, 0.25, 0.25, 1)
plane.data.materials.append(mat_g)

# Luzes
sun = bpy.data.lights.new("Sun", type='SUN')
sun.energy = 3.0
sun_obj = bpy.data.objects.new("Sun", sun)
bpy.context.scene.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (0.6, 0.2, -0.5)

fill = bpy.data.lights.new("Fill", type='AREA')
fill.energy = 80.0
fill.size = 2.0
fill_obj = bpy.data.objects.new("Fill", fill)
bpy.context.scene.collection.objects.link(fill_obj)
fill_obj.location = (1.5, 1.5, 1.5)

back = bpy.data.lights.new("Back", type='AREA')
back.energy = 40.0
back_obj = bpy.data.objects.new("Back", back)
bpy.context.scene.collection.objects.link(back_obj)
back_obj.location = (-0.5, 1.0, 1.0)

world = bpy.data.worlds.new("World")
world.use_nodes = True
bpy.context.scene.world = world
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.55, 0.55, 0.60, 1)
bg.inputs["Strength"].default_value = 0.6

sc = bpy.context.scene
sc.render.engine       = 'CYCLES'
sc.cycles.samples      = 256
sc.render.resolution_x = 640
sc.render.resolution_y = 960     # portrait para ver pés e cabeça
sc.render.filepath     = str(OUTPUT_IMG)
sc.render.image_settings.file_format = 'PNG'

bpy.ops.render.render(write_still=True)
print(f"[RENDER] Salvo em {OUTPUT_IMG}")
