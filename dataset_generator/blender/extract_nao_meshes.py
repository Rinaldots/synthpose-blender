"""Extrai malhas do Nao.proto e cria mesh objects no Blender parenteados ao armature.

Uso:
    blender --background nao_rig.blend --python extract_nao_meshes.py

Salva nao_full.blend com armature + meshes do NAO (cores, smooth, subdiv).

O parser captura tanto blocos 'coord DEF X Coordinate { ... }' quanto
reutilizações 'coord USE X' com coordIndex diferentes — imprescindível para
LEDs da cabeça, bumpers dos pés, coberturas de junta dos braços/quadril etc.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector

HERE = Path(__file__).parent
PROTO_PATH  = HERE.parent / "protos" / "Nao.proto"
WRIST_R     = HERE.parent / "protos" / "NaoRightWristH25Realistic.proto"
WRIST_L     = HERE.parent / "protos" / "NaoLeftWristH25Realistic.proto"
OUTPUT_PATH = HERE / "nao_full.blend"

NAO_Z_OFFSET = 0.334

# Cores base (PBRAppearance)
_RED   = (0.88, 0.01, 0.14)
_WHITE = (1.00, 1.00, 1.00)
_GRAY  = (0.42, 0.42, 0.42)
_LGRAY = (0.749, 0.749, 0.749)
_BLACK = (0.00, 0.00, 0.00)
NAO_TEMPLATE_COLOR = _RED        # resolve %<= color.r/g/b >%

# Translação de cada Solid no body frame NAO (pose zero, metros, espaço NAO)
_WRIST_R  = (0.16095, -0.113,  0.100)
_WRIST_L  = (0.16095,  0.113,  0.100)
_ELBOW_R  = (0.105,   -0.113,  0.100)
_ELBOW_L  = (0.105,    0.113,  0.100)
SOLID_BODY_OFFSET: dict[str, tuple] = {
    "base_link":        (0.000,  0.000,  0.0000),
    "HeadYaw":          (0.000,  0.000,  0.1265),
    "RShoulderPitch":   (0.000, -0.098,  0.1000),
    "LShoulderPitch":   (0.000,  0.098,  0.1000),
    "RElbowYaw":        _ELBOW_R,
    "LElbowYaw":        _ELBOW_L,
    "RElbowRoll":       _ELBOW_R,
    "LElbowRoll":       _ELBOW_L,
    "RHipYawPitch":     (0.000, -0.050, -0.0850),
    "RHipRoll":         (0.000, -0.050, -0.0850),
    "RHipPitch":        (0.000, -0.050, -0.0850),
    "RKneePitch":       (0.000, -0.050, -0.1850),
    "RAnklePitch":      (0.000, -0.050, -0.2879),
    "RAnkleRoll":       (0.000, -0.050, -0.2879),
    "LHipYawPitch":     (0.000,  0.050, -0.0850),
    "LHipRoll":         (0.000,  0.050, -0.0850),
    "LHipPitch":        (0.000,  0.050, -0.0850),
    "LKneePitch":       (0.000,  0.050, -0.1850),
    "LAnklePitch":      (0.000,  0.050, -0.2879),
    "LAnkleRoll":       (0.000,  0.050, -0.2879),
    "RWristYaw_PROTO":  _WRIST_R,
    "LWristYaw_PROTO":  _WRIST_L,
    # Antebraço: no referencial do cotovelo (RElbowRoll endPoint = raiz do wrist PROTO)
    "RForeArm_PROTO":   _ELBOW_R,
    "LForeArm_PROTO":   _ELBOW_L,
    **{f"RPhalanx{i}": _WRIST_R for i in range(1, 9)},
    **{f"LPhalanx{i}": _WRIST_L for i in range(1, 9)},
}

SOLID_TO_BONE: dict[str, str] = {
    "base_link":        "base_link",
    "HeadYaw":          "HeadPitch",
    "RShoulderPitch":   "RShoulderPitch",
    "LShoulderPitch":   "LShoulderPitch",
    "RElbowYaw":        "RElbowYaw",
    "LElbowYaw":        "LElbowYaw",
    "RElbowRoll":       "RWristYaw",
    "LElbowRoll":       "LWristYaw",
    "RHipYawPitch":     "RHipPitch",
    "RHipRoll":         "RHipPitch",
    "RHipPitch":        "RHipPitch",
    "RKneePitch":       "RKneePitch",
    "RAnklePitch":      "RAnklePitch",
    "RAnkleRoll":       "RAnklePitch",
    "LHipYawPitch":     "LHipPitch",
    "LHipRoll":         "LHipPitch",
    "LHipPitch":        "LHipPitch",
    "LKneePitch":       "LKneePitch",
    "LAnklePitch":      "LAnklePitch",
    "LAnkleRoll":       "LAnklePitch",
    "RWristYaw_PROTO":  "RWristYaw",
    "LWristYaw_PROTO":  "LWristYaw",
    "RForeArm_PROTO":   "RElbowYaw",
    "LForeArm_PROTO":   "LElbowYaw",
    **{f"RPhalanx{i}": "RWristYaw" for i in range(1, 9)},
    **{f"LPhalanx{i}": "LWristYaw" for i in range(1, 9)},
}

# Coordenadas que precisam de solid override (nome_coord → solid_tag)
# Necessário quando o posicionamento no texto não reflete a hierarquia real.
# coord_RForeArm / coord_LForeArm ficam no final do wrist PROTO mas estão
# no referencial do cotovelo (raiz do PROTO = endPoint de RElbowRoll).
COORD_SOLID_OVERRIDE: dict[str, str] = {
    "coord_RForeArm": "RForeArm_PROTO",
    "coord_LForeArm": "LForeArm_PROTO",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TEMPLATE    = re.compile(r'%<.*?>%', re.DOTALL)
_DEF_JOINT   = re.compile(r'\bDEF\s+(\w+)\s+Hinge2?Joint\b')
_COORD_DEF   = re.compile(r'\bcoord\s+DEF\s+(\w+)\s+Coordinate\s*\{')
_COORD_USE   = re.compile(r'\bcoord\s+USE\s+(\w+)\b')
_POINT_ARRAY = re.compile(r'\bpoint\s*\[([^\]]*)\]', re.DOTALL)
_INDEX_ARRAY = re.compile(r'\bcoordIndex\s*\[([^\]]*)\]', re.DOTALL)
_NEXT_GEOM   = re.compile(r'\bgeometry\b')

_APP_USE  = re.compile(r'\bappearance\s+USE\s+(\w+)')
_APP_DEF  = re.compile(r'\bappearance\s+DEF\s+(\w+)\s+PBRAppearance\b')
_APP_INL  = re.compile(r'\bappearance\s+PBRAppearance\b')
_BCOLOR   = re.compile(r'\bbaseColor\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)')
_BCOLOR_T = re.compile(r'\bbaseColor\s+%<')


def _strip_templates(text: str) -> str:
    return _TEMPLATE.sub(' ', text)


def _parse_float_array(s: str) -> list:
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)]


def _parse_int_array(s: str) -> list:
    return [int(x) for x in re.findall(r'-?\d+', s)]


def _parse_faces(idx_str: str) -> list:
    raw = _parse_int_array(idx_str)
    faces, face = [], []
    for i in raw:
        if i == -1:
            if len(face) >= 3:
                faces.append(tuple(face))
            face = []
        else:
            face.append(i)
    if len(face) >= 3:
        faces.append(tuple(face))
    return faces


def _find_block_end(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _build_app_map(text: str, template_color: tuple) -> dict:
    """Mapa {def_name: (r,g,b)} — deve ser chamado ANTES do strip de templates."""
    app_map = {}
    for m in _APP_DEF.finditer(text):
        name = m.group(1)
        rest = text[m.end(): m.end() + 500]
        mc = _BCOLOR.search(rest)
        if mc:
            app_map[name] = (float(mc.group(1)), float(mc.group(2)), float(mc.group(3)))
        elif _BCOLOR_T.search(rest):
            app_map[name] = template_color
    return app_map


def _color_at(text: str, pos: int, app_map: dict, template_color: tuple) -> tuple:
    start = max(0, pos - 700)
    snip  = text[start:pos]
    candidates = []
    for m in _APP_USE.finditer(snip):
        candidates.append((m.start(), 'use', m.group(1)))
    for m in _APP_DEF.finditer(snip):
        candidates.append((m.start(), 'def', m.group(1)))
    for m in _APP_INL.finditer(snip):
        candidates.append((m.start(), 'inl', None))
    if not candidates:
        return _WHITE
    last_pos, kind, name = max(candidates, key=lambda x: x[0])
    after = snip[last_pos:]
    if kind == 'use':
        return app_map.get(name, _WHITE)
    mc = _BCOLOR.search(after)
    if mc:
        col = (float(mc.group(1)), float(mc.group(2)), float(mc.group(3)))
        if kind == 'def':
            app_map[name] = col
        return col
    if _BCOLOR_T.search(after):
        if kind == 'def':
            app_map[name] = template_color
        return template_color
    # Fallback: app_map pré-construído no texto original (cobre template vars já strippados)
    if kind == 'def' and name in app_map:
        return app_map[name]
    if kind == 'inl':
        return template_color   # inline sem baseColor explícito → cor de template
    return _WHITE


def extract_meshes_from_text(
    text: str,
    proto_tag: str = "",
    template_color: tuple = NAO_TEMPLATE_COLOR,
) -> dict:
    """
    Retorna {solid_name: [(verts, faces, color), ...]}

    Captura tanto blocos DEF (Coordinate { point [...] }) quanto
    blocos USE (coord USE X + coordIndex [...]) com aparências distintas.
    """
    # Constrói mapa de aparências ANTES do strip (para capturar template vars)
    app_map = _build_app_map(text, template_color)
    text    = _strip_templates(text)

    joints_pos = [(m.start(), m.group(1)) for m in _DEF_JOINT.finditer(text)]

    def solid_at(pos: int) -> str:
        name = proto_tag if proto_tag else "base_link"
        for jpos, jname in joints_pos:
            if jpos < pos:
                name = jname
            else:
                break
        return name

    # --- Passo 1: cache de coordenadas DEF {nome: verts} ---
    coord_cache: dict[str, list] = {}
    for m in _COORD_DEF.finditer(text):
        cname = m.group(1)
        bstart = text.index('{', m.end() - 1)
        bend   = _find_block_end(text, bstart)
        pm = _POINT_ARRAY.search(text[bstart:bend + 1])
        if pm is None:
            continue
        raw = _parse_float_array(pm.group(1))
        if len(raw) % 3 == 0:
            coord_cache[cname] = [(raw[i], raw[i+1], raw[i+2])
                                  for i in range(0, len(raw), 3)]

    meshes: dict[str, list] = defaultdict(list)

    # --- Passo 2: blocos DEF (coord + coordIndex logo após) ---
    for m in _COORD_DEF.finditer(text):
        cname = m.group(1)
        verts = coord_cache.get(cname)
        if not verts:
            continue
        bstart = text.index('{', m.end() - 1)
        bend   = _find_block_end(text, bstart)
        im = _INDEX_ARRAY.search(text[bend:])
        if im is None:
            continue
        faces = _parse_faces(im.group(1))
        if not faces:
            continue
        color = _color_at(text, m.start(), app_map, template_color)
        solid = COORD_SOLID_OVERRIDE.get(cname) or solid_at(m.start())
        meshes[solid].append((verts, faces, color))

    # --- Passo 3: blocos USE (coord USE X + coordIndex até o próximo geometry) ---
    for m in _COORD_USE.finditer(text):
        cname = m.group(1)
        verts = coord_cache.get(cname)
        if not verts:
            continue
        after = text[m.end():]
        # coordIndex deve estar dentro do mesmo IndexedFaceSet,
        # ou seja, antes do próximo nó 'geometry'
        ng    = _NEXT_GEOM.search(after)
        limit = ng.start() if ng else len(after)
        im    = _INDEX_ARRAY.search(after[:limit])
        if im is None:
            continue
        faces = _parse_faces(im.group(1))
        if not faces:
            continue
        color = _color_at(text, m.start(), app_map, template_color)
        solid = solid_at(m.start())
        meshes[solid].append((verts, faces, color))

    return meshes


def nao_to_blender_v(v) -> Vector:
    return Vector((-v[1], v[0], v[2]))


# ---------------------------------------------------------------------------
# Material cache
# ---------------------------------------------------------------------------

_mat_cache: dict[tuple, bpy.types.Material] = {}


def _get_material(color: tuple) -> bpy.types.Material:
    key = tuple(round(c, 4) for c in color)
    if key in _mat_cache:
        return _mat_cache[key]
    r, g, b = color
    name = f"NAO_{r:.3f}_{g:.3f}_{b:.3f}"
    mat  = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        bsdf.inputs["Roughness"].default_value  = 0.3
        bsdf.inputs["Metallic"].default_value   = 0.0
    _mat_cache[key] = mat
    return mat


# ---------------------------------------------------------------------------
# Blender mesh creation
# ---------------------------------------------------------------------------

def create_mesh_object(name: str, verts_body_nao: list, faces: list) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)

    bm = bmesh.new()
    bv_list = []
    for v in verts_body_nao:
        bv = nao_to_blender_v(v)
        bv_list.append(bm.verts.new((bv.x, bv.y, bv.z + NAO_Z_OFFSET)))
    bm.verts.ensure_lookup_table()

    for face in faces:
        try:
            bm.faces.new([bv_list[i] for i in face])
        except Exception:
            pass

    bm.to_mesh(me)
    bm.free()

    for poly in me.polygons:
        poly.use_smooth = True
    me.update()
    return ob


def parent_to_armature(ob: bpy.types.Object, arm_obj: bpy.types.Object,
                        bone_name: str) -> None:
    if bone_name != "base_link":
        vg = ob.vertex_groups.new(name=bone_name)
        vg.add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')

    ob.parent = arm_obj
    ob.parent_type = 'OBJECT'
    ob.matrix_parent_inverse = arm_obj.matrix_world.inverted()

    mod_arm = ob.modifiers.new("Armature", 'ARMATURE')
    mod_arm.object = arm_obj
    mod_arm.use_vertex_groups = True

    mod_sub = ob.modifiers.new("SubDiv", 'SUBSURF')
    mod_sub.subdivision_type = 'CATMULL_CLARK'
    mod_sub.levels        = 1
    mod_sub.render_levels = 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    arm_obj = bpy.data.objects.get("NAO_Armature")
    if arm_obj is None:
        sys.exit("[ERRO] NAO_Armature não encontrado — rode create_nao_armature.py primeiro.")

    print(f"[MESH] Lendo PROTOs ...")
    proto_text   = PROTO_PATH.read_text(encoding="utf-8")
    wrist_r_text = WRIST_R.read_text(encoding="utf-8")
    wrist_l_text = WRIST_L.read_text(encoding="utf-8")

    all_meshes: dict[str, list] = defaultdict(list)
    for solid, ml in extract_meshes_from_text(proto_text).items():
        all_meshes[solid].extend(ml)
    for solid, ml in extract_meshes_from_text(wrist_r_text, "RWristYaw_PROTO").items():
        all_meshes[solid].extend(ml)
    for solid, ml in extract_meshes_from_text(wrist_l_text, "LWristYaw_PROTO").items():
        all_meshes[solid].extend(ml)

    print(f"[MESH] Solids extraídos: {sorted(all_meshes.keys())}")

    # Conta total de meshes por solid
    total = sum(len(v) for v in all_meshes.values())
    print(f"[MESH] Total de blocos de geometria: {total}")

    # Agrupa por (bone, color) → objetos separados por material
    bone_color_verts: dict = defaultdict(list)
    bone_color_faces: dict = defaultdict(list)

    for solid, mesh_list in all_meshes.items():
        bone = SOLID_TO_BONE.get(solid)
        if bone is None:
            print(f"[MESH] AVISO: solid '{solid}' sem bone — ignorado.")
            continue
        body_off = SOLID_BODY_OFFSET.get(solid, (0.0, 0.0, 0.0))
        for verts, faces, color in mesh_list:
            key    = (bone, tuple(round(c, 4) for c in color))
            offset = len(bone_color_verts[key])
            transformed = [
                (v[0] + body_off[0], v[1] + body_off[1], v[2] + body_off[2])
                for v in verts
            ]
            bone_color_verts[key].extend(transformed)
            bone_color_faces[key].extend(
                tuple(i + offset for i in f) for f in faces
            )

    color_label = {
        tuple(round(c, 4) for c in _WHITE): "white",
        tuple(round(c, 4) for c in _RED):   "red",
        tuple(round(c, 4) for c in _GRAY):  "gray",
        tuple(round(c, 4) for c in _LGRAY): "lgray",
        tuple(round(c, 4) for c in _BLACK): "black",
    }

    created = 0
    for (bone_name, ckey), verts in bone_color_verts.items():
        if not verts:
            continue
        if bone_name != "base_link" and bone_name not in arm_obj.data.bones:
            print(f"[MESH] AVISO: bone '{bone_name}' não existe — ignorado.")
            continue

        suf = color_label.get(ckey, f"{ckey[0]:.2f}_{ckey[1]:.2f}_{ckey[2]:.2f}")
        ob_name = f"NAO_{bone_name}_{suf}"

        if ob_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[ob_name], do_unlink=True)

        ob  = create_mesh_object(ob_name, verts, bone_color_faces[(bone_name, ckey)])
        mat = _get_material(ckey)
        ob.data.materials.append(mat)
        parent_to_armature(ob, arm_obj, bone_name)
        created += 1
        print(f"[MESH]  → {ob_name}: {len(verts)} verts")

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_PATH))
    print(f"[MESH] {created} meshes criados. Salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
