"""Cria o armature do NAO no Blender a partir do URDF e salva nao_rig.blend.

Uso:
    blender --background --python create_nao_armature.py

O script lê o URDF em ../protos/nao_exported.urdf (relativo a este arquivo),
cria um Armature object com bones nomeados como os joints do NAO, e salva
nao_rig.blend na mesma pasta deste script.

Convenção Blender vs. NAO:
  - NAO usa eixos Webots: X=frente, Y=esquerda, Z=cima
  - Blender usa: X=direita, Y=frente, Z=cima (modo padrão)
  - Fazemos uma rotação de -90° em Z ao importar para alinhar: X_ble=Y_nao, Y_ble=-X_nao
  NÃO alterar esta convenção sem atualizar nao_keypoints_blender.py.
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector

URDF_PATH = Path(__file__).parent.parent / "protos" / "nao_exported.urdf"
OUTPUT_PATH = Path(__file__).parent / "nao_rig.blend"

# Joints que correspondem a keypoints COCO (usados em keypoints.py).
# O bone desse joint = posição do keypoint quando todos os ângulos = 0.
KEYPOINT_JOINTS = {
    "HeadPitch",        # proxy: nariz, olhos, orelhas
    "LShoulderPitch",   # left_shoulder
    "RShoulderPitch",   # right_shoulder
    "LElbowYaw",        # left_elbow
    "RElbowYaw",        # right_elbow
    "LWristYaw",        # left_wrist
    "RWristYaw",        # right_wrist
    "LHipPitch",        # left_hip
    "RHipPitch",        # right_hip
    "LKneePitch",       # left_knee
    "RKneePitch",       # right_knee
    "LAnklePitch",      # left_ankle
    "RAnklePitch",      # right_ankle
}


def _parse_vec(s, default="0 0 0"):
    return [float(v) for v in (s or default).split()]


def parse_urdf(path: Path) -> dict:
    """Retorna dict joint_name -> {parent_link, child_link, xyz, rpy, axis}."""
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        orig = j.find("origin")
        ax = j.find("axis")
        joints[name] = {
            "type":        j.get("type"),
            "parent_link": j.find("parent").get("link"),
            "child_link":  j.find("child").get("link"),
            "xyz":  _parse_vec(orig.get("xyz") if orig is not None else None),
            "rpy":  _parse_vec(orig.get("rpy") if orig is not None else None, "0 0 0"),
            "axis": _parse_vec(ax.get("xyz") if ax is not None else None, "0 0 1"),
        }
    return joints


def _rpy_mat(rpy) -> Matrix:
    return Euler(rpy, "XYZ").to_matrix().to_4x4()


def _xyz_mat(xyz) -> Matrix:
    m = Matrix.Identity(4)
    m.translation = Vector(xyz)
    return m


def compute_world_poses(joints: dict) -> dict:
    """Retorna dict joint_name -> Matrix4x4 no referencial do mundo NAO."""
    link_tf: dict[str, Matrix] = {"base_link": Matrix.Identity(4)}

    def visit(link):
        for jname, jdata in joints.items():
            if jdata["parent_link"] != link:
                continue
            local = _xyz_mat(jdata["xyz"]) @ _rpy_mat(jdata["rpy"])
            world = link_tf[link] @ local
            link_tf[jdata["child_link"]] = world
            visit(jdata["child_link"])

    visit("base_link")

    # pose de cada joint = transform do child_link dele
    return {jname: link_tf[jdata["child_link"]]
            for jname, jdata in joints.items()
            if jdata["child_link"] in link_tf}


def nao_to_blender(v: Vector) -> Vector:
    """NAO (X=frente,Y=esq,Z=cima) → Blender (X=dir,Y=frente,Z=cima).

    Rotação -90° em Z: x_ble = y_nao, y_ble = -x_nao, z_ble = z_nao
    """
    return Vector((-v.y, v.x, v.z))


def build_joint_tree(joints: dict) -> dict:
    """link_name -> lista de joint_names que têm esse link como parent."""
    children: dict[str, list] = {}
    for jname, jdata in joints.items():
        children.setdefault(jdata["parent_link"], []).append(jname)
    return children


def create_armature(joints: dict, world_poses: dict) -> bpy.types.Object:
    """Cria e retorna o objeto Armature no Blender."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.armature_add(enter_editmode=True)
    arm_obj = bpy.context.active_object
    arm_obj.name = "NAO_Armature"
    arm_data = arm_obj.data
    arm_data.name = "NAO_Rig"

    # Remove o bone padrão gerado pelo Blender
    for b in list(arm_data.edit_bones):
        arm_data.edit_bones.remove(b)

    child_joints = build_joint_tree(joints)

    def find_child_position(jname: str) -> Vector | None:
        """Posição média dos joints filhos (para definir bone tail)."""
        jdata = joints[jname]
        child_link = jdata["child_link"]
        kids = child_joints.get(child_link, [])
        kids = [k for k in kids if joints[k]["type"] != "fixed"]
        if not kids:
            return None
        positions = []
        for k in kids:
            if k in world_poses:
                t = world_poses[k].translation
                positions.append(nao_to_blender(t))
        return sum(positions, Vector((0, 0, 0))) / len(positions) if positions else None

    bone_map: dict[str, bpy.types.EditBone] = {}

    # Primeiro passo: cria todos os bones
    for jname, jdata in joints.items():
        if jdata["type"] == "fixed":
            continue
        if jname not in world_poses:
            continue

        head_nao = world_poses[jname].translation
        head_ble = nao_to_blender(head_nao)

        tail_ble = find_child_position(jname)
        if tail_ble is None:
            # Leaf: bone curto apontando para cima
            tail_ble = head_ble + Vector((0, 0, 0.04))
        elif (tail_ble - head_ble).length < 1e-4:
            tail_ble = head_ble + Vector((0, 0, 0.04))

        bone = arm_data.edit_bones.new(jname)
        bone.head = head_ble
        bone.tail = tail_ble
        bone_map[jname] = bone

    # Segundo passo: parenteia os bones
    for jname, jdata in joints.items():
        if jname not in bone_map:
            continue
        # Procura o joint pai (aquele cujo child_link = parent_link deste)
        parent_jname = None
        for pj, pjdata in joints.items():
            if pjdata["child_link"] == jdata["parent_link"] and pj in bone_map:
                parent_jname = pj
                break
        if parent_jname:
            bone_map[jname].parent = bone_map[parent_jname]
            bone_map[jname].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def main():
    if not URDF_PATH.exists():
        sys.exit(f"URDF não encontrado: {URDF_PATH}")

    print(f"[NAO] Lendo URDF: {URDF_PATH}")
    joints = parse_urdf(URDF_PATH)
    print(f"[NAO] {len(joints)} joints encontrados")

    world_poses = compute_world_poses(joints)
    print(f"[NAO] {len(world_poses)} poses calculadas")

    # Limpa a cena padrão do Blender
    bpy.ops.wm.read_homefile(use_empty=True)

    arm_obj = create_armature(joints, world_poses)

    # Mostra quais bones correspondem a keypoints COCO
    missing = KEYPOINT_JOINTS - set(arm_obj.data.bones.keys())
    if missing:
        print(f"[NAO] AVISO: keypoints sem bone: {missing}")
    else:
        print(f"[NAO] Todos os {len(KEYPOINT_JOINTS)} keypoints COCO têm bone.")

    # Posiciona o armature na altura padrão do NAO (z=0.334)
    arm_obj.location = (0, 0, 0.334)

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_PATH))
    print(f"[NAO] Salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
