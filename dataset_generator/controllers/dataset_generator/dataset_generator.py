"""Controlador (Supervisor) do Webots — laço principal de geração do dataset.

Roda no robô-rig da câmera (DEF RIG em worlds/dataset.wbt), que tem poderes de
Supervisor: ele move a própria câmera, lê as juntas do NAO (DEF NAO) e aplica
as poses. Fluxo:

  configurar -> [loop N]: randomizar -> assentar -> capturar RGB
              -> ler juntas 3D -> projetar -> visibilidade/bbox -> anotar
              -> [fim] -> salvar JSON COCO

Pontos marcados como TODO dependem do ajuste fino da SUA cena.
"""
from __future__ import annotations

import sys
from pathlib import Path

# torna o pacote src/ importável quando o Webots roda este controlador
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # salvar imagens (instale no Python que o Webots usa)
from controller import Supervisor  # API do Webots

from nao_coco_pose import config as cfg
from nao_coco_pose.camera import CameraRig
from nao_coco_pose.coco_writer import CocoDatasetBuilder
from nao_coco_pose.keypoints import COCO_KEYPOINTS
from nao_coco_pose.nao_landmarks import KeypointResolver, NaoPoser, get_field  # , dump_solid_tree
from nao_coco_pose.projection import (
    look_at_rotation,
    rotation_to_axis_angle,
    world_to_pixels,
)
from nao_coco_pose.randomization import DomainRandomizer
from nao_coco_pose.visibility import bbox_from_keypoints, visibility_flag

# pose "de repouso" do NAO para reancorar a base a cada frame (em pé na arena)
NAO_HOME_TRANSLATION = [0.0, 0.0, 0.333]
NAO_HOME_ROTATION = [0.0, 0.0, 1.0, 0.0]


def main() -> None:
    conf = ROOT / "config"
    cam_cfg = cfg.load_camera(conf / "camera.yaml")
    rnd_cfg = cfg.load_randomization(conf / "randomization.yaml")
    ds_cfg = cfg.load_dataset(conf / "dataset.yaml")
    rng = cfg.make_rng(ds_cfg.seed)

    robot = Supervisor()                       # este controlador É o robô-rig
    timestep = int(robot.getBasicTimeStep())

    rig = CameraRig(robot, camera_name="camera", camera_def="CAMERA")
    rig.enable(timestep)
    rig_node = robot.getSelf()                 # para mover a câmera (o rig)

    nao_node = robot.getFromDef("NAO")
    if nao_node is None:
        raise RuntimeError("DEF NAO não encontrado na world")

    # Descomente UMA vez para inspecionar a árvore e conferir nomes de juntas:
    # dump_solid_tree(nao_node); return

    nao_translation = get_field(nao_node, "translation")
    nao_rotation = get_field(nao_node, "rotation")

    resolver = KeypointResolver(nao_node).resolve()
    poser = NaoPoser(nao_node, list(rnd_cfg.joint_limits))
    randomizer = DomainRandomizer(rng, rnd_cfg)

    builder = CocoDatasetBuilder()
    builder.info.update({
        "camera": {"width": rig.width, "height": rig.height},
        "robot": "NAO (MyNao.proto)",
        "seed": ds_cfg.seed,
        "pose_range_scale": rnd_cfg.pose_range_scale,
    })

    out_dir = ROOT / ds_cfg.output_dir
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    K = rig.intrinsics()
    settle = int(rnd_cfg.camera.get("settle_steps", 3))

    n = 0
    while robot.step(timestep) != -1 and n < ds_cfg.num_samples:
        # 1) randomizar cena
        nao_translation.setSFVec3f(NAO_HOME_TRANSLATION)
        nao_rotation.setSFRotation(NAO_HOME_ROTATION)
        poser.apply(randomizer.sample_pose())

        cam_pos, target = randomizer.sample_camera_pose(nao_node.getPosition())
        R = look_at_rotation(cam_pos, target)
        rig_node.getField("translation").setSFVec3f([float(v) for v in cam_pos])
        rig_node.getField("rotation").setSFRotation(rotation_to_axis_angle(R))
        # TODO: aplicar iluminação/fundo (randomizer.sample_lighting / .sample_background)

        # 2) deixar a física assentar e a câmera renderizar
        for _ in range(settle):
            if robot.step(timestep) == -1:
                break

        # 3) capturar as duas fontes do frame
        rgb = rig.capture_rgb()                       # imagem (input)
        kps_world = resolver.get_keypoints_world()    # juntas 3D (ground truth)
        cam_to_world = rig.cam_to_world()

        # 4) projetar 3D -> 2D na ordem COCO
        pts, present = [], []
        for name in COCO_KEYPOINTS:
            p = kps_world.get(name)
            present.append(p is not None)
            pts.append(p if p is not None else (0.0, 0.0, 0.0))
        uv, depth = world_to_pixels(pts, cam_to_world, K)

        # 5) visibilidade + bbox
        flags, kp_xyv = [], []
        for i, _name in enumerate(COCO_KEYPOINTS):
            f = 0 if not present[i] else visibility_flag(
                uv[i], depth[i], rig.width, rig.height, depth_image=None
            )  # TODO: passar mapa de profundidade do RangeFinder p/ oclusão real
            flags.append(f)
            kp_xyv.append((uv[i][0], uv[i][1], f))
        bbox, area = bbox_from_keypoints(uv, flags, rig.width, rig.height)

        # 6) gravar imagem + anotação
        fname = f"{ds_cfg.image_prefix}{n:06d}.{ds_cfg.image_format}"
        cv2.imwrite(str(img_dir / fname), rgb[:, :, ::-1])  # RGB -> BGR p/ OpenCV
        builder.add_sample(fname, rig.width, rig.height, kp_xyv, bbox, area)
        n += 1

    builder.save(out_dir / "annotations" / "person_keypoints.json")
    print(f"[ok] {n} amostras salvas em {out_dir}")


if __name__ == "__main__":
    main()
