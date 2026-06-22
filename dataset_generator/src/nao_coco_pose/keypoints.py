"""Definição dos keypoints COCO e o mapeamento NAO -> COCO.

Este arquivo é a PADRONIZAÇÃO do dataset: ele fixa a ordem dos 17 pontos,
o esqueleto e de onde no NAO sai cada ponto. Veja docs/keypoint_mapping.md
para a justificativa de cada escolha (em especial os pontos faciais).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Ordem oficial COCO-pose (17 keypoints). NÃO reordenar — vários frameworks
# (pycocotools, YOLO-pose, HRNet) assumem exatamente esta ordem.
COCO_KEYPOINTS: list[str] = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Esqueleto oficial COCO (índices 1-based, como vão para o JSON).
COCO_SKELETON: list[list[int]] = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]


class SourceKind(str, Enum):
    """Como a posição 3D de um keypoint é obtida no NAO."""
    NODE = "node"               # posição do Solid no fim de uma junta (endPoint)
    HEAD_OFFSET = "head_offset"  # origem da cabeça + offset fixo (pontos faciais)


@dataclass(frozen=True)
class KeypointSource:
    kind: SourceKind
    # Para NODE / HEAD_OFFSET: nome do MOTOR (servo) da junta de referência.
    # Os nomes de servo do NAO são estáveis (HeadPitch, LElbowYaw, ...), por
    # isso resolvemos a junta pelo motor e pegamos o Solid do endPoint dela.
    node: str | None = None
    # Para HEAD_OFFSET: offset (x, y, z) NO REFERENCIAL DA CABEÇA, em metros.
    # Convenção assumida: +x para frente, +y para a esquerda, +z para cima.
    # ATENÇÃO: valide os sinais com o overlay (scripts/visualize_sample.py);
    # se nariz/olhos saírem trocados, ajuste aqui.
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


# Mapeamento NAO -> COCO. Pontos do tronco/membros saem da junta correspondente;
# os faciais (que o NAO não tem) são proxies fixos no referencial da cabeça.
NAO_TO_COCO: dict[str, KeypointSource] = {
    # --- face (proxies a validar) ---
    "nose":       KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.06, 0.000, 0.02)),
    # left_eye/right_eye apontavam para "Face/Led/Left|Right" (LEDs, não juntas)
    # — find_hinge_by_motor nunca encontra um motor com esse nome, então o
    # keypoint sempre saía None. Trocado por proxy de offset da cabeça, igual
    # ao nariz/orelhas (offsets a calibrar pelo overlay).
    "left_eye":   KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.06, 0.030, 0.025)),
    "right_eye":  KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.06, -0.030, 0.025)),
    "left_ear":   KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.06, 0.000, 0.02)),
    "right_ear":  KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.06, 0.000, 0.02)),
    # --- tronco e braços ---
    "left_shoulder":  KeypointSource(SourceKind.NODE, "LShoulderPitch"),
    "right_shoulder": KeypointSource(SourceKind.NODE, "RShoulderPitch"),
    "left_elbow":     KeypointSource(SourceKind.NODE, "LElbowYaw"),
    "right_elbow":    KeypointSource(SourceKind.NODE, "RElbowYaw"),
    "left_wrist":     KeypointSource(SourceKind.NODE, "LWristYaw"),
    "right_wrist":    KeypointSource(SourceKind.NODE, "RWristYaw"),
    # --- pernas ---
    "left_hip":    KeypointSource(SourceKind.NODE, "LHipPitch"),
    "right_hip":   KeypointSource(SourceKind.NODE, "RHipPitch"),
    "left_knee":   KeypointSource(SourceKind.NODE, "LKneePitch"),
    "right_knee":  KeypointSource(SourceKind.NODE, "RKneePitch"),
    "left_ankle":  KeypointSource(SourceKind.NODE, "LAnklePitch"),
    "right_ankle": KeypointSource(SourceKind.NODE, "RAnklePitch"),
}

assert set(NAO_TO_COCO) == set(COCO_KEYPOINTS), "mapeamento incompleto"
