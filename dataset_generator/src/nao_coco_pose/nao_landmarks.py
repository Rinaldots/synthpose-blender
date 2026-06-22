"""Acesso à cinemática do NAO via Supervisor: localizar juntas e aplicar pose.

Os nomes de SERVO do NAO são estáveis (HeadPitch, LElbowYaw, RAnkleRoll, ...),
então localizamos cada junta pelo motor e usamos o Solid do `endPoint` dela.
Isso é mais robusto do que adivinhar nomes de Solids internos do PROTO.

Nenhuma função aqui importa a API do Webots no topo: todas recebem nós já
obtidos pelo controlador, mantendo o pacote importável fora do simulador.
"""
from __future__ import annotations

import numpy as np

from .keypoints import NAO_TO_COCO, SourceKind

_HINGE_TYPES = {"HingeJoint", "Hinge2Joint"}


def get_field(node, name):
    """Busca um campo do nó, recorrendo aos campos do nó-base se o PROTO não o expõe.

    `nao_node` é a instância do PROTO MyNao: `getField` só enxerga os campos
    declarados no PROTO, enquanto `getBaseNodeField` acessa os campos reais do
    nó Robot por baixo. `getBaseNodeField` só é válido em nós que são, eles
    próprios, instâncias de PROTO — chamá-lo num nó comum (Solid, HingeJoint,
    Shape, ...) dentro da árvore gera um erro do Webots, por isso o `isProto()`.
    """
    field = node.getField(name)
    if field is None and node.isProto():
        field = node.getBaseNodeField(name)
    return field


def dump_solid_tree(node, depth: int = 0, max_depth: int = 14) -> None:
    """Imprime a árvore (nome + tipo) abaixo de `node`. Utilitário de descoberta.

    Rode UMA vez no controlador para inspecionar a estrutura do seu NAO:
        from nao_coco_pose.nao_landmarks import dump_solid_tree
        dump_solid_tree(robot.getFromDef("NAO")); return
    """
    if node is None or depth > max_depth:
        return
    try:
        nf = get_field(node, "name")
        name = nf.getSFString() if nf else "<sem nome>"
    except Exception:
        name = "<?>"
    print("  " * depth + f"- {name} [{node.getTypeName()}]")
    for field_name in ("children", "device"):
        f = get_field(node, field_name)
        if f:
            for i in range(f.getCount()):
                dump_solid_tree(f.getMFNode(i), depth + 1, max_depth)
    ep = get_field(node, "endPoint")
    if ep:
        dump_solid_tree(ep.getSFNode(), depth + 1, max_depth)


def find_hinge_by_motor(root, motor_name: str):
    """Retorna o HingeJoint cujo `device`/`device2` contém um motor de nome `motor_name`.

    `device2` é necessário para o segundo grau de liberdade de um Hinge2Joint
    (ex.: HeadPitch é o `device2` do Hinge2Joint `HeadYaw`).
    """
    if root is None:
        return None
    if root.getTypeName() in _HINGE_TYPES:
        for dev_name in ("device", "device2"):
            dev = get_field(root, dev_name)
            if dev:
                for i in range(dev.getCount()):
                    d = dev.getMFNode(i)
                    nf = get_field(d, "name") if d else None
                    if nf and nf.getSFString() == motor_name:
                        return root
    children = get_field(root, "children")
    if children:
        for i in range(children.getCount()):
            found = find_hinge_by_motor(children.getMFNode(i), motor_name)
            if found:
                return found
    ep = get_field(root, "endPoint")
    if ep:
        return find_hinge_by_motor(ep.getSFNode(), motor_name)
    return None


def endpoint_solid(hinge):
    """Solid no `endPoint` de uma junta (o elo que ela movimenta)."""
    if hinge is None:
        return None
    ep = get_field(hinge, "endPoint")
    return ep.getSFNode() if ep else None


class NaoPoser:
    """Aplica ângulos de junta no NAO escrevendo os campos `position IS <Junta>`
    expostos por `MyNao.proto`/`Nao.proto` (IS-mapeados até `jointParameters`).

    Funciona a partir de um controlador externo (a câmera-rig) que tem poderes
    de Supervisor, sem o NAO precisar de controlador próprio. Para poses
    estáticas, escrever a posição "teletransporta" a junta — exatamente o que
    queremos. Campos obtidos via `getBaseNodeField` (fallback de `get_field`)
    são somente leitura no Webots, por isso cada junta precisa ser exposta como
    field IS-mapeado no PROTO (ver protos/Nao.proto e protos/MyNao.proto). Se
    preferir poses fisicamente assentadas, rode o NAO com um controlador
    próprio e use motor.setPosition() (ver docs/pipeline.md).
    """

    def __init__(self, nao_node, joint_names):
        self.nao_node = nao_node
        self._params = {}
        for j in joint_names:
            field = nao_node.getField(j)
            if field is None:
                print(f"[NaoPoser] AVISO: junta '{j}' não exposta no PROTO — será ignorada.")
            self._params[j] = field

    def apply(self, angles: dict) -> None:
        for joint, angle in angles.items():
            field = self._params.get(joint)
            if field is not None:
                field.setSFFloat(float(angle))


class KeypointResolver:
    """Resolve e lê as posições 3D (no mundo) dos 17 keypoints COCO no NAO."""

    def __init__(self, nao_node):
        self.nao_node = nao_node
        self._solids: dict[str, object] = {}   # keypoint -> Solid (kind=NODE)
        self._head = None                       # Solid da cabeça (head_offset)

    def resolve(self) -> "KeypointResolver":
        """Localiza os nós uma única vez (chamar após montar a cena)."""
        head_motor = None
        for kp, src in NAO_TO_COCO.items():
            if src.kind == SourceKind.NODE:
                solid = endpoint_solid(find_hinge_by_motor(self.nao_node, src.node))
                if solid is None:
                    print(f"[KeypointResolver] AVISO: junta '{src.node}' "
                          f"(keypoint '{kp}') não encontrada.")
                self._solids[kp] = solid
            elif src.kind == SourceKind.HEAD_OFFSET:
                head_motor = src.node or head_motor
        if head_motor:
            self._head = endpoint_solid(find_hinge_by_motor(self.nao_node, head_motor))
            if self._head is None:
                print(f"[KeypointResolver] AVISO: cabeça ('{head_motor}') não encontrada.")
        return self

    def _head_offset_world(self, offset):
        # ponto = R_cabeça @ offset_local + t_cabeça
        t = np.asarray(self._head.getPosition(), dtype=float)
        R = np.asarray(self._head.getOrientation(), dtype=float).reshape(3, 3)
        return R @ np.asarray(offset, dtype=float) + t

    def get_keypoints_world(self) -> dict:
        """{nome_keypoint: ndarray(x,y,z)} no mundo; None se o nó não resolveu."""
        out: dict[str, object] = {}
        for kp, src in NAO_TO_COCO.items():
            try:
                if src.kind == SourceKind.NODE:
                    solid = self._solids.get(kp)
                    out[kp] = np.asarray(solid.getPosition(), float) if solid else None
                elif src.kind == SourceKind.HEAD_OFFSET:
                    out[kp] = self._head_offset_world(src.offset) if self._head else None
                else:
                    out[kp] = None
            except Exception as exc:  # nó pode ter sido invalidado entre passos
                print(f"[KeypointResolver] erro lendo '{kp}': {exc}")
                out[kp] = None
        return out
