"""Domain randomization: pose do robô, câmera e ambiente.

A amostragem (sample_*) é pura/testável; a aplicação acontece no controlador.
Toda aleatoriedade vem de um numpy.random.Generator com semente controlada,
para o dataset ser reprodutível.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

# Resultado de sample_robot_pose: categoria escolhida, ângulos de junta (rad) e
# inclinação do tronco (base_link) em graus (pitch em torno de Y, roll em torno de X).
RobotPose = namedtuple("RobotPose", ["category", "joints", "tilt_deg"])

# Resultado de sample_camera_pose: posição da câmera, descentralização no plano da
# imagem (dx, dy como fração da margem disponível, em [-1,1]) e roll da câmera (rad,
# ângulo de captura / horizonte inclinado). O controlador converte dx/dy em offset
# de mira usando a geometria (bbox do robô + FOV) para o robô nunca sair do quadro.
CamPose = namedtuple("CamPose", ["pos", "dx", "dy", "roll"])


class DomainRandomizer:
    def __init__(self, rng: np.random.Generator, cfg):
        self.rng = rng
        self.cfg = cfg   # RandomizationConfig

    # ----- pose das juntas -----
    def _sample_joint(self, lo: float, hi: float, override=None) -> float:
        """Amostra um ângulo: range cheio da categoria (override) ou range global
        reduzido por pose_range_scale em torno do meio."""
        if override is not None:
            return float(self.rng.uniform(override[0], override[1]))
        scale = float(self.cfg.pose_range_scale)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * scale
        return float(self.rng.uniform(mid - half, mid + half))

    def sample_pose(self) -> dict:
        """{nome_junta: ângulo_rad} dentro dos limites, com escala opcional.

        pose_range_scale em (0, 1] reduz a amplitude em torno do meio da faixa,
        o que tende a gerar poses mais naturais do que uniforme no range cheio.
        """
        return {
            joint: self._sample_joint(lo, hi)
            for joint, (lo, hi) in self.cfg.joint_limits.items()
        }

    def sample_robot_pose(self) -> RobotPose:
        """Sorteia uma categoria de pose (de pé/agachado/sentado/caído) por peso,
        amostra as juntas com overrides da categoria e a inclinação do tronco.

        Sem `pose_categories` configuradas, cai no comportamento antigo
        (sample_pose, sem inclinação).
        """
        cats = getattr(self.cfg, "pose_categories", None) or {}
        if not cats:
            return RobotPose(None, self.sample_pose(), (0.0, 0.0))

        names = list(cats)
        weights = np.array([float(cats[n].get("weight", 1.0)) for n in names], dtype=float)
        weights /= weights.sum()
        category = names[int(self.rng.choice(len(names), p=weights))]
        spec = cats[category]

        overrides = spec.get("joint_overrides") or {}
        joints = {
            joint: self._sample_joint(lo, hi, overrides.get(joint))
            for joint, (lo, hi) in self.cfg.joint_limits.items()
        }

        tilt = spec.get("torso_tilt_deg") or {}
        pitch = float(self.rng.uniform(*tilt.get("pitch", (0.0, 0.0))))
        roll = float(self.rng.uniform(*tilt.get("roll", (0.0, 0.0))))
        return RobotPose(category, joints, (pitch, roll))

    # ----- câmera (esférica, olhando para o alvo) -----
    def sample_camera_pose(self, target, fit_distance):
        """Amostra posição da câmera numa casca esférica ao redor do alvo, mais
        a descentralização no plano da imagem e o roll (ângulo de captura).

        `fit_distance` é a distância em que o robô "encaixa" no quadro (calculada
        pelo bbox do robô e o FOV, no controlador). O raio = fit_distance * fator
        sorteado em `distance_factor` (>=1 mantém o robô no quadro; fatores maiores
        = robô menor / mais distância).

        Returns CamPose(pos, dx, dy, roll): dx/dy em [-1,1] são a fração da margem
        disponível para descentralizar o robô (o controlador garante que ele não
        saia do quadro); roll (rad) inclina a imagem.
        """
        c = self.cfg.camera
        fmin, fmax = c.get("distance_factor", (1.2, 2.0))
        r = float(fit_distance) * self.rng.uniform(fmin, fmax)
        az = self.rng.uniform(*np.deg2rad(c["azimuth_deg"]))
        el = self.rng.uniform(*np.deg2rad(c["elevation_deg"]))
        target = np.asarray(target, dtype=float)
        offset = r * np.array([np.cos(el) * np.cos(az),
                               np.cos(el) * np.sin(az),
                               np.sin(el)])
        # Descentralização: vetor num disco de raio `decenter` (<=1), escalado
        # depois pela margem geométrica no controlador.
        dc = float(c.get("decenter", 0.0))
        dx, dy = self.rng.uniform(-1.0, 1.0, size=2)
        n = float(np.hypot(dx, dy))
        if n > 1.0:
            dx, dy = dx / n, dy / n
        roll = float(self.rng.uniform(*np.deg2rad(c.get("roll_deg", (0.0, 0.0)))))
        return CamPose(target + offset, float(dx) * dc, float(dy) * dc, roll)

    # ----- ambiente -----
    def sample_lighting(self) -> dict:
        l = self.cfg.lighting
        return {
            "intensity": float(self.rng.uniform(*l["intensity"])),
            "color": [float(self.rng.uniform(*l["color_range"])) for _ in range(3)],
        }

    def sample_background(self):
        bgs = self.cfg.backgrounds
        return str(self.rng.choice(bgs)) if bgs else None
