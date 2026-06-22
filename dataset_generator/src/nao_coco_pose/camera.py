"""Câmera observadora: captura RGB e leitura dos parâmetros (Webots).

Recebe um Supervisor já criado pelo controlador. A câmera fica em um robô-rig
externo (ver worlds/dataset.wbt), apontada para o NAO. A intrínseca vem do
próprio dispositivo (fonte da verdade); a extrínseca é lida a cada frame —
essencial porque a câmera é randomizada.
"""
from __future__ import annotations

import numpy as np

from . import projection


class CameraRig:
    def __init__(self, supervisor, camera_name: str = "camera", camera_def: str = "CAMERA"):
        self.supervisor = supervisor
        self.camera = supervisor.getDevice(camera_name)
        if self.camera is None:
            raise RuntimeError(f"câmera '{camera_name}' não encontrada no robô do controlador")
        # Nó da câmera, para ler a pose no mundo. Requer DEF na world.
        self.camera_node = supervisor.getFromDef(camera_def)
        if self.camera_node is None:
            raise RuntimeError(f"DEF {camera_def} não encontrado (defina-o na world)")

    def enable(self, timestep: int) -> None:
        self.camera.enable(timestep)

    @property
    def width(self) -> int:
        return self.camera.getWidth()

    @property
    def height(self) -> int:
        return self.camera.getHeight()

    def intrinsics(self) -> np.ndarray:
        """Matriz K a partir do FOV horizontal do dispositivo."""
        return projection.build_intrinsics(self.width, self.height, self.camera.getFov())

    def cam_to_world(self) -> np.ndarray:
        """Pose 4x4 (câmera -> mundo) no instante atual."""
        return projection.pose_list_to_matrix(self.camera_node.getPose())

    def capture_rgb(self) -> np.ndarray:
        """Imagem RGB (H, W, 3) uint8. O Webots entrega BGRA."""
        raw = self.camera.getImage()
        arr = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 4))
        return arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
