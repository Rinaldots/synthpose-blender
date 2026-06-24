# CLAUDE.md — synthpose-webots

Geração de dataset sintético COCO-pose (17 keypoints) para o NAO via **Blender 5.1.2 headless**.
Pipeline: aplica pose aleatória no armature → posiciona câmera → renderiza →
projeta 3D→2D → calcula visibilidade/bbox → grava anotação COCO.

## Comandos
- Gerar dataset:
  `cd dataset_generator/blender && blender --background nao_full.blend --python dataset_generator_blender.py`
- Overlay de calibração (múltiplas poses):
  `blender --background nao_full.blend --python overlay_multi.py`
- Overlay simples (1 pose):
  `blender --background nao_full.blend --python overlay_test.py`
- Validar JSON gerado:
  `python scripts/validate_dataset.py output/annotations/person_keypoints_train.json`
- Testar módulos puros fora do Blender: `PYTHONPATH=src python -c "import nao_coco_pose"`

## Arquitetura (decisões que não devem ser quebradas)

### Blender
- `blender/nao_full.blend` — cena principal: NAO_Armature + 36 mesh objects.
- `blender/nao_poser_blender.py` — aplica FK ao armature e lê posições 3D dos keypoints.
- `blender/blender_camera.py` — extrai K e cam_to_world; define `BLENDER_AXIS_REMAP = diag(1,-1,-1)`.
- `blender/dataset_generator_blender.py` — loop de geração; roda DENTRO do Blender.

### Convenções de eixos
- NAO frame: X=frente, Y=esquerda, Z=cima.
- Blender world: X=direita, Y=frente, Z=cima.
- Conversão NAO→Blender: `(-v.y, v.x, v.z)` (rotação -90° em Z).
- `BLENDER_AXIS_REMAP = diag(1,-1,-1)`: converte cam-space Blender (Y cima, -Z frente) para CV-space (Y baixo, +Z frente).

### Keypoints faciais
- Definidos como offsets fixos no referencial do bone `HeadPitch` (`_HEAD_OFFSETS_NAO` em `nao_poser_blender.py`).
- Rotação aplicada via `R_delta = R_pose @ R_rest.inverted()` — usa FK do Blender diretamente, sem reconstruir ângulos.
- Nunca reconstruir o frame da cabeça manualmente por ângulos — o bone já tem a FK acumulada.

### Módulos puros (NumPy, testáveis sem Blender)
- `src/nao_coco_pose/projection.py` — `world_to_pixels()`
- `src/nao_coco_pose/coco_writer.py` — `CocoDatasetBuilder`
- `src/nao_coco_pose/visibility.py` — `visibility_flag()`, `bbox_from_keypoints()`
- `src/nao_coco_pose/randomization.py` — `DomainRandomizer`
- `src/nao_coco_pose/config.py` — carrega YAMLs

## Convenções
- Ordem dos 17 keypoints é a COCO oficial — NÃO reordenar (`keypoints.py`).
- Código com identificadores em inglês; docstrings/comentários em português.
- Não adicionar dependências sem necessidade (numpy, pyyaml, opencv, pycocotools).

## Pontos sensíveis / TODO
- Offsets faciais em `_HEAD_OFFSETS_NAO`: calibrar pelo overlay se necessário.
- `BLENDER_AXIS_REMAP`: validado pelo overlay_test.py — não alterar sem re-validar.
- Oclusão real (flag de visibilidade 1) ainda não implementada (exigiria ray casting).
- Randomização de fundo (HDRI / planos coloridos) ainda não aplicada.

## Limites de arquivos
- Não editar: `protos/` (PROTO/URDF do robô), `output/` (gerado), `blender/nao_full.blend` (salvar só via Blender).
- Editar à vontade: `blender/*.py`, `src/`, `scripts/`, `config/`, `docs/`.
