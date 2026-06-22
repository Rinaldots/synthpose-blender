# Pipeline

Visão geral do fluxo de dados, da cena no Webots até o dataset COCO-pose.

## Etapas

1. **Configurar cena** — `worlds/dataset.wbt`. Arena, NAO (`DEF NAO`) e o
   robô-rig da câmera (`DEF RIG`, supervisor, roda o controlador).
2. **Loop de captura** (repete N vezes, `controllers/dataset_generator/`):
   1. **Randomizar** — pose das juntas, posição/orientação da câmera e
      (TODO) iluminação/fundo. `randomization.py`.
   2. **Assentar** — alguns passos de simulação para a física estabilizar e a
      câmera renderizar com a nova pose.
   3. **Capturar RGB** — imagem do frame. `camera.py`.
   4. **Ler juntas 3D** — ground truth via Supervisor. `nao_landmarks.py`.
   5. **Projetar 3D→2D** — pinhole + matriz K + extrínseca atual. `projection.py`.
   6. **Visibilidade + bbox** — flags COCO e caixa. `visibility.py`.
   7. **Anotar** — acumula imagem + keypoints. `coco_writer.py`.
3. **Montar JSON COCO** — `images`, `annotations`, `categories`. `coco_writer.py`.
4. **Validar / QA** — `scripts/validate_dataset.py` e `visualize_sample.py`.

## A ideia central: duas fontes por frame

Cada frame produz **dois dados em paralelo**: a imagem RGB renderizada (o input
do modelo) e as coordenadas 3D reais das juntas lidas pelo Supervisor (a ground
truth). Elas se reencontram na etapa de visibilidade, onde os keypoints já
projetados em 2D são cruzados com a imagem/profundidade.

## Pontos sensíveis

- **Extrínseca por frame.** Como a câmera é randomizada, a pose dela é relida a
  cada captura (`CameraRig.cam_to_world`). Nunca cacheie a extrínseca.
- **Convenção de eixos.** `projection.AXIS_REMAP` assume que a câmera do Webots
  olha ao longo de −Z. Se o overlay sair invertido/espelhado, ajuste lá.
- **Oclusão.** Sem um RangeFinder, todo keypoint dentro do quadro vira "visível"
  (flag 2). Para flag 1 (ocluso) real, adicione um RangeFinder ao rig e passe a
  imagem de profundidade para `visibility.occluded_by_depth`.
- **Bbox.** A versão atual envolve os keypoints + margem. Para precisão, derive
  da máscara de segmentação (`Camera` com `recognitionSegmentation`).

## Aplicação de pose: duas estratégias

- **Padrão (deste esqueleto):** o rig escreve `jointParameters.position` via
  Supervisor (`NaoPoser`). Teletransporta as juntas — ideal para poses
  estáticas, sem o NAO precisar de controlador.
- **Alternativa física:** rode o NAO com controlador próprio e use
  `motor.setPosition()` para poses fisicamente assentadas (mais lento, útil se
  você quer poses dinâmicas/estáveis).
