# CLAUDE.md — synthpose-webots

Geração de dataset sintético COCO-pose (17 keypoints) para o NAO no Webots.
Pipeline: renderiza um frame → lê as juntas 3D via Supervisor → projeta 3D→2D →
calcula visibilidade/bbox → grava anotação COCO. Documentação em `docs/`.

## Comandos
- Instalar deps (no Python que o Webots usa): `pip install -e .`
- Gerar dataset: abrir `worlds/dataset.wbt` no Webots R2025a e rodar a simulação.
- Validar/QA: `python scripts/validate_dataset.py output/annotations/person_keypoints.json`
- Overlay (teste decisivo da projeção):
  `python scripts/visualize_sample.py output/annotations/person_keypoints.json output/images 0 overlay.png`
- Testar o núcleo fora do Webots: rodar com `PYTHONPATH=src`.

## Arquitetura (decisões que não devem ser quebradas)
- A câmera observadora vive num robô-rig externo (`DEF RIG`) que roda o
  controlador como Supervisor. A câmera PRECISA estar no mesmo robô do
  controlador para `getImage()` funcionar.
- As poses do NAO são aplicadas via Supervisor escrevendo
  `jointParameters.position` (`NaoPoser`); o NAO não tem controlador próprio.
- Juntas são resolvidas pelo NOME DO SERVO (estável no NAO), pegando o Solid do
  `endPoint`. Pontos faciais = offsets fixos no referencial da cabeça.
- A extrínseca da câmera é relida a cada frame — nunca cachear.
- `projection.py`, `coco_writer.py` e `visibility.py` são NumPy puro e testáveis
  sem o Webots.

## Convenções
- A ordem dos 17 keypoints é a COCO oficial — NÃO reordenar (`keypoints.py`).
- Código com identificadores em inglês; docstrings/comentários em português.
- Não adicionar dependências sem necessidade (numpy, pyyaml, opencv, pycocotools).

## Pontos sensíveis / TODO
- `projection.AXIS_REMAP`: convenção de eixos da câmera — validar pelo overlay.
- Offsets faciais em `keypoints.py`: calibrar pelo overlay.
- Oclusão real (flag de visibilidade 1) exige um RangeFinder no rig.
- Randomização de iluminação/fundo ainda não aplicada no controlador.

## Limites de arquivos
- Não editar: `protos/` (PROTO do robô), `output/` (gerado).
- Editar à vontade: `src/`, `controllers/`, `scripts/`, `config/`, `docs/`.
