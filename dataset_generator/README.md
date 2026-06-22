# nao-coco-pose

Geração de **dataset sintético de pose** para o robô humanoide **NAO** no
**Webots**, padronizado no formato **COCO-pose** (17 keypoints).

A cada frame o pipeline renderiza uma imagem do NAO e extrai a *ground truth* 3D
das juntas via Supervisor, projeta esses pontos para 2D, calcula visibilidade e
bounding box, e grava tudo como anotação COCO. Com randomização de pose, câmera
e ambiente, gera-se um dataset variado e reprodutível.

> Estado: **esqueleto funcional**. O núcleo matemático (projeção, escrita COCO,
> resolução de juntas) está implementado; alguns pontos acoplados à sua cena
> estão marcados como `TODO` (ver "O que ajustar").

## Estrutura

```
nao_coco_pose/
├── README.md
├── requirements.txt / pyproject.toml
├── docs/
│   ├── keypoint_mapping.md   # a padronização NAO→COCO (leia primeiro)
│   ├── pipeline.md           # etapas e fluxo de dados
│   └── datasheet.md          # modelo de datasheet do dataset
├── config/
│   ├── camera.yaml           # resolução e FOV
│   ├── randomization.yaml    # limites de juntas, faixas de câmera/ambiente
│   └── dataset.yaml          # N, splits, saída, semente
├── protos/MyNao.proto        # seu PROTO do NAO
├── worlds/dataset.wbt        # cena: NAO + rig da câmera observadora
├── controllers/dataset_generator/
│   └── dataset_generator.py  # ponto de entrada: o laço de captura
├── src/nao_coco_pose/
│   ├── keypoints.py          # 17 keypoints, esqueleto, mapeamento NAO→COCO
│   ├── projection.py         # pinhole 3D→2D, look-at, eixo-ângulo
│   ├── nao_landmarks.py      # juntas via Supervisor + aplicação de pose
│   ├── camera.py             # captura RGB, intrínseca, extrínseca
│   ├── visibility.py         # flags COCO + bbox
│   ├── randomization.py      # domain randomization
│   ├── coco_writer.py        # monta o JSON COCO
│   └── config.py             # carrega YAML + semente
└── scripts/
    ├── validate_dataset.py   # QA com pycocotools
    └── visualize_sample.py   # overlay de keypoints (valida a projeção)
```

## Setup

Instale as dependências no **mesmo Python que o Webots usa** (Preferences →
Python command):

```bash
pip install -e .            # ou: pip install -r requirements.txt
```

O controlador acrescenta `src/` ao `sys.path` sozinho; o `pip install -e .` é
opcional, mas ajuda a rodar os scripts e testes fora do simulador.

## Como rodar

1. Abra `worlds/dataset.wbt` no Webots R2025a.
2. Confirme os `DEF`: `NAO` (o robô) e `RIG`/`CAMERA` (o rig observador que roda
   o controlador `dataset_generator`).
3. Rode a simulação. As imagens vão para `output/images/` e a anotação para
   `output/annotations/person_keypoints.json`.
4. Valide e inspecione:

```bash
python scripts/validate_dataset.py output/annotations/person_keypoints.json
python scripts/visualize_sample.py \
    output/annotations/person_keypoints.json output/images 0 overlay.png
```

O `overlay.png` é o teste decisivo: se os pontos caem nas juntas, a projeção e o
mapeamento estão corretos.

## O que ajustar à sua cena

Quatro pontos dependem do seu ambiente e estão marcados como `TODO` no código:

1. **Convenção de eixos da câmera** — valide `projection.AXIS_REMAP` pelo
   overlay; inverta sinais se a imagem sair espelhada/de cabeça p/ baixo.
2. **Offsets faciais** — ajuste os proxies de nariz/olhos/orelhas em
   `keypoints.py` (ver `docs/keypoint_mapping.md`).
3. **Oclusão real** — adicione um RangeFinder ao rig e passe o mapa de
   profundidade para `visibility.occluded_by_depth` (hoje tudo no quadro = visível).
4. **Iluminação/fundo** — conecte `randomizer.sample_lighting/background` aos
   nós da cena no controlador.

Se algum nome de junta não resolver, descomente a chamada a `dump_solid_tree`
no controlador para listar a árvore do NAO e conferir os nomes.

## Próximos passos sugeridos

Gere ~10 frames com `pose_range_scale` baixo, rode o overlay e calibre eixos e
offsets faciais. Só depois suba o `num_samples` e ligue a randomização visual.
