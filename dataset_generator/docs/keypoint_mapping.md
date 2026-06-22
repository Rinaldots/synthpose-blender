# Mapeamento de keypoints (NAO → COCO)

Este é o documento que define a **padronização** do dataset. Qualquer mudança
aqui invalida a comparabilidade entre amostras, então trate-o como um contrato.

## Os 17 keypoints COCO

A ordem é a oficial do COCO-pose e **não pode mudar** (frameworks como
pycocotools, YOLO-pose e HRNet assumem exatamente esta sequência):

```
0  nose            6  right_shoulder   12 right_hip
1  left_eye        7  left_elbow       13 left_knee
2  right_eye       8  right_elbow      14 right_knee
3  left_ear        9  left_wrist       15 left_ankle
4  right_ear       10 right_wrist      16 right_ankle
5  left_shoulder   11 left_hip
```

A definição vive em `src/nao_coco_pose/keypoints.py` (`COCO_KEYPOINTS`,
`COCO_SKELETON`, `NAO_TO_COCO`).

## De onde sai cada ponto

O NAO expõe nomes de servo estáveis (`HeadPitch`, `LElbowYaw`, `RAnkleRoll`…),
então a posição 3D de cada keypoint de tronco/membro é a origem do `Solid` no
`endPoint` da junta correspondente. Isso é mais robusto do que tentar adivinhar
os nomes dos Solids internos do PROTO.

| Keypoint COCO | Junta de referência (motor) |
|---|---|
| left/right_shoulder | L/R ShoulderPitch |
| left/right_elbow | L/R ElbowYaw |
| left/right_wrist | L/R WristYaw |
| left/right_hip | L/R HipPitch |
| left/right_knee | L/R KneePitch |
| left/right_ankle | L/R AnklePitch |

## Pontos faciais — a decisão importante

O NAO **não tem** nariz, olhos nem orelhas como referências articulares. Para
manter os 17 keypoints (em vez de usar um subconjunto), eles são definidos como
**proxies fixos no referencial da cabeça** (`endPoint` de `HeadPitch`):

| Keypoint | Offset na cabeça (x_frente, y_esq, z_cima), em metros |
|---|---|
| nose | (0.06, 0.000, 0.02) |
| left_eye | (0.05, +0.025, 0.05) |
| right_eye | (0.05, −0.025, 0.05) |
| left_ear | (0.00, +0.045, 0.04) |
| right_ear | (0.00, −0.045, 0.04) |

O ponto é transformado para o mundo por `R_cabeça · offset + t_cabeça`, então
ele acompanha a rotação da cabeça automaticamente.

> **A validar:** os sinais/valores dos offsets dependem da orientação real do
> referencial da cabeça no seu NAO. Gere uma amostra, rode
> `scripts/visualize_sample.py` e ajuste os offsets em `keypoints.py` até nariz,
> olhos e orelhas caírem onde deveriam.

## Flags de visibilidade (padrão COCO)

- `0` — não rotulado: fora do quadro, atrás da câmera ou junta não resolvida.
- `1` — rotulado, porém ocluso (precisa de mapa de profundidade; ver pipeline).
- `2` — rotulado e visível.

## Esqueleto

Usamos o esqueleto oficial COCO de 19 arestas (`COCO_SKELETON`), em índices
1-based, gravado em `categories[0].skeleton` do JSON.
