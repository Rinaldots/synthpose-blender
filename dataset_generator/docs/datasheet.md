# Datasheet do dataset

Modelo a preencher quando o dataset for gerado. Mantê-lo atualizado é o que
torna o dataset reprodutível e reutilizável por terceiros.

## Motivação
- **Objetivo:** dataset sintético de pose 2D para robôs humanoides (NAO),
  padronizado no formato COCO-pose.
- **Uso pretendido:** treino/avaliação de estimadores de pose; pré-treino antes
  de fine-tuning em dados reais.

## Composição
- **Nº de imagens:** _(preencher)_
- **Resolução:** _(ex.: 640×480)_
- **Keypoints:** 17 (ver `docs/keypoint_mapping.md`).
- **Anotação por imagem:** uma instância "person" (o NAO).
- **Splits:** train / val / test = _(ex.: 0.8 / 0.1 / 0.1)_

## Processo de geração
- **Simulador / versão:** Webots R2025a.
- **Robô:** NAO via `protos/MyNao.proto`.
- **Semente:** _(preencher — `dataset.yaml: seed`)_
- **Faixas de randomização:**
  - Pose das juntas: limites em `config/randomization.yaml`, escala
    `pose_range_scale = _(preencher)_`.
  - Câmera: raio _(min–max)_, azimute _(faixa)_, elevação _(faixa)_.
  - Iluminação / fundo: _(preencher quando implementado)_.
- **Câmera:** intrínseca derivada de `fieldOfView`; extrínseca relida por frame.
- **Convenção de eixos:** `AXIS_REMAP` validada? _(sim/não)_

## Pré-processamento e rótulos
- **Visibilidade:** flags COCO 0/1/2 (ver mapeamento). Oclusão via RangeFinder?
  _(sim/não)_
- **Bbox:** keypoints + margem _(ou segmentação, se aplicável)_.

## Limitações conhecidas
- Pontos faciais são proxies fixos na cabeça (não anatômicos).
- Gap sim-to-real: aparência sintética; considerar mais randomização visual.
- _(outras a registrar)_

## Manutenção
- **Responsável:** _(preencher)_
- **Versão do dataset:** _(preencher)_
- **Como regenerar:** `config/` + semente + esta versão do código → idêntico.
