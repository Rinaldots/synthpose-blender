"""Particionamento de um job de geração entre várias máquinas (Python puro).

O gerador é *sharding-friendly* por construção: o nome do arquivo é
``nao_{start+i:05d}`` e a semente da amostragem deriva de ``start``
(``make_rng(seed + start)``). Logo, basta dar a cada máquina um intervalo
``[start, start+num)`` disjunto para que imagens **e** poses não colidam — e o
merge posterior só precise renumerar os IDs COCO.

Este módulo transforma a tripla (total, world_size, rank) nesse intervalo, para
que o usuário nunca precise calcular ``--start``/``--num`` na mão.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shard:
    """Fatia de trabalho de uma máquina: gera ``num`` amostras a partir de ``start``."""
    rank: int
    world_size: int
    start: int
    num: int

    @property
    def end(self) -> int:
        """Índice exclusivo do fim do intervalo (start + num)."""
        return self.start + self.num


def shard_range(total: int, world_size: int, rank: int) -> Shard:
    """Divide ``total`` amostras em ``world_size`` fatias contíguas e disjuntas.

    O resto da divisão é distribuído às primeiras máquinas (uma amostra extra
    cada), de modo que a soma dos ``num`` seja exatamente ``total`` e os
    tamanhos difiram no máximo em 1.

    Ex.: total=10000, world_size=2
        rank 0 -> Shard(start=0,    num=5000)
        rank 1 -> Shard(start=5000, num=5000)

    Ex.: total=101, world_size=2
        rank 0 -> Shard(start=0,  num=51)
        rank 1 -> Shard(start=51, num=50)
    """
    if world_size < 1:
        raise ValueError(f"world_size deve ser >= 1 (recebido {world_size})")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank deve estar em [0, {world_size}) (recebido {rank})")
    if total < 0:
        raise ValueError(f"total deve ser >= 0 (recebido {total})")

    base = total // world_size
    rem  = total % world_size          # primeiras `rem` máquinas ganham +1
    num   = base + (1 if rank < rem else 0)
    start = rank * base + min(rank, rem)
    return Shard(rank=rank, world_size=world_size, start=start, num=num)
