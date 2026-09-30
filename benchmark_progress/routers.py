"""Original prompt-stable random binary gates."""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Iterable

# Fixed byte domain separator used to generate the archived random sequence.
_ROUTER_DOMAIN = bytes([112, 114, 111, 98, 101, 124])
import numpy as np


@dataclass(frozen=True)
class GateBank:
    candidate_gates: dict[str, np.ndarray]
    tie_gates: dict[str, np.ndarray]
    candidate_seed_hex: tuple[str, ...]
    seed_manifest_sha256: str



def _candidate_key(master_seed: str, candidate_index: int) -> bytes:
    return hashlib.sha256(_ROUTER_DOMAIN + f"{master_seed}|{candidate_index}".encode("utf-8")).digest()



def _tie_key(tie_seed: str) -> bytes:
    return hashlib.sha256(f"tie|{tie_seed}".encode("utf-8")).digest()



def _prf_sign(key: bytes, canonical_id: str) -> np.int8:
    bit = hashlib.blake2b(canonical_id.encode("utf-8"), key=key, digest_size=1).digest()[0] & 1
    return np.int8(1 if bit else -1)



def build_gate_bank(
    tasks: Iterable, max_candidates: int, master_seed: str, tie_seed: str
) -> GateBank:
    candidate_keys = tuple(_candidate_key(master_seed, index) for index in range(max_candidates))
    seed_hex = tuple(key.hex() for key in candidate_keys)
    manifest_hash = hashlib.sha256("\n".join(seed_hex).encode("ascii")).hexdigest()
    tie_key = _tie_key(tie_seed)
    candidate_gates: dict[str, np.ndarray] = {}
    tie_gates: dict[str, np.ndarray] = {}
    for task in tasks:
        # The routing key is the normalized observable prompt hash.  It must
        # not contain an occurrence counter, dataset row, task position, gold
        # label, or model outcome: repeated observable prompts are one x and
        # therefore receive the same g_t(x).
        routing_keys = task.content_hashes
        gates = np.empty((max_candidates, len(routing_keys)), dtype=np.int8)
        for candidate_index, key in enumerate(candidate_keys):
            gates[candidate_index] = np.fromiter(
                (_prf_sign(key, routing_key) for routing_key in routing_keys),
                dtype=np.int8,
                count=len(routing_keys),
            )
        candidate_gates[task.name] = gates
        tie_gates[task.name] = np.fromiter(
            (_prf_sign(tie_key, routing_key) for routing_key in routing_keys),
            dtype=np.int8,
            count=len(routing_keys),
        )
    return GateBank(candidate_gates, tie_gates, seed_hex, manifest_hash)
