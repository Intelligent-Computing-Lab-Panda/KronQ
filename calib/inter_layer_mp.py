"""Inter-layer mixed-precision allocation via the KronQ sensitivity score.

Sublayer types are ranked by the KronQ score  s = tr(H_G) * tr(H_X)  (measured
post-incoherence). To spend a bit budget, upgrade the top-N most sensitive
types by +1 bit across ALL layers.

  n_upgrade (args.inter_layer_mp) = number of sublayer types to upgrade
    e.g. n_upgrade=1 -> down_proj of every layer goes base->base+1
         n_upgrade=2 -> down_proj + gate_proj, etc.
  avg_bits = base_bits + n_upgrade / len(SUBLAYER_RANK)

The ranking below is the measured order on LLaMA-2-7B / LLaMA-3-8B (identical on
both): down > gate > up > o > v > k > q. Note the activation-only score tr(H_X)
cannot separate q/k/v (shared input); the H_G factor breaks that degeneracy.
"""

# Sublayer types in descending KronQ sensitivity s = tr(H_G) * tr(H_X).
import os as _os
SUBLAYER_RANK = _os.environ.get("KRONQ_MP_RANK", "down_proj,gate_proj,up_proj,o_proj,v_proj,k_proj,q_proj").split(",")


def get_sublayer_bits(name, base_bits, n_upgrade):
    """Bit-width for sublayer `name`: base_bits+1 if its type is among the top
    `n_upgrade` most sensitive types, else base_bits."""
    if not n_upgrade:
        return base_bits
    for sublayer_type in SUBLAYER_RANK[:int(n_upgrade)]:
        if sublayer_type in name:
            return base_bits + 1
    return base_bits
