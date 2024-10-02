import jax
from flax import linen as nn
import jax.numpy as jnp

from typing import Optional
from layers import quantizations
from layers import linears
from layers import initializers
from layers import normalizations
from layers import attentions
from layers import models

import common_types

Config = common_types.Config
Mesh = common_types.Mesh

Attention = attentions.Attention
Quant = quantizations.AqtQuantization


class GrokDecoderLayer(nn.Module):
  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  def layer_norm(self, name:str):
    cfg = self.config
    return models.RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=name,
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
    )

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      shard_activations = True,
  ):
    cfg = self.config

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))

    lnx = self.layer_norm("pre_self_attention_layer_norm")(inputs)
    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    # Self-attention block
    attention_layer = Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim, # key_size in grok-1
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        mesh=self.mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    attention_lnx = self.layer_norm("post_self_attention_layer_norm")(attention_lnx)
    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))

    # Fully Connected
    hidden_states = inputs + attention_lnx
    hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", "activation_length", "activation_embed"))

    if cfg.num_experts > 1:
      mlp_lnx, _ = linears.MoeBlock(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=self.mesh,
          kernel_init=initializers.nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
          kernel_axes=('embed', 'mlp'),
          weight_dtype=cfg.weight_dtype,
          dtype=cfg.dtype,
      )(self.layer_norm("moe_layer_norm")(hidden_states), cfg.ffn_size)
      mlp_lnx = nn.with_logical_constraint(
          mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
      )
    else:
      raise NotImplementedError(f"Not implemented for {cfg.num_experts=}")

    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(mlp_lnx, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
