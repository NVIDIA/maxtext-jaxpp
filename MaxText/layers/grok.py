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
import max_logging

Config = common_types.Config
Mesh = common_types.Mesh

Attention = attentions.Attention
Quant = quantizations.AqtQuantization


class GrokMoeBlock(linears.MoeBlock):

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    router_logits = linears.DenseGeneral(
            self.num_experts,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self.kernel_axes,
            name="router")(inputs)

    routing_probs = jax.nn.softmax(router_logits.astype(jnp.float32))
    top_k_routing_probs, top_k_indices = jax.lax.top_k(routing_probs, k=self.num_experts_per_tok)
    softmax_probs = top_k_routing_probs.astype(self.weight_dtype)

    weights = jnp.zeros_like(router_logits, dtype=self.weight_dtype)
    index_update = (jnp.arange(router_logits.shape[0])[:, None, None], jnp.arange(router_logits.shape[1])[:, None], top_k_indices)
    weights = weights.at[index_update].set(softmax_probs)

    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(self.num_experts,
                                                            cfg.emb_dim,
                                                            self.ffn_size(cfg.emb_dim))

    with jax.named_scope("wi_0"):
      layer_w0 = jnp.einsum("BLE,NEH -> BLNH", inputs, w0_kernel)
    with jax.named_scope("wi_1"):
      layer_w1 = jnp.einsum("BLE,NEH -> BLNH", inputs, w1_kernel)
    layer_w0_act = linears._convert_to_activation_function(cfg.mlp_activations[0])(layer_w0)
    layer_multiply = jnp.multiply(layer_w0_act, layer_w1)
    with jax.named_scope("wo"):
      intermediate_layer = jnp.einsum("BLNH,NHE -> BLNE", layer_multiply, wo_kernel)
    with jax.named_scope("w_sum"):
      output = jnp.einsum("BLNE,BLN -> BLE", intermediate_layer, weights)

    return output

  def ffn_size(self, emb_size, widening_factor=8.0):
      _ffn_size = int(widening_factor * emb_size) * 2 // 3
      _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
      max_logging.log(f"emd_size: {emb_size} adjusted ffn_size: {_ffn_size}")
      return _ffn_size


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
        quantize_kvcache=cfg.quantize_kvcache,
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
      mlp_lnx = GrokMoeBlock(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          kernel_init=initializers.nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
          kernel_axes=('embed', 'mlp'),
          dtype=cfg.dtype,
      )(self.layer_norm("moe_layer_norm")(hidden_states))
      mlp_lnx = nn.with_logical_constraint(
          mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
      )
    else:
      raise NotImplementedError(f"Not implemented for {num_experts=}")

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
