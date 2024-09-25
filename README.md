# Triton-Version-of-Flash-Attention-with-Flexible-Masks

## Flash Attention with Flexible Masks in Triton

- [Triton](https://github.com/triton-lang/triton) provides a flexible way to write customized kernels for user-defined operations, such as flash attention with flexible masks.
- This repo provides a very drafty and prototypical implementation, which is mostly based upon and adapted from (actually mostly contributes to these great works!):
  - Triton-kernels: https://github.com/triton-lang/kernels/blob/main/kernels/flash_attention.py
  - (This kernel uses nested-loops by default and is slow in backward, see [here](https://github.com/triton-lang/triton/issues/2046).)
  - FlagAttention: https://github.com/FlagOpen/FlagAttention/blob/main/src/flag_attn/flash.py
  - (This kernel uses two separate kernels for dkdv and dq in backward, and seems to be much faster, see [here](https://github.com/FlagOpen/FlagAttention/issues/4).)
- `flash_triton.py` contains the kernels and `testing.py` contains some simple testing codes.
- One addition feature added is support for flexible masks:
  - For example, the `attn_mask` argument in [sdpa](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), which currently the CUDA-version flash attention does not support (for example, see these issues, [here](https://github.com/Dao-AILab/flash-attention/issues/840), [and here](https://github.com/Dao-AILab/flash-attention/issues/1179)).
  - Exactly the same motivation as [FlexAttention](https://pytorch.org/blog/flexattention/).
  - For this extra feature, you can provide two extra input tensors of `eq` and `ek`, (which have the shape of [bs, H, Lq] and [bs, H, Lk],) and a `score_func_mode` indicating how the masks would be calculated with these extra inputs.
  - Currently, an example mode `SCORE_FUNC_MODE1_DOC` is implemented, which supports a document attention mask of `extra_attn_mask = ((eq.unsqueeze(-1) >= 0) | (eq.unsqueeze(-1) == eq.unsqueeze(-2)))  # [bs, H, Lq, Lk]` (see [here](https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences)).
  - Similar flexible masking modes can be implemented similarly.
  - Things not implemented and might need more efforts: real block-sparse attention.
- Testings are done with the environment of `triton==3.0.0 torch==2.4.0` and with one `A100-SXM4-40GB` GPU.
- See `2410_flash.pdf` for a simple illustration of flash attention and some related results.
