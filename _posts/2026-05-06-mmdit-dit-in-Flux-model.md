---
title: MM-DiT & DiT in Flux model
author: yuchong
date: 2026-05-06 00:34:00 +0800
categories: [AI]
tags: [AI]
math: true
---

FLUX 模型中“双流注意力（Double-Stream Attention）+ 单流注意力（Single-Stream Attention）

# 双流注意力模块 (Double-Stream Blocks)

负责处理文本和图像两种不同模态的信息。在此阶段，文本 Token 和图像 Token 分别通过独立的权重进行处理，并通过交叉注意力机制进行双向信息交互和融合，通常包含 19 个双流模块。类似于Stable Diffusion 3 (SD3) 中的Multi-modal Diffusion Transformer（MM-DiT ,多模态扩散 Transformer ）。


# 单流注意力模块 (Single-Stream Blocks)

在双流模块完成初步的图文融合后，文本和图像的 Token 会被拼接成一个统一的序列。单流模块使用共享的权重对这个合并后的序列进行处理，进一步细化图像特征。通常包含 38 个单流模块，专注于图像内容的最终生成。

代码实现：

```python
self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )


#双流
for index_block, block in enumerate(self.transformer_blocks):
      encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, #图像token对应的Embedding
                    encoder_hidden_states=encoder_hidden_states, #text token对应的Embedding
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
      )

# 单流
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
for index_block, block in enumerate(self.single_transformer_blocks):
     hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
     )

```




