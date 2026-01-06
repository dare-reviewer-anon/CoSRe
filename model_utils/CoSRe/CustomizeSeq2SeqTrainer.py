# utils/cosre_generate.py
import torch

@torch.no_grad()
def greedy_decode_with_prefix(model, prefix_embeds, max_new_tokens, stopping_criteria=None):
    """
    prefix_embeds: (1, Lm, d)  memory tokens embeddings
    """
    out = model(inputs_embeds=prefix_embeds, use_cache=True)
    past = out.past_key_values

    # start token: use model's decoder_start_token_id or BOS
    if hasattr(model.config, "decoder_start_token_id") and model.config.decoder_start_token_id is not None:
        cur = torch.tensor([[model.config.decoder_start_token_id]], device=prefix_embeds.device)
    else:
        cur = torch.tensor([[model.config.bos_token_id]], device=prefix_embeds.device)

    generated = []
    for _ in range(max_new_tokens):
        o = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)  # (1,1)

        generated.append(nxt)

        # stopping criteria (your StopStringCriteria is token-based via tokenizer; you can also stop on eos)
        if model.config.eos_token_id is not None and int(nxt.item()) == int(model.config.eos_token_id):
            break

        cur = nxt

    return torch.cat(generated, dim=1) if generated else torch.empty((1,0), dtype=torch.long, device=prefix_embeds.device)

@torch.no_grad()
def cosre_generate_one_batch(model, processor, batch, cosre_module, image_token_mask_fn,
                            max_new_tokens=256, stopping_criteria=None):
    """
    batch: your trainer batch dict (contains input_ids, attention_mask, etc.)
    image_token_mask_fn: function(input_ids)-> bool mask of shape (seq_len,) selecting image tokens
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)  # (B,L)
    B, L = input_ids.shape

    emb_layer = model.get_input_embeddings()

    outputs = []
    for b in range(B):
        ids = input_ids[b]  # (L,)
        mask_img = image_token_mask_fn(ids)  # (L,) bool

        # split ids into text part + image part (interleaved case: simplest is: use embeddings and mask)
        embs = emb_layer(ids)  # (L,d)

        vis_embs = embs[mask_img]            # (Nv,d)
        text_embs = embs[~mask_img]          # (Nt,d)

        # build compact memory prefix
        Xs = cosre_module(text_embs, vis_embs)          # (Lm,d)
        prefix = Xs.unsqueeze(0).contiguous()           # (1,Lm,d)

        gen = greedy_decode_with_prefix(model, prefix, max_new_tokens, stopping_criteria=stopping_criteria)
        outputs.append(gen)

    return outputs  # list of (1,T) token ids
