"""WikiText-2 perplexity evaluation, matching Frantar & Alistarh (2023):
non-overlapping 2048-token chunks, loss computed only over the
next-token-prediction targets within each chunk."""

import torch


@torch.no_grad()
def eval_ppl(model, chunks: list[torch.Tensor], device: str = "cpu") -> float:
    model.eval()
    nlls = []
    total_tokens = 0
    for chunk in chunks:
        chunk = chunk.to(device)
        out = model(chunk, labels=chunk)
        seqlen = chunk.shape[1]
        # HF's `labels` loss already shifts by one and averages over
        # (seqlen - 1) tokens; recover the summed NLL for this chunk so
        # chunks of equal length can be pooled into one corpus-level PPL.
        neg_log_likelihood = out.loss.float() * (seqlen - 1)
        nlls.append(neg_log_likelihood)
        total_tokens += seqlen - 1
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()
