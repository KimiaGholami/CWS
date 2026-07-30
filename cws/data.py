"""Calibration (C4) and evaluation (WikiText-2) data loading.

Matches the CWS paper's protocol (Sec 3.1): 64 calibration batches of
batch size 4, sequence length 512, drawn from C4; WikiText-2 test-set
perplexity computed over non-overlapping 2048-token chunks (Frantar &
Alistarh, 2023).
"""

import random

import torch


def get_c4_calibration(tokenizer, nsamples: int = 64, seqlen: int = 512, seed: int = 0):
    """Return `nsamples` random `seqlen`-token windows sampled from C4,
    each shaped (1, seqlen), matching the paper's batch-size-4 / seqlen-512
    protocol (call sites can group 4 consecutive windows into one batch, or
    feed them one at a time -- the Hessian accumulator is agnostic to batch
    grouping since it flattens batch and sequence dims before accumulating).
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        streaming=True,
    )

    rng = random.Random(seed)
    samples = []
    it = iter(dataset)
    buffer = []
    while len(samples) < nsamples:
        try:
            example = next(it)
        except StopIteration:
            it = iter(dataset)
            continue
        buffer.append(example["text"])
        if len(buffer) < 32:
            continue
        text = buffer[rng.randrange(len(buffer))]
        buffer = []
        enc = tokenizer(text, return_tensors="pt")
        if enc.input_ids.shape[1] <= seqlen:
            continue
        start = rng.randrange(enc.input_ids.shape[1] - seqlen)
        samples.append(enc.input_ids[:, start : start + seqlen])
    return samples


def get_wikitext2_test(tokenizer, seqlen: int = 2048):
    """Tokenize the full WikiText-2 test split and chunk it into
    non-overlapping `seqlen`-token windows for perplexity evaluation."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    n_chunks = input_ids.shape[1] // seqlen
    chunks = [
        input_ids[:, i * seqlen : (i + 1) * seqlen] for i in range(n_chunks)
    ]
    return chunks
