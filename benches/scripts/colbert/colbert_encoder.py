"""
Lightweight, transformers-only ColBERTv2 encoder.

Reproduces colbert-ir/colbertv2.0 token embeddings WITHOUT the heavy
colbert-ir framework. A ColBERT encoder is a BERT (768-d) whose
last_hidden_state is projected to 128-d by a single linear layer
(`linear.weight`, shape (128,768)) and L2-normalised per token.

Faithful to the official ColBERTv2 inference recipe:
  - Query:    [CLS] [unused0] <query tokens> [MASK]*pad ... [SEP]
              query is padded to a fixed length with [MASK] (query
              augmentation); ALL positions (incl. [MASK]) are kept.
  - Document: [CLS] [unused1] <doc tokens> [SEP]
              punctuation tokens are filtered out (ColBERT skiplist);
              padding [PAD] positions are dropped.
  - Both projected to 128-d and L2-normalised.

Memory discipline: batched, CPU torch, torch.set_num_threads(4).
"""
import string
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

torch.set_num_threads(4)

MODEL = "colbert-ir/colbertv2.0"
DIM = 128
Q_MARKER = 1   # [unused0]
D_MARKER = 2   # [unused1]


class ColBERT:
    def __init__(self, q_maxlen=32, d_maxlen=180):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.bert = AutoModel.from_pretrained(MODEL)  # BertModel
        self.bert.eval()
        # load the 128-d projection
        sd = load_file(hf_hub_download(MODEL, "model.safetensors"))
        w = sd["linear.weight"]  # (128, 768)
        self.linear = nn.Linear(768, DIM, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(w)
        self.linear.eval()
        self.q_maxlen = q_maxlen
        self.d_maxlen = d_maxlen
        self.mask_id = self.tok.mask_token_id
        # ColBERT punctuation skiplist (doc-side)
        self.skiplist = {
            self.tok.encode(c, add_special_tokens=False)[0]
            for c in string.punctuation
            if len(self.tok.encode(c, add_special_tokens=False)) == 1
        }

    @torch.no_grad()
    def _encode_batch(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.linear(out.last_hidden_state)  # (B, L, 128)
        emb = torch.nn.functional.normalize(emb, p=2, dim=2)
        return emb

    @torch.no_grad()
    def encode_queries(self, texts):
        """Returns list of (n_tok, 128) float32 arrays. Query augmentation:
        pad to q_maxlen with [MASK]; keep all positions."""
        # tokenize without specials, then build [CLS][Q] ... pad-with-MASK [SEP]
        enc = self.tok(
            texts, add_special_tokens=False, truncation=True,
            max_length=self.q_maxlen - 3,
        )
        batch_ids, batch_mask = [], []
        L = self.q_maxlen
        for ids in enc["input_ids"]:
            seq = [self.tok.cls_token_id, Q_MARKER] + ids + [self.tok.sep_token_id]
            attn = [1] * len(seq)
            # pad remaining with [MASK] (query augmentation), attended
            while len(seq) < L:
                seq.append(self.mask_id)
                attn.append(1)
            batch_ids.append(seq[:L])
            batch_mask.append(attn[:L])
        input_ids = torch.tensor(batch_ids)
        attn = torch.tensor(batch_mask)
        emb = self._encode_batch(input_ids, attn)  # (B,L,128)
        results = []
        for b in range(emb.shape[0]):
            results.append(emb[b].cpu().numpy().astype(np.float32))
        return results

    @torch.no_grad()
    def encode_docs(self, texts):
        """Returns list of (n_tok, 128) float32 arrays. Filters punctuation
        and padding; keeps real content tokens + [CLS] + [D] marker."""
        enc = self.tok(
            texts, add_special_tokens=False, truncation=True,
            max_length=self.d_maxlen - 3,
        )
        batch_ids, batch_mask, keep_masks = [], [], []
        # build sequences, pad to max len in batch
        seqs = []
        for ids in enc["input_ids"]:
            seq = [self.tok.cls_token_id, D_MARKER] + ids + [self.tok.sep_token_id]
            seqs.append(seq)
        maxL = max(len(s) for s in seqs)
        for seq in seqs:
            attn = [1] * len(seq) + [0] * (maxL - len(seq))
            # keep mask: drop pad + punctuation skiplist tokens
            keep = []
            for i, t in enumerate(seq):
                keep.append(1 if t not in self.skiplist else 0)
            keep = keep + [0] * (maxL - len(seq))
            padded = seq + [self.tok.pad_token_id] * (maxL - len(seq))
            batch_ids.append(padded)
            batch_mask.append(attn)
            keep_masks.append(keep)
        input_ids = torch.tensor(batch_ids)
        attn = torch.tensor(batch_mask)
        emb = self._encode_batch(input_ids, attn)  # (B,maxL,128)
        results = []
        for b in range(emb.shape[0]):
            km = np.array(keep_masks[b], dtype=bool)
            results.append(emb[b].cpu().numpy().astype(np.float32)[km])
        return results

    @staticmethod
    def pooled(tok_arr):
        """Mean of token vectors, re-normalised -> (128,) float32."""
        m = tok_arr.mean(axis=0)
        n = np.linalg.norm(m)
        if n > 0:
            m = m / n
        return m.astype(np.float32)
