"""System C - semantic encoder (Keras, to match TIM-Net's framework).

Contract (mirrors System B):
    input :  (batch,) raw transcript strings
    output:  seq    (batch, max_len, 2*units)   <- sequence for cross-attention
             pooled (batch, 2*units)            <- fixed vector
             mask   (batch, max_len)            <- True where a real word sits
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


class TextEncoder:
    def __init__(self, max_len=32, vocab_size=3000, embed_dim=200, units=64,
                 dropout=0.3, use_self_attention=False, n_heads=4):
        self.max_len, self.units = max_len, units
        self.vectorizer = layers.TextVectorization(
            max_tokens=vocab_size, output_sequence_length=max_len,
            standardize="lower_and_strip_punctuation")
        self.cfg = dict(vocab_size=vocab_size, embed_dim=embed_dim, units=units,
                        dropout=dropout, use_self_attention=use_self_attention,
                        n_heads=n_heads)
        self.model = None

    def adapt(self, texts):
        """Fit the vocabulary. IMPORTANT: call on TRAIN texts only (no leakage)."""
        self.vectorizer.adapt(texts)
        return self

    def build(self, embedding_matrix=None):
        c = self.cfg
        inp = layers.Input(shape=(), dtype=tf.string, name="text")
        ids = self.vectorizer(inp)                                   # (B, L)
        mask = tf.not_equal(ids, 0)
        emb_kw = dict(input_dim=c["vocab_size"], output_dim=c["embed_dim"], mask_zero=True)
        if embedding_matrix is not None:                             # e.g. GloVe
            emb_kw.update(weights=[embedding_matrix], trainable=True)
        x = layers.Embedding(**emb_kw, name="embedding")(ids)
        x = layers.Dropout(c["dropout"])(x)
        x = layers.Bidirectional(
            layers.GRU(c["units"], return_sequences=True), name="bi_gru")(x, mask=mask)
        if c["use_self_attention"]:                                  # intra-modal attention
            att = layers.MultiHeadAttention(num_heads=c["n_heads"],
                                            key_dim=2 * c["units"] // c["n_heads"],
                                            name="text_self_attn")
            x = layers.LayerNormalization()(x + att(x, x, attention_mask=None))
        seq = x                                                      # (B, L, 2*units)
        m = tf.cast(mask, seq.dtype)[..., None]
        pooled = tf.reduce_sum(seq * m, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1e-6)
        self.model = Model(inp, [seq, pooled, mask], name="SystemC_TextEncoder")
        return self.model


if __name__ == "__main__":
    import csv, numpy as np, os
    rows = list(csv.DictReader(open("../iemocap_index.csv")))
    texts = np.array([r["transcript"] for r in rows])
    print(f"loaded {len(texts)} transcripts")

    enc = TextEncoder(use_self_attention=True).adapt(texts[:4000])   # train split only
    m = enc.build()
    seq, pooled, mask = m(tf.constant(texts[:6]))
    print(f"\nseq    {tuple(seq.shape)}   <- feeds cross-attention")
    print(f"pooled {tuple(pooled.shape)}")
    print(f"mask   {tuple(mask.shape)}")
    print(f"params {m.count_params():,}")

    print("\nmasking check (real words per utterance):")
    for t, k in zip(texts[:5], mask.numpy().sum(1)):
        print(f"  {k:2d} tokens | {t[:52]}")

    print("\nvocab learned:", len(enc.vectorizer.get_vocabulary()))
    print("sample tokens:", enc.vectorizer.get_vocabulary()[:10])
