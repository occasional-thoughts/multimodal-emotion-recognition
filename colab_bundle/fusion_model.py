"""System D - fusion. Cross-attention vs self-attention (Rajan et al.'s question),
built on TIM-Net (System B) + text encoder (System C). All Keras.

Variants:
  'cross'      audio<->text cross-attention   (the mechanism most papers assume)
  'self'       each modality self-attends     (Rajan's stronger baseline)
  'concat'     no attention, just concatenate (sanity baseline)
  'audio_only' / 'text_only'                  (unimodal references)
"""
import sys, os
import tensorflow as tf
from tensorflow.keras import layers, Model

from TIMNET import TIMNET
from Model import WeightLayer
from text_encoder import TextEncoder




def build_fusion(fusion="cross", audio_frames=606, audio_dim=39, text_vectorizer=None,
                 d_model=64, n_heads=4, n_classes=4, dilations=10,
                 nb_filters=39, dropout=0.3, text_units=64):
    """Returns a compiled-ready Keras model taking [audio (B,T,F), text (B,) strings]."""
    a_in = layers.Input(shape=(audio_frames, audio_dim), name="audio")
    t_in = layers.Input(shape=(), dtype=tf.string, name="text")

    # ---- System B : TIM-Net, unmodified -> (B, dilations, nb_filters)
    a_seq = TIMNET(nb_filters=nb_filters, kernel_size=2, nb_stacks=1, dilations=dilations,
                   dropout_rate=0.1, activation="relu", name="TIMNET")(a_in)

    # ---- System C : text -> (B, L, 2*units) + mask
    ids = text_vectorizer(t_in)
    t_mask = tf.not_equal(ids, 0)                                   # (B, L)
    x = layers.Embedding(text_vectorizer.vocabulary_size(), 200, mask_zero=True)(ids)
    x = layers.Dropout(dropout)(x)
    t_seq = layers.Bidirectional(layers.GRU(text_units, return_sequences=True))(x, mask=t_mask)

    if fusion == "audio_only":
        fused = WeightLayer()(a_seq)
    elif fusion == "text_only":
        m = tf.cast(t_mask, t_seq.dtype)[..., None]
        fused = tf.reduce_sum(t_seq * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1e-6)
    else:
        # project both streams into a shared space so attention can compare them
        a_p = layers.Dense(d_model, name="audio_proj")(a_seq)       # (B, Na, d)
        t_p = layers.Dense(d_model, name="text_proj")(t_seq)        # (B, L,  d)
        Na = dilations
        # mask over TEXT keys: (B, query_len, key_len)
        m_t_keys = tf.cast(t_mask, tf.bool)[:, None, :]

        if fusion == "concat":
            m = tf.cast(t_mask, t_p.dtype)[..., None]
            fused = layers.Concatenate()([
                tf.reduce_mean(a_p, 1),
                tf.reduce_sum(t_p * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1e-6)])
        elif fusion == "self":
            # each modality attends to ITSELF, then pool and concatenate
            a_a = layers.MultiHeadAttention(n_heads, d_model // n_heads, name="a_self")(a_p, a_p)
            t_a = layers.MultiHeadAttention(n_heads, d_model // n_heads, name="t_self")(
                t_p, t_p, attention_mask=tf.repeat(m_t_keys, tf.shape(t_p)[1], axis=1))
            m = tf.cast(t_mask, t_a.dtype)[..., None]
            fused = layers.Concatenate()([
                tf.reduce_mean(a_a, 1),
                tf.reduce_sum(t_a * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1e-6)])
        elif fusion == "cross":
            # text queries audio  (text learns what the VOICE is doing)
            t2a = layers.MultiHeadAttention(n_heads, d_model // n_heads, name="text_to_audio")(
                t_p, a_p)                                            # (B, L, d)
            # audio queries text  (voice learns what the WORDS say)
            a2t = layers.MultiHeadAttention(n_heads, d_model // n_heads, name="audio_to_text")(
                a_p, t_p, attention_mask=tf.repeat(m_t_keys, Na, axis=1))   # (B, Na, d)
            m = tf.cast(t_mask, t2a.dtype)[..., None]
            fused = layers.Concatenate()([
                tf.reduce_sum(t2a * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1e-6),
                tf.reduce_mean(a2t, 1)])
        else:
            raise ValueError(fusion)

    h = layers.Dropout(dropout)(fused)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(dropout)(h)
    out = layers.Dense(n_classes, activation="softmax", name="emotion")(h)
    return Model([a_in, t_in], out, name=f"SystemD_{fusion}")


if __name__ == "__main__":
    import numpy as np
    z = np.load("../data/iemocap_dataset.npz", allow_pickle=True)
    texts = np.array([str(t) for t in z["transcript"]])
    vec = layers.TextVectorization(max_tokens=3000, output_sequence_length=32,
                                   standardize="lower_and_strip_punctuation")
    vec.adapt(texts[:4000])
    A = z["mfcc"][:8]
    T = tf.constant(texts[:8])
    print(f"{'variant':12} {'output':>12}  {'params':>10}")
    for v in ["audio_only", "text_only", "concat", "self", "cross"]:
        m = build_fusion(fusion=v, text_vectorizer=vec)
        y = m([A, T])
        print(f"{v:12} {str(tuple(y.shape)):>12}  {m.count_params():>10,}")
    # one training step to prove gradients flow
    m = build_fusion(fusion="cross", text_vectorizer=vec)
    m.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
    lab = tf.one_hot(np.random.randint(0, 4, 8), 4)
    h = m.fit([A, T], lab, epochs=1, verbose=0)
    print(f"\ntraining step OK  loss={h.history['loss'][0]:.4f}")
