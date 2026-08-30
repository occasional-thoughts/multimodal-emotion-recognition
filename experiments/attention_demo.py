"""Worked example: "Oh what a fantastic day" said while crying."""
import numpy as np
np.set_printoptions(precision=2, suppress=True)

# 4 interpretable dimensions so we can read the vectors:
#   [ positive , negative , high-energy , low-energy ]
DIMS = ["pos", "neg", "hiE", "loE"]

# ---------- SYSTEM C : text -> sequence of word vectors ----------
words = ["oh", "what", "fantastic", "day"]
TEXT = np.array([
    [0.1, 0.1, 0.2, 0.1],   # "oh"        - neutral filler
    [0.1, 0.1, 0.1, 0.1],   # "what"      - neutral
    [0.9, 0.0, 0.7, 0.0],   # "fantastic" - STRONGLY positive
    [0.4, 0.0, 0.2, 0.1],   # "day"       - mildly positive
])

# ---------- SYSTEM B : audio -> sequence of frame vectors ----------
frames = ["frame1", "frame2", "frame3"]
AUDIO = np.array([
    [0.0, 0.8, 0.0, 0.9],   # shaky, low energy  -> sad
    [0.0, 0.9, 0.1, 0.8],   # trembling          -> sad
    [0.0, 0.7, 0.0, 0.9],   # breaking voice     -> sad
])

print("SYSTEM C output (text), shape", TEXT.shape, " dims =", DIMS)
for w, v in zip(words, TEXT): print(f"   {w:10} {v}")
print("\nSYSTEM B output (audio), shape", AUDIO.shape)
for f, v in zip(frames, AUDIO): print(f"   {f:10} {v}")

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, qn, kn, title):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)      # how well each query matches each key
    W = softmax(scores)                # turn into weights that sum to 1
    out = W @ V                        # weighted average of the values
    print("\n" + "="*70); print(title); print("="*70)
    print(f"  Q (queries) = {qn[0]}...  |  K,V (keys/values) = {kn[0]}...")
    print("\n  attention weights (each ROW sums to 1):")
    print("      " + "".join(f"{n:>12}" for n in kn))
    for i, r in enumerate(W): print(f"  {qn[i]:>10}" + "".join(f"{v:12.2f}" for v in r))
    print("\n  output vectors:")
    for i, r in enumerate(out): print(f"  {qn[i]:>10} {r}")
    return out

# ---------- SELF-ATTENTION : text looks at ITSELF ----------
self_out = attention(TEXT, TEXT, TEXT, words, words,
    "SELF-ATTENTION on text  (Q=text, K=text, V=text)")
print("\n  -> 'fantastic' mostly attends to ITSELF and 'day' (both positive).")
print("     It never sees the audio. Result stays POSITIVE.")

# ---------- CROSS-ATTENTION : text looks at AUDIO ----------
cross_out = attention(TEXT, AUDIO, AUDIO, words, frames,
    "CROSS-ATTENTION text->audio  (Q=text, K=audio, V=audio)")
print("\n  -> every word now retrieves AUDIO values (all sad).")
print("     Result flips NEGATIVE - the crying is detected.")

# ---------- pool to the single fixed vector (what System D outputs) ----------
print("\n" + "="*70); print("FINAL POOLING -> one fixed vector per utterance"); print("="*70)
print(f"  self-attention  mean: {self_out.mean(0)}")
print(f"  cross-attention mean: {cross_out.mean(0)}")
s, c = self_out.mean(0), cross_out.mean(0)
print(f"\n  self  : pos={s[0]:.2f} vs neg={s[1]:.2f}  -> predicts HAPPY  (wrong)")
print(f"  cross : pos={c[0]:.2f} vs neg={c[1]:.2f}  -> predicts SAD    (correct)")
