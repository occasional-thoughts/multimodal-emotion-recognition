"""System E (comparison view) - shows audio-only, text-only and fusion side by side.

When the streams disagree, that IS the project's thesis on screen: the words say
one thing, the voice says another, and fusion has to arbitrate.
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, gradio as gr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa, librosa.display
from compare import ThreeWay
from predict import LAB

TW = ThreeWay()
NAMES = {"audio": "Audio only (System B)", "text": "Text only (System C)",
         "fusion": "Fusion (System D)"}
HAVE = list(TW.models.keys())
HDR = " &nbsp;|&nbsp; ".join(
    f"**{NAMES[k]}**: {TW.models[k].meta['wa']*100:.1f}%" for k in HAVE)


def spectrogram(sig, sr):
    t, _ = librosa.effects.trim(sig, top_db=30)
    if len(t) > 100:
        sig = t
    S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024, hop_length=256)), ref=np.max)
    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=110)
    librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="hz",
                             ax=ax, cmap="magma")
    ax.set_ylim(0, 4000)
    ax.set_title("STFT magnitude spectrogram (0-4 kHz, linear scale)", fontsize=9)
    fig.tight_layout()
    return fig


def analyse(wav, manual):
    empty = ({}, {}, {}, "", None, "Record or upload audio first.")
    if not wav:
        return empty
    try:
        out = TW.predict(wav, transcript=(manual.strip() or None))
    except Exception as e:
        return ({}, {}, {}, "", None, f"Error: {e}")

    s = out["streams"]
    preds = {k: v["emotion"] for k, v in s.items()}
    agree = len(set(preds.values())) == 1

    if agree:
        verdict = (f"### All streams agree: **{list(preds.values())[0].upper()}**\n\n"
                   "The words and the voice point the same way - an unambiguous case.")
    else:
        lines = "  \n".join(f"- {NAMES[k]} -> **{v.upper()}**" for k, v in preds.items())
        verdict = ("### Streams DISAGREE\n\n" + lines +
                   "\n\nThis is the ambiguity the system exists to resolve: one modality "
                   "alone would be misleading here, and fusion has to arbitrate.")
    if "fusion" in s:
        verdict += f"\n\n**Final answer (fusion): {s['fusion']['emotion'].upper()}**"

    g = lambda k: s[k]["probs"] if k in s else {}
    return (g("audio"), g("text"), g("fusion"),
            out["transcript"], spectrogram(out["signal"], out["sr"]), verdict)


with gr.Blocks(title="Multimodal Emotion Recognition - stream comparison") as demo:
    gr.Markdown("# Multimodal Emotion Recognition\n"
                "### Audio vs Text vs Fusion, on the same utterance")
    gr.Markdown(HDR + "  \n*(held-out accuracy of each loaded checkpoint)*")
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speech")
            manual = gr.Textbox(label="Transcript (optional)",
                                placeholder="Leave blank to let Whisper transcribe",
                                info="Type the words to bypass ASR - run it both ways to "
                                     "show how ASR errors change the prediction.")
            btn = gr.Button("Analyse", variant="primary")
            gr.Markdown("**Try a sarcastic line** - say *\"oh great, just wonderful\"* "
                        "in a flat, defeated voice. Text should read positive; audio should not.")
        with gr.Column(scale=1):
            verdict = gr.Markdown()
            with gr.Row():
                p_audio = gr.Label(label="Audio only", num_top_classes=4)
                p_text = gr.Label(label="Text only", num_top_classes=4)
                p_fuse = gr.Label(label="Fusion", num_top_classes=4)
            trans = gr.Textbox(label="Transcript used", interactive=False)
            spec = gr.Plot(label="Spectrogram")
    btn.click(analyse, [audio, manual], [p_audio, p_text, p_fuse, trans, spec, verdict])
    gr.Markdown("---\n*Trained on IEMOCAP, 4-class, leave-one-session-out. "
                "Transcripts here come from Whisper ASR; training used ground-truth "
                "transcripts, so live accuracy is lower than the reported figures.*")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, inbrowser=False)
