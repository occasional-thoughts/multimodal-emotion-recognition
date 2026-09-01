"""System E - web interface. Record or upload speech, see transcript,
spectrogram, and predicted emotion.

  python app.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, gradio as gr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa, librosa.display
from predict import EmotionRecognizer, LAB

print("loading emotion model...")
REC = EmotionRecognizer()
print("loading Whisper (one-time, ~30s) so the first demo click is fast...")
import predict as _p, whisper as _w
_p._asr = _w.load_model("base")
print("ready.")
INFO = (f"model: **{REC.tag}** &nbsp;|&nbsp; features: **{REC.meta['features']}** "
        f"&nbsp;|&nbsp; fusion: **{REC.meta['fusion']}** "
        f"&nbsp;|&nbsp; held-out WA: **{REC.meta['wa']*100:.2f}%** (session {REC.meta['fold']})")


def spectrogram(sig, sr):
    trimmed, _ = librosa.effects.trim(sig, top_db=30)
    if len(trimmed) > 100:
        sig = trimmed
    S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024, hop_length=256)), ref=np.max)
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=110)
    librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.set_ylim(0, 4000); ax.set_title("STFT magnitude spectrogram (0-4 kHz, linear scale)", fontsize=9)
    fig.tight_layout()
    return fig


def analyse(wav, manual_text):
    if not wav:
        return {}, "", None, "Record or upload audio first."
    text = manual_text.strip() or None
    try:
        out = REC.predict(wav, transcript=text)
    except Exception as e:
        return {}, "", None, f"Error: {e}"
    src = "typed by user" if text else "Whisper ASR"
    note = (f"**{out['emotion'].upper()}** &nbsp;|&nbsp; transcript source: {src}\n\n"
            f"Confidence: {max(out['probs'].values())*100:.1f}%")
    return out["probs"], out["transcript"], spectrogram(out["signal"], out["sr"]), note


with gr.Blocks(title="Multimodal Emotion Recognition") as demo:
    gr.Markdown("# Multimodal Emotion Recognition\n"
                "Dual-stream **audio + text** fusion on IEMOCAP (angry / happy / neutral / sad).")
    gr.Markdown(INFO)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speech")
            manual = gr.Textbox(label="Transcript (optional)",
                                placeholder="Leave blank to let Whisper transcribe it",
                                info="Type the words here to bypass ASR - useful for showing "
                                     "how ASR errors affect the prediction (Gap 3).")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            verdict = gr.Markdown()
            probs = gr.Label(label="Emotion probabilities", num_top_classes=4)
            trans = gr.Textbox(label="Transcript used", interactive=False)
            spec = gr.Plot(label="Spectrogram")
    btn.click(analyse, [audio, manual], [probs, trans, spec, verdict])
    gr.Markdown("---\n*System B: TIM-Net (MFCC+GFCC) - System C: BiGRU text encoder - "
                "System D: cross/self-attention fusion - System E: this interface.*\n\n"
                "Note: at training time transcripts were ground truth; here they come from ASR. "
                "That mismatch is the ASR-robustness gap the project investigates.")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
