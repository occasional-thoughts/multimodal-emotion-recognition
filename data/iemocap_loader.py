"""Parse IEMOCAP into a clean utterance table: wav path, emotion, speaker, session, transcript.

IEMOCAP layout:
  Session{1..5}/dialog/EmoEvaluation/*.txt   <- emotion labels
  Session{1..5}/dialog/transcriptions/*.txt  <- text
  Session{1..5}/sentences/wav/<dialog>/*.wav <- per-utterance audio
"""
import os, re, glob, sys, csv
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/IEMOCAP_full_release")

# [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
LINE = re.compile(r"^\[([\d.]+) - ([\d.]+)\]\s+(\S+)\s+(\w+)\s+\[([-\d., ]+)\]")

# standard 4-class protocol: excited is merged into happy
MERGE = {"ang": "angry", "hap": "happy", "exc": "happy", "neu": "neutral", "sad": "sad"}

def parse():
    rows, raw_counts = [], Counter()
    for ses in range(1, 6):
        emo_dir = os.path.join(ROOT, f"Session{ses}", "dialog", "EmoEvaluation")
        if not os.path.isdir(emo_dir):
            print(f"  ! missing {emo_dir}"); continue

        # transcripts for this session
        trans = {}
        for tf in glob.glob(os.path.join(ROOT, f"Session{ses}", "dialog", "transcriptions", "*.txt")):
            for ln in open(tf, encoding="utf-8", errors="ignore"):
                if ":" in ln and ln.startswith("Ses"):
                    uid, txt = ln.split(":", 1)
                    trans[uid.split(" ")[0].strip()] = txt.strip()

        for f in sorted(glob.glob(os.path.join(emo_dir, "*.txt"))):
            dialog = os.path.basename(f)[:-4]
            for ln in open(f, encoding="utf-8", errors="ignore"):
                m = LINE.match(ln)
                if not m: continue
                start, end, uid, emo, vad = m.groups()
                raw_counts[emo] += 1
                if emo not in MERGE: continue
                wav = os.path.join(ROOT, f"Session{ses}", "sentences", "wav", dialog, uid + ".wav")
                if not os.path.exists(wav): continue
                v, a, d = [float(x) for x in vad.split(",")]
                # speaker = session + gender letter of THIS utterance (10 speakers total)
                g = uid.split("_")[-1][0]
                rows.append({
                    "utt_id": uid, "wav": wav, "emotion": MERGE[emo], "raw_emotion": emo,
                    "speaker": f"Ses{ses:02d}{g}", "session": ses,
                    "start": float(start), "end": float(end), "dur": round(float(end)-float(start), 3),
                    "valence": v, "arousal": a, "dominance": d,
                    "transcript": trans.get(uid, ""),
                })
    return rows, raw_counts

rows, raw = parse()
if not rows:
    print("No utterances parsed — is extraction finished? Check ROOT:", ROOT); sys.exit(1)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iemocap_index.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"\nparsed {len(rows)} utterances (4-class) -> {out}\n")
print("all raw labels found in corpus:")
for k, v in raw.most_common(): print(f"   {k:5} {v:6}")
print("\n4-class distribution (exc merged into hap):")
for k, v in Counter(r["emotion"] for r in rows).most_common(): print(f"   {k:9} {v:5}")
print("\nper session:")
for s in sorted(set(r["session"] for r in rows)):
    sub = [r for r in rows if r["session"] == s]
    print(f"   Session{s}: {len(sub):5} utts, speakers {sorted(set(r['speaker'] for r in sub))}")
tot = sum(r["dur"] for r in rows)
withtxt = sum(1 for r in rows if r["transcript"])
print(f"\ntotal audio: {tot/3600:.2f} h   mean utt {tot/len(rows):.2f}s")
print(f"transcripts matched: {withtxt}/{len(rows)}")
