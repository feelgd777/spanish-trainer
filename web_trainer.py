from flask import Flask, render_template, request
import json
import random

app = Flask(__name__)

# ----- Load and clean vocab -----
with open("vocab.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

VOCAB = []
for item in raw:
    if not isinstance(item, dict):
        continue
    es = item.get("es", "").strip()
    en = item.get("en", "").strip()
    ru = item.get("ru", "").strip()
    if es and en and ru:
        VOCAB.append({"es": es, "en": en, "ru": ru})

if not VOCAB:
    raise RuntimeError("No valid vocab entries in vocab.json")

LANG_NAMES = {"es": "Spanish", "en": "English", "ru": "Russian"}
OPTION_LABELS = ["A", "B", "C", "D"]


def available_langs(entry):
    return [l for l in ("es", "en", "ru") if entry.get(l, "").strip()]


def choose_source_target(entry, mode="mixed"):
    langs = available_langs(entry)
    if len(langs) < 2:
        return None, None

    mapping = {
        "es-en": ("es", "en"),
        "es-ru": ("es", "ru"),
        "en-es": ("en", "es"),
        "ru-es": ("ru", "es"),
    }

    if mode == "mixed":
        s = random.choice(langs)
        t_candidates = [l for l in langs if l != s]
        if not t_candidates:
            return None, None
        t = random.choice(t_candidates)
        return s, t

    if mode in mapping:
        s, t = mapping[mode]
        if s in langs and t in langs and s != t:
            return s, t

    # fallback to mixed
    s = random.choice(langs)
    t_candidates = [l for l in langs if l != s]
    if not t_candidates:
        return None, None
    t = random.choice(t_candidates)
    return s, t


def build_question(mode="mixed", num_options=4):
    for _ in range(100):
        entry = random.choice(VOCAB)
        src, trg = choose_source_target(entry, mode)
        if src is None:
            continue

        source_word = entry[src]
        correct_answer = entry[trg]

        # Distractors from same target language
        distractors = []
        candidates = [e for e in VOCAB if e is not entry]
        random.shuffle(candidates)
        for c in candidates:
            cand = c[trg]
            if cand != correct_answer and cand not in distractors:
                distractors.append(cand)
            if len(distractors) >= (num_options - 1):
                break

        while len(distractors) < (num_options - 1):
            cand = random.choice(VOCAB)[trg]
            if cand != correct_answer:
                distractors.append(cand)

        options = [correct_answer] + distractors[: num_options - 1]
        random.shuffle(options)
        correct_index = options.index(correct_answer)

        return {
            "source_lang": src,
            "target_lang": trg,
            "source_word": source_word,
            "options": options,
            "correct_index": correct_index,
        }

    raise RuntimeError("Could not build a question.")


@app.route("/")
def index():
    mode = request.args.get("mode", "mixed")
    question = build_question(mode)

    return render_template(
        "index.html",
        question=question,
        mode=mode,
        src_name=LANG_NAMES[question["source_lang"]],
        trg_name=LANG_NAMES[question["target_lang"]],
        option_labels=OPTION_LABELS,
    )


if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
