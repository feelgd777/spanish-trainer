import json
import random
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

DATA_DIR = Path("data")


def load_json(name: str):
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


DATA = {
    "vocab": load_json("vocab"),
    "prepositions": load_json("prepositions"),
    "verbs": load_json("verbs"),
    "gustar": load_json("gustar"),
    "future": load_json("future"),
    "reflexive": load_json("reflexive"),
    "preposition_contrast": load_json("preposition_contrast"),
    "context_vocab": load_json("context_vocab"),
}


@app.route("/")
def index():
    # Serve your index.html
    return send_from_directory(".", "index.html")


# ---------- QUESTION BUILDERS ----------

def make_vocab_question(direction: str):
    items = DATA.get("vocab", [])
    if not items:
        return {"question": "No hay vocabulario.", "options": ["—"], "correct_index": 0}

    item = random.choice(items)
    es = item.get("es", "")
    en = item.get("en", "")
    ru = item.get("ru", "")

    # Directions:
    # es_en: show ES, options with EN
    # en_es: show EN, options with ES
    # es_ru: show ES, options with RU
    # ru_es: show RU, options with ES

    # Build pool for distractors from all items at same language side
    pool = [x for x in items if x is not item]

    if direction == "en_es":
        prompt = f"{en} → español"
        correct = es
        candidates = [x.get("es", "") for x in pool if x.get("es")]
    elif direction == "es_en":
        prompt = f"{es} → inglés"
        correct = en
        candidates = [x.get("en", "") for x in pool if x.get("en")]
    elif direction == "es_ru":
        prompt = f"{es} → ruso"
        correct = ru
        candidates = [x.get("ru", "") for x in pool if x.get("ru")]
    elif direction == "ru_es":
        prompt = f"{ru} → español"
        correct = es
        candidates = [x.get("es", "") for x in pool if x.get("es")]
    else:
        prompt = f"{es} → inglés"
        correct = en
        candidates = [x.get("en", "") for x in pool if x.get("en")]

    # Fallback if something missing
    if not correct:
        correct = en or es
    if not candidates:
        candidates = [correct]

    distractors = random.sample(candidates, k=min(3, len(candidates)))
    options = [correct] + distractors
    random.shuffle(options)
    correct_index = options.index(correct)

    return {
        "question": prompt,
        "options": options,
        "correct_index": correct_index,
        "explanation": "",
    }


def make_simple_cloze_question(mode: str):
    """
    For modes where JSON items look like:
    { "sentence_with_blank": "...____...", "correct": "...", "options": [...], "explanation": "..." }
    and only ONE blank.
    """
    items = DATA.get(mode, [])
    if not items:
        return {"question": f"No hay datos para {mode}.", "options": ["—"], "correct_index": 0}

    item = random.choice(items)
    sent = item.get("sentence_with_blank") or item.get("sentence") or ""
    correct = item.get("correct")
    options = item.get("options") or []
    explanation = item.get("explanation") or ""

    if not options and isinstance(correct, str):
        options = [correct]

    # Ensure correct in options
    if isinstance(correct, str) and correct not in options:
        options = [correct] + [o for o in options if o != correct]

    # Remove duplicates
    options = list(dict.fromkeys(options))
    if not options:
        options = [correct or ""]

    correct_index = 0
    if isinstance(correct, str) and correct in options:
        correct_index = options.index(correct)

    return {
        "question": sent or "Pregunta sin texto.",
        "options": options,
        "correct_index": correct_index,
        "explanation": explanation,
    }


def make_verbs_question():
    items = DATA.get("verbs", [])
    if not items:
        return {"question": "No hay datos de verbos.", "options": ["—"], "correct_index": 0}

    item = random.choice(items)
    question = item.get("question") or "Conjuga el verbo."
    correct = item.get("correct")
    options = item.get("options") or []
    explanation = item.get("explanation") or ""

    if isinstance(correct, str) and correct not in options:
        options = [correct] + [o for o in options if o != correct]

    options = list(dict.fromkeys(options))
    if not options:
        options = [correct or ""]

    correct_index = 0
    if isinstance(correct, str) and correct in options:
        correct_index = options.index(correct)

    return {
        "question": question,
        "options": options,
        "correct_index": correct_index,
        "explanation": explanation,
    }


def make_preposition_contrast_question():
    """
    Returns either:
    - single-blank format:
        {
          "question": <str>,
          "options": [..],
          "correct_index": int,
          "explanation": <str>
        }
    - or multi-blank format (only if JSON is well-formed):
        {
          "question": <str>,
          "blanks": [
            {"options": [..], "correct_index": int},
            {"options": [..], "correct_index": int}
          ],
          "explanation": <str>
        }

    If an item has multiple '____' but 'correct' is NOT a list
    with the same length, we SKIP it and pick another item.
    """
    import random

    # safety loop so we can skip bad items without crashing
    for _ in range(50):
        item = random.choice(PREP_CONTRAST)
        sent = item.get("sentence_with_blank", "")
        options = item.get("options", [])
        correct = item.get("correct")

        # sanitize options
        options = [o for o in options if isinstance(o, str)]
        if not options:
            options = ["a", "en", "por", "para"]

        blanks_count = sent.count("____")

        # --- MULTI-BLANK PATH ---
        if blanks_count > 1:
            # we only accept it if 'correct' is a list with one entry per blank
            if isinstance(correct, list) and len(correct) == blanks_count:
                blanks = []
                for c in correct:
                    # find or insert each correct option
                    if c not in options:
                        options = [c] + [o for o in options if o != c]
                    idx = options.index(c)
                    blanks.append(
                        {
                            "options": options,
                            "correct_index": idx,
                        }
                    )

                return {
                    "question": sent,
                    "blanks": blanks,
                    "explanation": item.get("explanation", ""),
                }

            # malformed multi-blank item → skip and try another
            continue

        # --- SINGLE-BLANK PATH ---
        if isinstance(correct, str):
            if correct not in options:
                options = [correct] + [o for o in options if o != correct]
            correct_index = options.index(correct)
            return {
                "question": sent,
                "options": options,
                "correct_index": correct_index,
                "explanation": item.get("explanation", ""),
            }

        # malformed single-blank → skip
        continue

    # fallback if everything is garbage
    return {
        "question": "Voy ____ Madrid mañana.",
        "options": ["a", "en", "por", "para"],
        "correct_index": 0,
        "explanation": "Ejemplo de emergencia: 'a' indica dirección.",
    }


def make_context_vocab_question():
    items = DATA.get("context_vocab", [])
    if not items:
        return {
            "question": "No hay datos para vocabulario en contexto.",
            "options": ["—"],
            "correct_index": 0,
            "explanation": "",
        }

    item = random.choice(items)
    sent = item.get("sentence_with_blank") or ""
    correct = item.get("correct")
    options = item.get("options") or []
    explanation = item.get("explanation") or ""
    tr_en = item.get("translation_en") or ""
    tr_ru = item.get("translation_ru") or ""

    if isinstance(correct, str) and correct not in options:
        options = [correct] + [o for o in options if o != correct]
    options = list(dict.fromkeys(options))
    if not options:
        options = [correct or ""]

    correct_index = 0
    if isinstance(correct, str) and correct in options:
        correct_index = options.index(correct)

    full_expl = explanation
    if tr_en or tr_ru:
        extra = []
        if tr_en:
            extra.append(f"EN: {tr_en}")
        if tr_ru:
            extra.append(f"RU: {tr_ru}")
        if full_expl:
            full_expl = full_expl + "\n\n" + "\n".join(extra)
        else:
            full_expl = "\n".join(extra)

    return {
        "question": sent or "Pregunta sin texto.",
        "options": options,
        "correct_index": correct_index,
        "explanation": full_expl,
    }


# ---------- API ----------

@app.route("/api/question")
def api_question():
    mode = request.args.get("mode", "vocab")
    direction = request.args.get("direction", "es_en")

    if mode == "vocab":
        q = make_vocab_question(direction=direction)
    elif mode == "prepositions":
        q = make_prepositions_question()
    elif mode == "preposition_contrast":
        q = make_preposition_contrast_question()
    elif mode == "verbs":
        q = make_verbs_question()
    elif mode == "gustar":
        q = make_gustar_question()
    elif mode == "future":
        q = make_future_question()
    elif mode == "reflexive":
        q = make_reflexive_question()
    elif mode == "context_vocab":
        q = make_context_vocab_question()
    else:
        q = {"question": "Modo desconocido", "options": [], "correct_index": 0}

    return jsonify(q)



if __name__ == "__main__":
    # For local testing. On Render you'll use gunicorn web_trainer:app
    app.run(host="0.0.0.0", port=8000, debug=True)
