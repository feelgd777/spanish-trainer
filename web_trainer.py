import json
import random
from pathlib import Path

from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__, static_folder=".", static_url_path="")

# ---------- HELPERS TO LOAD DATA ----------

def load_json(name):
    path = DATA_DIR / name
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[WARN] {name} is not a list, ignoring")
            return []
        print(f"[INFO] Loaded {len(data)} items from {name}")
        return data
    except FileNotFoundError:
        print(f"[WARN] {name} not found in data/")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load {name}: {e}")
        return []


vocab_items = load_json("vocab.json")
prep_items = load_json("prepositions.json")
verbs_items = load_json("verbs.json")
pronouns_items = load_json("pronouns.json")
future_items = load_json("future.json")
reflexive_items = load_json("reflexive.json")
prep_contrast_items = load_json("preposition_contrast.json")
context_vocab_items = load_json("context_vocab.json")
comparisons_items = load_json("comparisons.json")



# ---------- GENERIC MC-QUESTION HELPERS ----------

def make_simple_mc_question(items, field_sentence="sentence_with_blank"):
    """
    For standard MC questions with:
      - sentence_with_blank
      - options
      - correct  (string OR list of strings)
      - explanation (optional)
    """
    usable = [
        it for it in items
        if isinstance(it, dict)
        and it.get(field_sentence)
        and isinstance(it.get("options"), list)
        and it.get("options")
    ]
    if not usable:
        raise ValueError("No usable items for this mode.")

    raw = random.choice(usable)
    options = raw["options"]
    correct = raw.get("correct")

    correct_indices = []

    # Case 1: correct is a list of option-texts
    if isinstance(correct, list):
        for c in correct:
            if c in options:
                idx = options.index(c)
                if idx not in correct_indices:
                    correct_indices.append(idx)

    # Case 2: correct is a single option-text
    elif isinstance(correct, str) and correct in options:
        correct_indices.append(options.index(correct))

    # Case 3: fall back to numeric correct_index if provided
    if not correct_indices and "correct_index" in raw:
        ci = raw["correct_index"]
        if isinstance(ci, list):
            for v in ci:
                try:
                    i = int(v)
                    if 0 <= i < len(options) and i not in correct_indices:
                        correct_indices.append(i)
                except (TypeError, ValueError):
                    pass
        else:
            try:
                i = int(ci)
                if 0 <= i < len(options):
                    correct_indices.append(i)
            except (TypeError, ValueError):
                pass

    # Case 4: ultimate fallback – first option
    if not correct_indices:
        correct_indices = [0]

    # For backward compatibility keep a single correct_index too
    primary_correct_index = correct_indices[0]

    return {
        "question": raw[field_sentence],
        "options": options,
        "correct_index": primary_correct_index,
        "correct_indices": correct_indices,
        "explanation": raw.get("explanation", ""),
    }


# ---------- MODE: VOCAB ----------

def make_vocab_question(direction):
    """
    direction: es_en, en_es, es_ru, ru_es
    Uses vocab_items: {es, en, ru}
    """
    usable = [
        it for it in vocab_items
        if isinstance(it, dict)
        and it.get("es")
        and (it.get("en") or it.get("ru"))
    ]
    if not usable:
        raise ValueError("No vocab items available.")

    item = random.choice(usable)

    # Decide source and target language fields
    if direction == "es_en":
        prompt_lang = "ES"
        prompt_value = item["es"]
        target_lang = "EN"
        correct_answer = item.get("en", "")
        distract_pool = [i.get("en") for i in usable if i is not item and i.get("en")]
    elif direction == "en_es":
        prompt_lang = "EN"
        prompt_value = item["en"]
        target_lang = "ES"
        correct_answer = item.get("es", "")
        distract_pool = [i.get("es") for i in usable if i is not item and i.get("es")]
    elif direction == "es_ru":
        prompt_lang = "ES"
        prompt_value = item["es"]
        target_lang = "RU"
        correct_answer = item.get("ru", "")
        distract_pool = [i.get("ru") for i in usable if i is not item and i.get("ru")]
    elif direction == "ru_es":
        prompt_lang = "RU"
        prompt_value = item["ru"]
        target_lang = "ES"
        correct_answer = item.get("es", "")
        distract_pool = [i.get("es") for i in usable if i is not item and i.get("es")]
    else:
        # Default
        prompt_lang = "ES"
        prompt_value = item["es"]
        target_lang = "EN"
        correct_answer = item.get("en", "")
        distract_pool = [i.get("en") for i in usable if i is not item and i.get("en")]

    if not prompt_value or not correct_answer:
        # If something missing, just retry
        return make_vocab_question("es_en")

    # Build options
    distract_pool = list({d for d in distract_pool if d and d != correct_answer})
    random.shuffle(distract_pool)
    distractors = distract_pool[:3]  # up to 3 distractors
    options = [correct_answer] + distractors
    random.shuffle(options)
    correct_index = options.index(correct_answer)

    question_text = f"Traduce ({prompt_lang} → {target_lang}): {prompt_value}"

    return {
        "question": question_text,
        "options": options,
        "correct_index": correct_index,
        "explanation": "",
    }


# ---------- MODE: VERBS (conjugations) ----------

def make_verbs_question():
    usable = [
        it for it in verbs_items
        if isinstance(it, dict)
        and (it.get("question") or it.get("infinitive"))
        and isinstance(it.get("options"), list)
        and it.get("options")
    ]
    if not usable:
        raise ValueError("No verbs items available.")

    raw = random.choice(usable)
    options = raw["options"]
    correct = raw.get("correct")

    if raw.get("question"):
        question_text = raw["question"]
    else:
        infinitive = raw.get("infinitive", "???")
        tense = raw.get("tense", "presente")
        person = raw.get("person", "")
        question_text = f"Conjuga '{infinitive}' para '{person}' en {tense}."

    if correct in options:
        correct_index = options.index(correct)
    else:
        correct_index = 0

    return {
        "question": question_text,
        "options": options,
        "correct_index": correct_index,
        "explanation": raw.get("explanation", ""),
    }




# ---------- MODE: FUTURE ----------

def make_future_question():
    return make_simple_mc_question(future_items, field_sentence="sentence_with_blank")


# ---------- MODE: REFLEXIVE ----------

def make_reflexive_question():
    return make_simple_mc_question(reflexive_items, field_sentence="sentence_with_blank")

# ---------- MODE: PRONOUNS (me/te/le, mi/mis, tu/tus, su/sus, nuestros) ----------

def make_pronouns_question():
    return make_simple_mc_question(pronouns_items, field_sentence="sentence_with_blank")

# ---------- MODE: PREPOSITIONS (BASIC) ----------

def make_prepositions_question():
    return make_simple_mc_question(prep_items, field_sentence="sentence_with_blank")


# ---------- MODE: PREPOSITION CONTRAST (multi-blank support) ----------

def make_preposition_contrast_question():
    """
    Supports two shapes:
      1) Simple (legacy):
         { sentence_with_blank, options, correct, explanation }

      2) Multi-blank:
         {
           "sentence_with_blank": "Trabajo ____ la escuela ____ dos horas.",
           "blanks": [
              {"options": [...], "correct": "para"},
              {"options": [...], "correct": "por"}
           ],
           "explanation": "..."
         }

    If 'blanks' exists and is non-empty, we return 'blanks' for the frontend
    to handle per-blank buttons (your JS already knows how to handle q.blanks).
    Otherwise, we fall back to simple MC question.
    """
    usable = [it for it in prep_contrast_items if isinstance(it, dict)]
    if not usable:
        raise ValueError("No preposition_contrast items available.")

    raw = random.choice(usable)

    # Multi-blank path
    if isinstance(raw.get("blanks"), list) and raw["blanks"]:
        blanks_payload = []
        for b in raw["blanks"]:
            if not isinstance(b, dict):
                continue
            opts = b.get("options") or raw.get("options") or []
            if not opts:
                continue
            correct = b.get("correct")
            if "correct_index" in b and isinstance(b["correct_index"], int):
                ci = b["correct_index"]
                if not 0 <= ci < len(opts):
                    ci = 0
            else:
                if correct in opts:
                    ci = opts.index(correct)
                else:
                    ci = 0
            blanks_payload.append(
                {
                    "options": opts,
                    "correct_index": ci,
                }
            )

        if blanks_payload:
            return {
                "question": raw.get("sentence_with_blank", ""),
                "blanks": blanks_payload,
                "explanation": raw.get("explanation", ""),
            }

    # Fallback: treat as simple MC
    return make_simple_mc_question(prep_contrast_items, field_sentence="sentence_with_blank")


# ---------- MODE: CONTEXT VOCAB ----------

def make_context_vocab_question():
    usable = [
        it for it in context_vocab_items
        if isinstance(it, dict)
        and it.get("sentence_with_blank")
        and isinstance(it.get("options"), list)
        and it.get("options")
    ]
    if not usable:
        raise ValueError("No context_vocab items available.")

    raw = random.choice(usable)
    options = raw["options"]
    correct = raw.get("correct")

    if correct in options:
        correct_index = options.index(correct)
    else:
        correct_index = 0

    explanation_parts = []
    if raw.get("explanation"):
        explanation_parts.append(raw["explanation"])
    if raw.get("translation_en"):
        explanation_parts.append(f"EN: {raw['translation_en']}")
    if raw.get("translation_ru"):
        explanation_parts.append(f"RU: {raw['translation_ru']}")

    explanation = "\n\n".join(explanation_parts).strip()

    return {
        "question": raw["sentence_with_blank"],
        "options": options,
        "correct_index": correct_index,
        "explanation": explanation,
    }
# ---------- MODE: COMPARISONS (más que / menos que / más de / tan ... como / superlativos) ----------

def make_comparisons_question():
    # Standard MC: one blank, one correct form
    return make_simple_mc_question(comparisons_items, field_sentence="sentence_with_blank")


# ---------- ROUTES ----------

@app.route("/")
def index():
    # Serve index.html from the project root (your current file)
    return app.send_static_file("index.html")


@app.route("/api/question")
def api_question():
    mode = request.args.get("mode", "vocab")
    direction = request.args.get("direction", "es_en")

    try:
        if mode == "vocab":
            q = make_vocab_question(direction)
        elif mode == "prepositions":
            q = make_prepositions_question()
        elif mode == "preposition_contrast":
            q = make_preposition_contrast_question()
        elif mode == "verbs":
            q = make_verbs_question()
        elif mode == "future":
            q = make_future_question()
        elif mode == "reflexive":
            q = make_reflexive_question()
        elif mode == "context_vocab":
            q = make_context_vocab_question()
        elif mode == "pronouns":
            q = make_pronouns_question()
        elif mode == "comparisons":
            q = make_comparisons_question()
 
        else:
            return jsonify({"error": f"Unknown mode {mode}"}), 400

        return jsonify(q)

    except Exception as e:
        # Very important: we don't crash silently; you see the error in logs.
        print(f"[ERROR] /api/question failed for mode={mode}: {e}")
        return jsonify({"error": "internal server error"}), 500


if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=5000, debug=True)
