import json
from pathlib import Path
from openai import OpenAI
import sys

# ---------- CONFIG ----------
MODEL = "gpt-4.1-mini"  # or "gpt-4o-mini", etc.
CORPUS_PATH = Path("corpus/master.txt")

DATA_DIR = Path("data")
VOCAB_PATH = DATA_DIR / "vocab.json"
PREP_PATH = DATA_DIR / "prepositions.json"
VERBS_PATH = DATA_DIR / "verbs.json"
GUSTAR_PATH = DATA_DIR / "gustar.json"
FUTURE_PATH = DATA_DIR / "future.json"
REFLEXIVE_PATH = DATA_DIR / "reflexive.json"
PREP_CONTRAST_PATH = DATA_DIR / "preposition_contrast.json"
CONTEXT_PATH = DATA_DIR / "context_vocab.json"
PRONOUNS_PATH = DATA_DIR / "pronouns.json"
COMPARISONS_PATH = DATA_DIR / "comparisons.json"

# ----------------------------

client = OpenAI()  # uses OPENAI_API_KEY (+ optionally OPENAI_PROJECT)


def load_corpus() -> str:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"{CORPUS_PATH} not found. Create it and paste your Spanish notes/corpus."
        )
    text = CORPUS_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{CORPUS_PATH} is empty.")
    return text


def call_openai(corpus: str) -> str:
    system_msg = (
        "You are a Spanish teacher and data engineer. "
        "Your job is to turn a mixed Spanish learning corpus into structured JSON for a quiz app.\n"
        "\n"
        "The user is learning Spanish with English and Russian as support languages.\n"
        "From the corpus, you must extract EIGHT kinds of items:\n"
        "\n"
        "1) vocab  (word-level vocabulary)\n"
        "   - JSON array under key 'vocab'\n"
        "   - Each item: {\"es\": <spanish word or short phrase>, "
        "\"en\": <english>, \"ru\": <russian>}\n"
        "   - Only include items where you are reasonably confident about EN & RU.\n"
        "   - Avoid duplicates (same Spanish or same EN/RU meaning).\n"
        "   - You MUST produce AT LEAST 80 vocab items if the corpus is long enough.\n"
        "   - You may freely create new short phrases using the same vocabulary and "
        "     grammar patterns, you do NOT need to copy sentences verbatim from the corpus.\n"
        "\n"
        "2) prepositions (multiple-choice questions about a/en/con/por/para/de/sin/sobre)\n"
        "   - JSON array under key 'prepositions'\n"
        "   - You MUST produce AT LEAST 25 items if the corpus is long enough.\n"
        "   - Each item:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"Voy ___ Madrid mañana.\",\n"
        "         \"correct\": \"a\",\n"
        "         \"options\": [\"a\", \"en\", \"por\", \"de\"],\n"
        "         \"explanation\": \"Short explanation in Spanish why 'a' is correct.\"\n"
        "       }\n"
        "   - The blank replaces exactly ONE of: a, en, con, por, para, de, sin, sobre.\n"
        "   - options must be 3–5 strings including the correct one.\n"
        "   - VERY IMPORTANT: in each preposition item there must be EXACTLY ONE correct "
        "     option for that sentence. All other options must be clearly incorrect "
        "     grammatically or semantically. Do NOT write sentences where two different "
        "     prepositions would both be natural.\n"
        "   - You may invent new sentences that fit the student's level and use the same "
        "     vocabulary and structures from the corpus.\n"
        "\n"
        "3) verbs (conjugation multiple-choice questions)\n"
        "   - JSON array under key 'verbs'\n"
        "   - You MUST produce AT LEAST 30 items if the corpus is long enough.\n"
        "   - Focus on high-value verbs present in or compatible with the corpus: ser, estar, ir, "
        "     tener, hacer, poder, querer, pensar, dormir, pedir, servir, decir, venir, and "
        "     reflexives like despertarse, ducharse, enojarse, etc.\n"
        "   - For now, only use presente (present indicative) and persons "
        "     yo / tú / él/ella / nosotros / ustedes/ellos.\n"
        "   - Each item:\n"
        "       {\n"
        "         \"infinitive\": \"pensar\",\n"
        "         \"tense\": \"presente\",\n"
        "         \"person\": \"yo\",\n"
        "         \"question\": \"Conjuga 'pensar' para 'yo' en presente.\",\n"
        "         \"correct\": \"pienso\",\n"
        "         \"options\": [\"pienso\", \"piensas\", \"pensé\", \"pensaba\"],\n"
        "         \"explanation\": \"Short Spanish explanation (e.g. e→ie).\"\n"
        "       }\n"
        "   - options must include the correct form.\n"
        "   - Distractors should be forms that are clearly wrong for that person/tense "
        "     (e.g. other persons of the same verb, other tenses, or similar irregular verbs), "
        "     not alternative correct answers.\n"
        "   - You may invent new prompts using the same verbs, but keep all Spanish at a "
        "     beginner / lower-intermediate level and consistent with the corpus.\n"
        "\n"
         "4) pronouns  (me/te/le/nos/les and possessives mi/mis, tu/tus, su/sus, nuestro/a/os/as)\n"
        "   - JSON array under key 'pronouns'\n"
        "   - You MUST produce AT LEAST 25 items if the corpus is long enough.\n"
        "   - Each item is a multiple-choice cloze question:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"____ gusta la música.\",\n"
        "         \"correct\": \"Me\",\n"
        "         \"options\": [\"Me\", \"Mi\", \"Yo\", \"Nos\"],\n"
        "         \"explanation\": \"Short explanation in Spanish why this pronoun is correct.\"\n"
        "       }\n"
        "   - Focus on:\n"
        "       * Pronombres de objeto indirecto: me, te, le, nos, les.\n"
        "       * Adjetivos posesivos átonos: mi/mis, tu/tus, su/sus, nuestro/nuestra/nuestros/nuestras.\n"
        "   - The blank MUST be exactly \"____\" and appear exactly once in the sentence.\n"
        "   - Options must be 3–5 strings including the correct one.\n"
        "   - Only ONE option should produce a natural, grammatically correct sentence.\n"
        "     The others should be wrong for person/number/gender or change the meaning\n"
        "     (e.g. confusing 'mi' vs 'me', or 'tu' vs 'te').\n"
        "   - Use simple present-tense sentences aligned with the learner's level in the corpus.\n"
        "   - Try to cover:\n"
        "       - me/te/le/nos/les before verbs (gustar-type or regular verbs),\n"
        "       - mi/mis, tu/tus, su/sus, nuestro/a/os/as before nouns.\n"
        "   - Do NOT include vosotros/vuestro forms.\n"
        "   - For each item, provide a short Spanish explanation mentioning person/number\n"
        "     agreement or the possessive pattern.\n"
        "\n"
        "5) future  (ir a + infinitivo / pensar + infinitivo / tener planes de + infinitivo)\n"
        "   - JSON array under key 'future'\n"
        "   - You MUST produce AT LEAST 20 items if the corpus is long enough.\n"
        "   - Use the structures that appear in the corpus:\n"
        "       - IR + A + INFINITIVO (voy a estudiar, vamos a viajar...)\n"
        "       - PENSAR + INFINITIVO (pienso estudiar, piensan salir...)\n"
        "       - TENER PLANES DE + INFINITIVO (tengo planes de viajar...)\n"
        "     combined with time expressions like: mañana, pasado mañana, la próxima semana,\n"
        "     el próximo año, dentro de dos días, esta noche, más tarde, etc.\n"
        "   - Each item:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"Mañana yo ____ visitar a mis abuelos.\",\n"
        "         \"correct\": \"voy a\",\n"
        "         \"options\": [\"voy a\", \"pienso\", \"tengo planes de\", \"fui a\"],\n"
        "         \"explanation\": \"Short Spanish explanation (why this structure is best for a future plan).\"\n"
        "       }\n"
        "   - The blank MUST be ONLY the future-structure chunk (e.g. 'voy a', 'pienso', "
        "     'tienen planes de'), not other parts of the sentence.\n"
        "   - sentence_with_blank MUST contain \"____\" EXACTLY ONCE.\n"
        "   - The correct phrase MUST NOT appear anywhere else in sentence_with_blank; "
        "     the only place where that exact phrase occurs should be as the completed "
        "     answer for \"____\".\n"
        "   - options must be 3–5 strings including the correct one.\n"
        "   - VERY IMPORTANT: each sentence must be written so that EXACTLY ONE option "
        "     produces a grammatically correct and semantically natural sentence.\n"
        "   - All distractors must be clearly wrong due to person, tense, or structure "
        "     (e.g. past tense instead of future, wrong subject agreement, or a construction "
        "     that does not match the sentence).\n"
        "   - CRITICAL RULE: do NOT create sentences where more than one of\n"
        "     \"voy a\", \"vas a\", \"va a\", \"vamos a\", \"van a\",\n"
        "     \"pienso\", \"piensas\", \"piensa\", \"pensamos\", \"piensan\",\n"
        "     or \"tengo/tienes/tiene/tenemos/tienen planes de\" could naturally fit\n"
        "     for the given subject.\n"
        "     For example, NO uses like:\n"
        "       \"¿Tú ____ trabajar este fin de semana?\" with options\n"
        "       [\"piensas\", \"vas a\", \"tienes planes de\", ...], because all\n"
        "       are natural future plans.\n"
        "   - In practice, most distractors should be:\n"
        "       * same verb but wrong person or wrong tense (pienso / piensas / pensaron...),\n"
        "       * same IR + A structure but wrong person or past form (voy a / vas a / fue a...),\n"
        "       * wrong fragment that breaks the fixed expression \"tener planes de\".\n"
        "   - A good pattern is: make the correct option the only one that is BOTH\n"
        "     grammatically and semantically valid for the specific subject and context,\n"
        "     and make the others clearly wrong (e.g. wrong tense, wrong person, or\n"
        "     broken expression).\n"
        "   - You may invent many examples that mix the student's known vocabulary with these\n"
        "     future structures, but always enforce the \"exactly one valid option\" rule.\n"
        "\n"
        "6) reflexive  (daily routine with reflexive verbs)\n"
        "   - JSON array under key 'reflexive'\n"
        "   - Use verbs like: despertarse, levantarse, ducharse, lavarse, ponerse, "
        "     quedarse, relajarse, arreglarse, acostarse, cepillarse, etc.\n"
        "   - Sentences should be simple daily-routine contexts, similar level as the corpus.\n"
        "   - Each item:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"Yo me ____ a las seis todos los días.\",\n"
        "         \"correct\": \"despierto\",\n"
        "         \"options\": [\"despierto\", \"despiertas\", \"despierta\", \"despertar\"],\n"
        "         \"explanation\": \"Short Spanish explanation about person/number and reflexive structure.\"\n"
        "       }\n"
        "   - sentence_with_blank MUST contain \"____\" EXACTLY ONCE.\n"
        "   - The blank MUST be a single verb form (no pronouns, only the conjugated verb).\n"
        "   - The corresponding reflexive pronoun (me, te, se, nos, se) MUST already be in the sentence\n"
        "     outside the blank and must agree with the subject.\n"
        "   - The correct form MUST NOT appear anywhere else in sentence_with_blank; the only place\n"
        "     where that form should exist is as the filled-in answer for \"____\".\n"
        "   - options must be 3–5 strings including the correct one.\n"
        "   - VERY IMPORTANT: exactly ONE option must produce a grammatically correct and natural\n"
        "     sentence. All other options must be clearly wrong by person, number or reflexive\n"
        "     agreement.\n"
        "   - Do NOT write prompts where two different verb forms could both work (e.g. where both\n"
        "     \"levanto\" and \"despierto\" could fit logically). If multiple verbs would make sense,\n"
        "     rewrite the sentence until only one option is clearly correct.\n"
        "\n"
        "7) preposition_contrast  (por/para, a/en, con/sin, de/sobre)\n"
        "   - JSON array under key 'preposition_contrast'\n"
        "   - Focus on contrasts:\n"
        "       - por vs para (cause vs purpose, duration vs deadline, exchange, etc.)\n"
        "       - a vs en (direction vs location, time expressions with 'a las ...' vs 'en verano', etc.)\n"
        "       - con vs sin\n"
        "       - de vs sobre (possession/origin/material vs topic/about)\n"
        "   - EVERY item MUST use the following schema (STRUCTURE IS NON-NEGOTIABLE):\n"
        "       {\n"
        "         \"sentence_with_blank\": \"Trabajo ____ la escuela ____ dos horas.\",\n"
        "         \"blanks\": [\n"
        "           {\n"
        "             \"options\": [\"para\", \"por\", \"con\", \"sin\"],\n"
        "             \"correct\": \"para\"\n"
        "           },\n"
        "           {\n"
        "             \"options\": [\"por\", \"para\", \"en\", \"sin\"],\n"
        "             \"correct\": \"por\"\n"
        "           }\n"
        "         ],\n"
        "         \"explanation\": \"Short Spanish explanation why each preposition is correct.\"\n"
        "       }\n"
        "   - The field \"sentence_with_blank\" MUST contain one \"____\" placeholder for EACH entry in\n"
        "     the \"blanks\" array. If there are 2 blanks in the sentence, there MUST be exactly\n"
        "     2 objects inside \"blanks\". If there is only 1 blank, then \"blanks\" MUST have length 1.\n"
        "   - You MAY create some items with only one blank, but they STILL MUST use the \"blanks\" array,\n"
        "     e.g.:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"Voy ____ Madrid mañana.\",\n"
        "         \"blanks\": [\n"
        "           {\n"
        "             \"options\": [\"a\", \"en\", \"por\", \"de\"],\n"
        "             \"correct\": \"a\"\n"
        "           }\n"
        "         ],\n"
        "         \"explanation\": \"Short explanation in Spanish why 'a' is correct.\"\n"
        "       }\n"
        "   - For each element of \"blanks\":\n"
        "       * \"options\" MUST be a flat list (NOT nested) of 3–5 strings.\n"
        "       * \"correct\" MUST be a single string that is one of the options in that list.\n"
        "       * Allowed prepositions: \"a\", \"en\", \"por\", \"para\", \"con\", \"sin\", \"de\", \"sobre\".\n"
        "   - DO NOT use any of the following shapes for preposition_contrast items:\n"
        "       * top-level \"correct\" as a list (e.g. [\"para\", \"por\"])\n"
        "       * top-level \"options\" as a list of lists (e.g. [[...],[...]])\n"
        "       * any extra fields not specified in the schema.\n"
        "   - The correct preposition for each blank MUST NOT appear anywhere else in\n"
        "     \"sentence_with_blank\" except in that blank position.\n"
        "   - For each blank, exactly ONE option must produce a grammatically and semantically\n"
        "     natural sentence. All other options must feel clearly wrong to a native speaker.\n"
        "   - Avoid prompts where more than one preposition could be reasonably used. If that happens,\n"
        "     rewrite the sentence or change distractors so there is only one truly correct option.\n"
        "   - BEFORE you finish your JSON, re-check the entire \"preposition_contrast\" array and fix\n"
        "     any item that does not follow this schema exactly. The final response MUST already be\n"
        "     corrected; do not mention your checking process.\n"
        "\n"

        "\n"
        "8) context_vocab  (fill-in-the-blank using the student's vocabulary in context)\n"
        "   - JSON array under key 'context_vocab'\n"
        "   - Use ONLY Spanish words and structures that are compatible with the supplied corpus:\n"
        "     basic present tense, simple future structures (ir a / pensar / tener planes de),\n"
        "     gustar-type verbs, parece, simple prepositions, questions, conjucations and the vocab fields given.\n"
        "   - For each item, create a SHORT Spanish sentence where exactly one content word (verb,\n"
        "     noun or adjective) is removed and replaced by \"____\".\n"
        "   - Each item:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"Mañana voy a ____ con mis amigos.\",\n"
        "         \"correct\": \"salir\",\n"
        "         \"options\": [\"salir\", \"comer\", \"estudiar\", \"dormir\"],\n"
        "         \"translation_en\": \"Tomorrow I'm going to go out with my friends.\",\n"
        "         \"translation_ru\": \"Завтра я собираюсь гулять с друзьями.\",\n"
        "         \"explanation\": \"Explain briefly why 'salir' is the natural choice here.\"\n"
        "       }\n"
        "   - sentence_with_blank MUST contain \"____\" EXACTLY ONCE.\n"
        "   - The correct word MUST NOT appear anywhere else in the sentence.\n"
        "   - options must be 3–5 Spanish words INCLUDING the correct one.\n"
        "   - All distractor options must be Spanish words that the student could plausibly know\n"
        "     at this level (no exotic C1 vocabulary). Prefer words already present in the corpus.\n"
        "   - VERY IMPORTANT: design sentences so that only ONE option yields a natural, idiomatic\n"
        "     sentence. If more than one option would make sense, rewrite the sentence.\n"
        "   - translation_en and translation_ru should be brief, literal-enough translations using\n"
        "     simple English and Russian.\n"
        "\n"
        "9) comparisons  (comparativos y superlativos: más que, menos que, más de, tan ... como)\n"
        "   - JSON array under key 'comparisons'\n"
        "   - You MUST produce AT LEAST 20 items if the corpus is long enough.\n"
        "   - Each item is a multiple-choice cloze:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"Mi casa es ____ grande que la tuya.\",\n"
        "         \"correct\": [\"más\"],\n"
        "         \"options\": [\"más\", \"menos\", \"tan\", \"muy\"],\n"
        "         \"explanation\": \"Breve explicación en español sobre por qué esta opción es correcta.\"\n"
        "       }\n"
        "\n"
        "   - SCHEMA RULES:\n"
        "       * \"sentence_with_blank\" MUST contain \"____\" EXACTLY ONCE.\n"
        "       * \"options\" must be a list of 3–5 strings.\n"
        "       * \"correct\" is ALWAYS a JSON array (list) of one or more strings.\n"
        "       * EVERY string in \"correct\" MUST appear in \"options\".\n"
        "\n"
        "   - SEMANTIC RULES:\n"
        "       * Usually there will be ONE clearly correct option (por ejemplo, 'más' en 'más ... que'),\n"
        "         so \"correct\" tendrá a menudo un solo elemento, por ejemplo [\"más\"].\n"
        "       * Si en alguna oración MÁS DE UNA opción sería español natural en ese hueco,\n"
        "         ENTONCES TODAS esas opciones deben incluirse en \"correct\".\n"
        "         No marques como incorrecta una forma que sea natural sólo para crear contraste.\n"
        "       * Debe haber por lo menos un distractor claramente incorrecto por forma o uso,\n"
        "         por ejemplo:\n"
        "           - Usar 'tan' donde la estructura pide 'más' o 'menos',\n"
        "           - Usar 'más que' en vez de 'más de' antes de un número,\n"
        "           - Usar 'muy' en una estructura comparativa con 'que'.\n"
        "\n"
        "   - ENFOQUE:\n"
        "       * Comparativos de superioridad: más + adjetivo + que (más grande que...).\n"
        "       * Comparativos de inferioridad: menos + adjetivo + que.\n"
        "       * Cantidad con números: más de / menos de + número (más de tres, menos de diez...).\n"
        "       * Igualdad: tan + adjetivo + como (tan alto como), y algunos con\n"
        "         tanto/a/os/as + sustantivo + como si aparecen en el corpus.\n"
        "       * Superlativos: el/la/los/las + más/menos + adjetivo + de\n"
        "         (el más alto de la clase, la menos cara del grupo).\n"
        "\n"
        "   - NUNCA generes:\n"
        "       * \"correct\": \"cadena_sola\" (siempre lista),\n"
        "       * Frases donde dos opciones sean igual de naturales y sólo una aparezca en \"correct\".\n"
        "         En esos casos, incluye todas las formas naturales en \"correct\".\n"
        "\n"

        "GLOBAL RULES:\n"
        "- You can and should create additional sentences that were NOT in the corpus, "
        "  as long as you mainly use verbs, prepositions, and vocabulary that appear in the "
        "  corpus or are very close to them.\n"
        "- Avoid rare or advanced vocabulary that the student has never seen.\n"
        "- Your entire response MUST be valid JSON.\n"
        "- Top-level object MUST have exactly these keys: "
        "vocab, prepositions, verbs, gustar, future, reflexive, preposition_contrast, context_vocab.\n"
        "- Example skeleton:\n"
        "  {\n"
        "    \"vocab\": [ ... ],\n"
        "    \"prepositions\": [ ... ],\n"
        "    \"verbs\": [ ... ],\n"
        "    \"gustar\": [ ... ],\n"
        "    \"future\": [ ... ],\n"
        "    \"reflexive\": [ ... ],\n"
        "    \"preposition_contrast\": [ ... ],\n"
        "    \"context_vocab\": [ ... ]\n"
        "  }\n"
        "- Do NOT include any extra commentary or text outside the JSON.\n"
    )

    user_msg = (
        "Here is the mixed Spanish learning corpus from the student. Extract as much useful data "
        "as you can into the required JSON structure.\n\n"
        "CORPUS:\n"
        f"{corpus}"
    )

    print("Requesting structured content from OpenAI...")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )

    content = resp.choices[0].message.content
    return content


def parse_master_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("OpenAI did not return valid JSON. Raw content:")
        print(raw)
        raise e

    if not isinstance(data, dict):
        raise ValueError(
            "Expected a JSON object with keys: "
            "vocab, prepositions, verbs, gustar, future, reflexive, "
            "preposition_contrast, context_vocab."
        )

    required_keys = [
        "vocab",
        "prepositions",
        "verbs",
        "gustar",
        "future",
        "reflexive",
        "preposition_contrast",
        "context_vocab",
        "pronouns",
        "comparisons",
    ]

    for key in required_keys:
        if key not in data or not isinstance(data[key], list):
            data[key] = []

    return data


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, items: list):
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items to {path}")


def main():
    # Which datasets to update, based on CLI args
    # Usage:
    #   python build_all.py           -> update all
    #   python build_all.py future    -> only future.json
    #   python build_all.py vocab future comparisons -> only those
    argv = sys.argv[1:]
    available = {
        "vocab": ("vocab", VOCAB_PATH),
        "prepositions": ("prepositions", PREP_PATH),
        "verbs": ("verbs", VERBS_PATH),
        "future": ("future", FUTURE_PATH),
        "reflexive": ("reflexive", REFLEXIVE_PATH),
        "preposition_contrast": ("preposition_contrast", PREP_CONTRAST_PATH),
        "context_vocab": ("context_vocab", CONTEXT_PATH),
        "comparisons": ("comparisons", PRONOUNS_PATH),
        "comparisons": ("comparisons", COMPARISONS_PATH),
    }

    if not argv or "all" in argv:
        selected_keys = list(available.keys())
    else:
        selected_keys = []
        for name in argv:
            if name not in available:
                print(f"[WARN] Unknown dataset name '{name}', skipping.")
            else:
                selected_keys.append(name)
        if not selected_keys:
            print("[INFO] No valid dataset names provided. Nothing to do.")
            return

    print(f"[INFO] Will update: {', '.join(selected_keys)}")

    corpus = load_corpus()
    raw = call_openai(corpus)
    master = parse_master_json(raw)

    ensure_data_dir()

    for cli_name in selected_keys:
        json_key, path = available[cli_name]
        items = master.get(json_key, [])
        save_json(path, items)

    print("\nDone. Updated datasets:")
    for cli_name in selected_keys:
        _, path = available[cli_name]
        print(f"  {path}")

if __name__ == "__main__":
    main()
