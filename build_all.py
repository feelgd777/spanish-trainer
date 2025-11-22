import json
from pathlib import Path
from openai import OpenAI

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
        "4) gustar (gustar / encantar / interesar / parecer / aburrir / molestar / "
        "   fascinar / importar style structures)\n"
        "   - JSON array under key 'gustar'\n"
        "   - You MUST produce AT LEAST 20 items if the corpus is long enough.\n"
        "   - Use sentences at a similar level to the corpus, e.g. "
        "     \"A mis amigos les interesa aprender español y otros idiomas.\", "
        "     \"A nosotros nos parece una buena idea ir al cine los fines de semana porque hay películas divertidas.\", "
        "     \"A Ana le interesa aprender francés.\", "
        "     \"A nosotros nos parece una buena idea ir al cine los fines de semana porque hay películas divertidas.\", etc.\n"
        "   - Turn them into cloze questions where the main verb is BLANKED OUT with "
        "     exactly the placeholder \"____\" (four underscores).\n"
        "   - Each item:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"A mis amigos les ____ los animales.\",\n"
        "         \"correct\": \"gustan\",\n"
        "         \"options\": [\"gustan\", \"gusta\", \"encanta\", \"parece\"],\n"
        "         \"explanation\": \"Short Spanish explanation (e.g. plural subject 'los animales').\"\n"
        "       }\n"
        "   - Only blank out ONE verb per sentence.\n"
        "   - sentence_with_blank MUST contain \"____\" EXACTLY ONCE.\n"
        "   - VERY IMPORTANT: the correct verb form MUST NOT appear anywhere else "
        "     in sentence_with_blank. The only place where the correct form appears "
        "     should be as the filled-in answer for \"____\".\n"
        "   - options must be 3–5 strings including the correct one.\n"
        "   - VERY IMPORTANT: in each gustar-type item there must be EXACTLY ONE option "
        "     that produces a natural, grammatically correct sentence when substituted "
        "     into \"____\".\n"
        "   - All other options must be clearly incorrect either by person/number agreement "
        "     OR by meaning.\n"
        "   - CRITICAL RULE: do NOT create items where two different verbs would both be "
        "     natural in real Spanish. For example, avoid prompts where both "
        "     \"gusta\" and \"encanta\" or both \"interesa\" and \"parece\" could fit.\n"
        "   - In practice, this means:\n"
        "       * If the correct answer is a verb like \"interesa\", DO NOT include \"parece\" "
        "         or another near-synonym in options if it would also sound natural.\n"
        "       * If the correct answer is \"gusta\", DO NOT include \"encanta\" as an option "
        "         unless the sentence is written so that \"encanta\" would clearly sound wrong.\n"
        "       * Prefer distractors that are:\n"
        "           - wrong person/number of the same verb (gusta/gustan, encanta/encantan), or\n"
        "           - a completely inappropriate verb for that context.\n"
        "   - The explanation must NOT claim that a distractor is wrong if it would actually "
        "     be natural Spanish. Instead, you MUST design the sentence so that the distractors "
        "     are genuinely incorrect.\n"
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
        "   - CRITICAL RULE: do NOT create sentences where more than one of "
        "     \"voy a\", \"pienso\", \"tengo planes de\" (or similar) could naturally fit.\n"
        "     For example, avoid prompts like \"Tú ____ estudiar mañana\" with options "
        "     \"piensas\", \"vas a\", \"tienes planes de\", because all are natural ways "
        "     to express a future plan.\n"
        "   - If multiple future structures could fit a given sentence, you MUST rephrase the "
        "     sentence or choose different distractors so that only ONE option is truly correct.\n"
        "   - A good pattern is: make the correct option the only one that is BOTH grammatically "
        "     and semantically valid for the specific subject and context, and make the others "
        "     clearly wrong (e.g. wrong tense, wrong person, or mismatched structure).\n"
        "   - You may invent many examples that mix the student's known vocabulary with these "
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
        "   - Each item:\n"
        "       {\n"
        "         \"sentence_with_blank\": \"Trabajo ____ la escuela ____ dos horas.\",\n"
        "         \"correct\": [\"para\", \"por\"],\n"
        "         \"options\": [[\"para\", \"por\", \"con\", \"sin\"], [\"por\", \"para\", \"en\", \"sin\"]],\n"
        "         \"explanation\": \"Short Spanish explanation why each preposition is correct.\"\n"
        "       }\n"
        "   - The field is still called \"sentence_with_blank\", but it may contain ONE or TWO\n"
        "     occurrences of the placeholder \"____\".\n"
        "   - If there is only ONE blank:\n"
        "       * sentence_with_blank contains \"____\" exactly once.\n"
        "       * \"correct\" MUST be a single string (e.g. \"a\").\n"
        "       * \"options\" MUST be a flat list of strings (e.g. [\"a\", \"en\", \"por\", \"para\"]).\n"
        "   - If there are TWO blanks:\n"
        "       * sentence_with_blank MUST contain \"____\" exactly twice.\n"
        "       * \"correct\" MUST be a list of TWO strings, in order of the blanks\n"
        "         (e.g. [\"para\", \"por\"]).\n"
        "       * \"options\" MUST be a list of TWO lists of strings, one list per blank, in order.\n"
        "   - In all cases, each blank corresponds to exactly ONE correct preposition.\n"
        "   - Each blank MUST be a single preposition: \"a\", \"en\", \"por\", \"para\", \"con\", \"sin\",\n"
        "     \"de\" or \"sobre\".\n"
        "   - The correct preposition for a given blank MUST NOT appear anywhere else in\n"
        "     sentence_with_blank.\n"
        "   - For each blank, its options list must contain 3–5 strings including the correct one.\n"
        "   - VERY IMPORTANT: for each blank, exactly ONE option must be grammatically and\n"
        "     semantically natural. All other options must sound clearly wrong to a native speaker.\n"
        "   - Avoid prompts where more than one preposition could be arguable in real use for a\n"
        "     given blank. If multiple prepositions could fit, rewrite the sentence.\n"
        "   - CRITICAL RULE: when you use TWO blanks, they must genuinely be two independent\n"
        "     prepositional slots (like \"para la escuela\" vs \"por dos horas\"), and you MUST\n"
        "     provide separate correct answers and options for EACH blank.\n"
        "   - Do NOT mix the roles: do not design sentences that conceptually need two different\n"
        "     prepositions but then try to force a single correct answer string for both at once.\n"
        "\n"
        "8) context_vocab  (fill-in-the-blank using the student's vocabulary in context)\n"
        "   - JSON array under key 'context_vocab'\n"
        "   - Use ONLY Spanish words and structures that are compatible with the supplied corpus:\n"
        "     basic present tense, simple future structures (ir a / pensar / tener planes de),\n"
        "     gustar-type verbs, simple prepositions, and the vocab fields given.\n"
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
    corpus = load_corpus()
    raw = call_openai(corpus)
    master = parse_master_json(raw)

    ensure_data_dir()
    save_json(VOCAB_PATH, master["vocab"])
    save_json(PREP_PATH, master["prepositions"])
    save_json(VERBS_PATH, master["verbs"])
    save_json(GUSTAR_PATH, master["gustar"])
    save_json(FUTURE_PATH, master["future"])
    save_json(REFLEXIVE_PATH, master["reflexive"])
    save_json(PREP_CONTRAST_PATH, master["preposition_contrast"])
    save_json(CONTEXT_PATH, master["context_vocab"])

    print(
        "Done. Now your app can load:\n"
        "  data/vocab.json\n"
        "  data/prepositions.json\n"
        "  data/verbs.json\n"
        "  data/gustar.json\n"
        "  data/future.json\n"
        "  data/reflexive.json\n"
        "  data/preposition_contrast.json\n"
        "  data/context_vocab.json\n"
    )


if __name__ == "__main__":
    main()
