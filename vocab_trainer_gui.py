import json
import random
import tkinter as tk
from tkinter import messagebox

LANG_NAMES = {
    "es": "Spanish",
    "en": "English",
    "ru": "Russian",
}

OPTION_LABELS = ["A", "B", "C", "D"]


def load_vocab(path="vocab.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Filter out bad entries
    clean = []
    for item in data:
        if not isinstance(item, dict):
            continue
        es = item.get("es", "").strip()
        en = item.get("en", "").strip()
        ru = item.get("ru", "").strip()
        if es and en and ru:
            clean.append({"es": es, "en": en, "ru": ru})
    if not clean:
        raise ValueError("No valid vocab entries found in vocab.json")
    return clean


def get_available_languages(entry):
    return [lang for lang in LANG_NAMES.keys() if entry.get(lang, "").strip()]


def choose_source_target(entry, mode="mixed"):
    """
    mode:
      - 'mixed': any ordered pair of distinct languages present
      - 'es-en', 'es-ru', 'en-es', 'ru-es'
    """
    langs = get_available_languages(entry)
    if len(langs) < 2:
        return None, None

    if mode == "mixed":
        pairs = []
        for s in langs:
            for t in langs:
                if s != t:
                    pairs.append((s, t))
        if not pairs:
            return None, None
        return random.choice(pairs)

    # Fixed-direction modes
    mapping = {
        "es-en": ("es", "en"),
        "es-ru": ("es", "ru"),
        "en-es": ("en", "es"),
        "ru-es": ("ru", "es"),
    }
    if mode not in mapping:
        mode = "mixed"

    source, target = mapping.get(mode, (None, None))
    if source in langs and target in langs and source != target:
        return source, target

    # If entry doesn't support this pair, signal fail
    return None, None


def build_question(vocab, mode="mixed", num_options=4):
    """
    Returns: (entry, source_lang, target_lang, question_text, options_list, correct_index)
    """
    # Try a few times to find a compatible entry
    for _ in range(100):
        entry = random.choice(vocab)
        source, target = choose_source_target(entry, mode=mode)
        if source is None:
            continue

        source_word = entry[source]
        correct_answer = entry[target]

        # Distractors: from other entries' target language
        distractors = []
        candidates = [e for e in vocab if e is not entry and e.get(target, "").strip()]
        random.shuffle(candidates)

        for c in candidates:
            cand = c[target]
            if cand != correct_answer and cand not in distractors:
                distractors.append(cand)
            if len(distractors) >= (num_options - 1):
                break

        # If not enough distractors, allow duplicates from the pool (ugly but safe)
        while len(distractors) < (num_options - 1):
            cand = random.choice(vocab)[target]
            if cand != correct_answer:
                distractors.append(cand)

        options = [correct_answer] + distractors[: num_options - 1]
        random.shuffle(options)
        correct_index = options.index(correct_answer)

        question_text = (
            f"Translate from {LANG_NAMES[source]} to {LANG_NAMES[target]}:\n"
            f"   “{source_word}”"
        )
        return entry, source, target, question_text, options, correct_index

    raise RuntimeError("Could not build a question with current vocab and mode.")


class VocabTrainerApp:
    def __init__(self, master, vocab):
        self.master = master
        self.vocab = vocab
        self.mode = tk.StringVar(value="mixed")

        self.score = 0
        self.total = 0

        self.current_entry = None
        self.current_source = None
        self.current_target = None
        self.current_options = []
        self.correct_index = None

        self.selected_option = tk.IntVar(value=-1)

        self.build_ui()
        self.next_question()

    def build_ui(self):
        self.master.title("Spanish–English–Russian Vocab Trainer")

        # Top frame: mode selector + score
        top_frame = tk.Frame(self.master)
        top_frame.pack(padx=10, pady=10, fill="x")

        tk.Label(top_frame, text="Mode:").pack(side="left")
        modes = [
            ("Mixed", "mixed"),
            ("ES → EN", "es-en"),
            ("ES → RU", "es-ru"),
            ("EN → ES", "en-es"),
            ("RU → ES", "ru-es"),
        ]
        mode_menu = tk.OptionMenu(top_frame, self.mode, *[m[1] for m in modes])
        mode_menu.pack(side="left", padx=5)

        self.score_label = tk.Label(top_frame, text="Score: 0 / 0")
        self.score_label.pack(side="right")

        # Question label
        self.question_label = tk.Label(
            self.master, text="", justify="left", wraplength=500, font=("Arial", 12, "bold")
        )
        self.question_label.pack(padx=10, pady=(0, 10), anchor="w")

        # Options (radio buttons)
        self.option_buttons = []
        for i in range(4):
            rb = tk.Radiobutton(
                self.master,
                text="",
                variable=self.selected_option,
                value=i,
                anchor="w",
                justify="left",
                wraplength=500,
            )
            rb.pack(fill="x", padx=25, pady=2, anchor="w")
            self.option_buttons.append(rb)

        # Feedback
        self.feedback_label = tk.Label(self.master, text="", fg="blue", wraplength=500, justify="left")
        self.feedback_label.pack(padx=10, pady=(10, 5), anchor="w")

        # Buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(padx=10, pady=10)

        self.check_button = tk.Button(button_frame, text="Check Answer", command=self.check_answer)
        self.check_button.pack(side="left", padx=5)

        self.next_button = tk.Button(button_frame, text="Next", command=self.next_question)
        self.next_button.pack(side="left", padx=5)

    def next_question(self):
        self.feedback_label.config(text="")
        self.selected_option.set(-1)

        try:
            (
                self.current_entry,
                self.current_source,
                self.current_target,
                question_text,
                options,
                correct_index,
            ) = build_question(self.vocab, mode=self.mode.get(), num_options=4)
        except Exception as e:
            messagebox.showerror("Error", f"Could not create question:\n{e}")
            return

        self.current_options = options
        self.correct_index = correct_index

        self.question_label.config(text=question_text)

        for i, opt in enumerate(options):
            self.option_buttons[i].config(text=f"{OPTION_LABELS[i]}. {opt}", state="normal")
        # In case fewer than 4 in future variants, disable extras
        for j in range(len(options), 4):
            self.option_buttons[j].config(text="", state="disabled")

        self.check_button.config(state="normal")

    def check_answer(self):
        idx = self.selected_option.get()
        if idx < 0 or idx >= len(self.current_options):
            messagebox.showinfo("Choose an option", "Please select an answer first.")
            return

        self.total += 1
        if idx == self.correct_index:
            self.score += 1
            self.feedback_label.config(text="✅ Correct!", fg="green")
        else:
            correct_text = self.current_options[self.correct_index]
            self.feedback_label.config(
                text=f"❌ Incorrect.\nCorrect answer: {OPTION_LABELS[self.correct_index]}. {correct_text}",
                fg="red",
            )

        self.score_label.config(text=f"Score: {self.score} / {self.total}")
        self.check_button.config(state="disabled")


def main():
    try:
        vocab = load_vocab("vocab.json")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load vocab.json:\n{e}")
        return

    root = tk.Tk()
    app = VocabTrainerApp(root, vocab)
    root.mainloop()


if __name__ == "__main__":
    # For messagebox on startup errors
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
