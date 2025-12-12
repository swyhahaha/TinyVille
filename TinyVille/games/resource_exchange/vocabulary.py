import random
import string
from typing import Dict, List, Tuple


CONSONANTS = "ptkbdgmnszfvlr"
VOWELS = "aeiou"


class AlienVocabularyGenerator:
    """
    Generates a 19-word alien vocabulary (2-3 CV syllables) and maps to meanings.
    """

    MEANINGS = [
        "hello",
        "goodbye",
        "please",
        "thanks",
        "yes",
        "no",
        "and",
        "I",
        "you",
        "have",
        "want",
        "give",
        "meat",
        "grain",
        "water",
        "fruit",
        "fish",
        "much",
        "question",
    ]

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def _syllable(self) -> str:
        return f"{self.rng.choice(CONSONANTS)}{self.rng.choice(VOWELS)}"

    def _word(self) -> str:
        syllables = self.rng.choice([2, 3])
        return "".join(self._syllable() for _ in range(syllables))

    def generate(self) -> Dict[str, str]:
        words: List[str] = []
        while len(words) < len(self.MEANINGS):
            w = self._word()
            if w not in words:
                words.append(w)

        self.rng.shuffle(words)
        return {word: meaning for word, meaning in zip(words, self.MEANINGS)}

    def training_examples(self, vocab: Dict[str, str]) -> List[str]:
        """
        Provide loose example sentences (SVO bias, question word fronted, neg variants).
        """
        # pick helper tokens
        def tok(m):
            for k, v in vocab.items():
                if v == m:
                    return k
            return f"[{m}]"

        hello = tok("hello")
        give = tok("give")
        I = tok("I")
        you = tok("you")
        want = tok("want")
        question = tok("question")
        yes = tok("yes")
        no = tok("no")
        meat = tok("meat")
        water = tok("water")
        much = tok("much")

        examples = [
            f"{hello} {you}",  # greeting
            f"{I} {want} {meat}",  # SVO
            f"{question} {you} {give} {water}?",  # fronted wh
            f"{no} {give} {meat}",  # preverbal neg
            f"{give} {water} {no}",  # postverbal neg
            f"{I} {give} {you} {meat} {much}",  # extra modifier
        ]
        return examples

