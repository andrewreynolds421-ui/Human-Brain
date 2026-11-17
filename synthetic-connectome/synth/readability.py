"""Flesch-Kincaid grade with a rule-based syllable count (no dictionary download needed)."""
import re

def syllables(w):
    w = w.lower()
    if len(w) <= 3: return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w); w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))

def fk_grade(text):
    sents = [s for s in re.split(r"(?<=[.!?;:])\s+", text) if re.search(r"[A-Za-z]", s)]
    words = re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text)
    if not sents or not words: return None
    return 0.39 * len(words) / len(sents) + 11.8 * sum(map(syllables, words)) / len(words) - 15.59
