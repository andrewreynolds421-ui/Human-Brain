"""McGuffey's First Eclectic Reader (revised edition, 1879; Project Gutenberg #14640, public domain) as a
reading curriculum: lessons in order, each with its sentences and the description of its picture.

Words are lowercased for identity; `shown` keeps the printed capitals.  Apostrophes and hyphens are dropped
("kitty's" -> "kittys"); punctuation is not read.
"""
import os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIRST = os.path.join(ROOT, "data", "primer", "mcguffey_first_reader.txt")


def lessons(path=FIRST):
    txt = open(path, encoding="latin-1").read()
    a = txt.find("LESSON I."); b = txt.find("End of the Project Gutenberg")
    if b < 0:
        b = txt.find("End of Project Gutenberg")
    body = txt[a:b]
    out = []
    for chunk in re.split(r"\n\s*LESSON [IVXLC]+\.[^\n]*\n", "\n" + body)[1:]:
        pics = [re.sub(r"\s+", " ", p) for p in re.findall(r"\[Illustration:?\s*([^\]]*)\]", chunk)]
        chunk = re.sub(r"\[Illustration[^\]]*\]", " ", chunk)
        sents = []
        for para in re.split(r"\n\s*\n", chunk):
            lines = [l for l in para.split("\n") if l.strip()]
            if not lines or any(re.search(r"\S\s{2,}\S", l.strip()) for l in lines):      # word / sound lists
                continue
            text = " ".join(l.strip() for l in lines)
            if not re.search(r"[.!?]", text) or text.isupper():
                continue
            for s in re.split(r"(?<=[.!?])\s+", text):
                toks = [re.sub(r"[^A-Za-z]", "", t) for t in s.replace("-", " ").split()]
                toks = [t for t in toks if t]
                if toks:
                    sents.append(toks)
        if sents:
            out.append(dict(pictures=pics, sentences=sents))
    return out
