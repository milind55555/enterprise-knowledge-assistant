import re

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those do does did can could should would will shall may might must i you we they he she
what which who whom how when where why my our your their me us them about into than then so such not no
per any each""".split())


def stem(tok: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9@.]+", text.lower().replace(",", ""))
    toks = [t.strip(".") for t in toks]
    return [stem(t) for t in toks if t and t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]
