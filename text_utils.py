import re

_TOKEN = re.compile(r"[a-z']+")
_PLACEHOLDER = re.compile(r"\[(URL|EMAIL|IP|USER)\]")


def tokenize(text: str) -> list[str]:
    """Paprastas žodžių skaidymas: mažosios raidės, tik lotyniškos raidės ir apostrofai.

    Anonimizavimo žymės ([URL], [IP], ...) nelaikomos žodžiais.
    """
    text = _PLACEHOLDER.sub(" ", text)
    return [t.strip("'") for t in _TOKEN.findall(text.lower()) if len(t.strip("'")) > 1]
