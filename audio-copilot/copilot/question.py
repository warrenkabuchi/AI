import re

_INTERROGATIVES = {
    "what", "why", "how", "when", "where", "who", "which",
    "can", "could", "would", "should", "do", "does", "did",
    "is", "are", "will", "explain", "describe", "tell",
}
_PHRASES = ("can you", "could you", "walk me through", "tell me about",
            "what about", "talk about")


def looks_like_question(text):
    t = text.strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    words = re.findall(r"[a-z']+", t)
    if words and words[0] in _INTERROGATIVES and len(words) >= 3:
        return True
    return any(p in t for p in _PHRASES)
