"""Official Photo Desk UI translations from PhotoDeskUIStrings.u16.

The file is a UTF-16 tab separated table:
LOCATION, KEY, DESCRIPTION, English, Chinese, French, German, Italian,
Japanese, Korean, Portugese, Spanish, Traditional Chinese.
The English text of the original resources is looked up and replaced by the
selected language's text.
"""
import locale
import os
import re

LANGUAGES = ("English", "Chinese", "French", "German", "Italian", "Japanese", "Korean",
             "Portugese", "Spanish", "Traditional Chinese")
_LOCALE_PREFIX = {"zh_TW": "Traditional Chinese", "zh_HK": "Traditional Chinese", "zh": "Chinese",
                  "fr": "French", "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean",
                  "pt": "Portugese", "es": "Spanish"}


def _unquote(s):
    if len(s) > 1 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].replace('""', '"')
    return s.replace("\\t", "\t").replace("\\n", "\n").replace("\\\\", "\\")


def _loose(s):
    """Key used for tolerant matching: no accelerator, mnemonic, ellipsis or colon."""
    s = s.split("\t")[0].replace("&", "").replace("…", "...")
    return re.sub(r"[\s.:]+$", "", s).strip().lower()


def system_language():
    env = os.environ.get("PHOTODESK_LANG")
    if env in LANGUAGES:
        return env
    try:
        loc = locale.getlocale()[0] or ""
    except ValueError:
        loc = ""
    if not loc and os.name == "nt":
        import ctypes
        loc = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
    loc = loc.replace("-", "_")
    for prefix, lang in _LOCALE_PREFIX.items():
        if loc.startswith(prefix) or loc.lower().startswith(lang.lower()):
            return lang
    return "English"


class Translator:
    def __init__(self, table_bytes=None, language="English"):
        self.language = language
        self.exact, self.loose = {}, {}
        if not table_bytes or language == "English":
            return
        col = 3 + LANGUAGES.index(language)
        text = table_bytes.decode("utf-16")
        for line in text.split("\r\n")[1:]:
            row = line.split("\t")
            if len(row) <= col:
                continue
            eng, other = _unquote(row[3]), _unquote(row[col])
            if not eng or not other:
                continue
            self.exact.setdefault(eng, other)
            self.loose.setdefault(_loose(eng), other)

    def __call__(self, text):
        if not text or not self.exact:
            return text
        if text in self.exact:
            return self.exact[text]
        head, sep, accel = text.partition("\t")
        if head in self.exact:
            return self.exact[head].split("\t")[0] + sep + accel
        hit = self.loose.get(_loose(text))
        if hit is None:
            return text
        # Give the translation the source's trailing "..." / ":" (or none).
        hit = re.sub(r"[\s.:…]+$", "", hit.split("\t")[0])
        tail = re.search(r"(\.\.\.|…|:)?\s*$", head).group(1) or ""
        return hit + tail + sep + accel
