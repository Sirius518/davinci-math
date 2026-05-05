from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
import unicodedata


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
HTML_IMAGE_PATTERN = re.compile(r"<img\b[^>]*>", flags=re.IGNORECASE)
BASE64_PATTERN = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", flags=re.IGNORECASE)
POINTS_PATTERN = re.compile(r"\(\s*\d+\s*points?\s*\)", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
CJK_PATTERN = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF"
    r"\u2E80-\u2EFF"
    r"\u3000-\u303F"
    r"\uF900-\uFAFF"
    r"\U00020000-\U0002A6DF]"
)
OTHER_NON_ENGLISH_PATTERN = re.compile(
    r"[\u0400-\u052F"
    r"\u0590-\u05FF"
    r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    r"\u0900-\u097F"
    r"\u0980-\u09FF"
    r"\u0A80-\u0AFF"
    r"\u0B00-\u0B7F"
    r"\u0B80-\u0BFF"
    r"\u0C00-\u0C7F"
    r"\u0C80-\u0CFF"
    r"\u0D00-\u0D7F"
    r"\u0E00-\u0E7F"
    r"\u0E80-\u0EFF"
    r"\u10A0-\u10FF"
    r"\u1100-\u11FF"
    r"\u3040-\u30FF"
    r"\uAC00-\uD7AF]"
)
NON_ENGLISH_PATTERN = re.compile(
    r"[\u0400-\u052F\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    r"\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF"
    r"\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0E00-\u0E7F\u0E80-\u0EFF"
    r"\u10A0-\u10FF\u1100-\u11FF"
    r"\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]"
)
INLINE_OPTION_PATTERN = re.compile(
    r"(?:^|\n)\s*\(?\s*A\s*[\).:]\s*.+?"
    r"\(?\s*B\s*[\).:]\s*.+?"
    r"(?:\(?\s*C\s*[\).:]\s*.+?)?",
    re.IGNORECASE | re.DOTALL,
)
COMPACT_OPTION_PATTERN = re.compile(
    r"(?:^|\s)\(?\s*(?:A|B|C|D|E|F)\s*[\).:]\s*[^\n]{1,120}"
    r"(?=(?:\s+\(?\s*(?:A|B|C|D|E|F)\s*[\).:])|\s*$)",
    re.IGNORECASE,
)
ANSWER_LETTER_PATTERN = re.compile(r"\b([A-F])\b", flags=re.IGNORECASE)
OPTION_LINE_PATTERN = re.compile(
    r"(?im)^[\(\[]?\s*([A-F])[\)\].:-]\s*(.+?)\s*(?=^[\(\[]?\s*[A-F][\)\].:-]\s*|\Z)"
)
SELECT_ALL_PATTERN = re.compile(r"\b(?:select|choose)\s+all\b", flags=re.IGNORECASE)
NEGATIVE_CHOICE_PATTERN = re.compile(r"\bwhich\s+(?:is|are)\s+not\b", flags=re.IGNORECASE)
WHICH_OF_PATTERN = re.compile(r"\bwhich of the (?:following|above)\b", flags=re.IGNORECASE)
WHICH_OF_THESE_PATTERN = re.compile(r"\bwhich of these\b", flags=re.IGNORECASE)
ROMAN_MARKER_PATTERN = re.compile(
    r"^\s*(?:\(?[ivx]+\)?[\).:-]|\(?\d+\)?[\).:-])\s+",
    flags=re.IGNORECASE | re.MULTILINE,
)
NONE_OR_ALL_ABOVE_PATTERN = re.compile(r"\b(?:none|all)\s+of\s+the\s+above\b", flags=re.IGNORECASE)
CROSS_OPTION_REFERENCE_PATTERN = re.compile(
    r"\bboth\s+[A-F]\s+and\s+[A-F]\b|\b[A-F]\s+and\s+[A-F]\b",
    flags=re.IGNORECASE,
)
INDETERMINATE_OPTION_PATTERN = re.compile(
    r"\bcannot be determined\b|\bnot enough information\b|\binsufficient\b",
    flags=re.IGNORECASE,
)
LATEX_OPTION_PATTERN = re.compile(
    r"(?:\\textbf\s*\{?\s*\(?\s*([A-F])\s*\)?\s*\}?"
    r"|\\text\s*\{?\s*\(?\s*([A-F])\s*\)?\s*\}?)"
    r"\s*\\?\s*(.+?)(?=\\textbf|\\text\s*\{|$)",
    re.IGNORECASE | re.MULTILINE,
)
THE_OPTIONS_ARE_PATTERN = re.compile(r"\bthe options are\s*:", re.IGNORECASE)

_INVISIBLE_CODEPOINTS = {
    0x00AD,  # soft hyphen
    0x034F,  # combining grapheme joiner
    0x061C,  # arabic letter mark
    0x180E,  # mongolian vowel separator
    0x200B,  # zero width space
    0x200C,  # zero width non-joiner
    0x200D,  # zero width joiner
    0x200E,  # left-to-right mark
    0x200F,  # right-to-left mark
    0x2060,  # word joiner
    0x2066,  # left-to-right isolate
    0x2067,  # right-to-left isolate
    0x2068,  # first strong isolate
    0x2069,  # pop directional isolate
    0xFEFF,  # zero width no-break space
}
_UNICODE_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201B": "'",
        "\u2032": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201F": '"',
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00A0": " ",
    }
)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def fullwidth_to_halfwidth(text: str) -> str:
    output: list[str] = []
    for char in text:
        codepoint = ord(char)
        if codepoint == 0x3000:
            output.append(" ")
        elif 0xFF01 <= codepoint <= 0xFF5E:
            output.append(chr(codepoint - 0xFEE0))
        else:
            output.append(char)
    return "".join(output)


def strip_invisible_chars(text: str) -> str:
    output: list[str] = []
    for char in text:
        codepoint = ord(char)
        if codepoint in _INVISIBLE_CODEPOINTS:
            continue
        if unicodedata.category(char) == "Cc" and char not in {"\n", "\t", "\r"}:
            continue
        output.append(char)
    return "".join(output)


def normalize_unicode_punctuation(text: str) -> str:
    return text.translate(_UNICODE_PUNCT_TRANSLATION)


def normalize_text(text: str, *, normalize_punctuation: bool = True) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = strip_invisible_chars(normalized)
    normalized = fullwidth_to_halfwidth(normalized)
    if normalize_punctuation:
        normalized = normalize_unicode_punctuation(normalized)
    return format_text(normalized)


def clean_urls_and_images(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = MARKDOWN_IMAGE_PATTERN.sub(" ", text)
    text = HTML_IMAGE_PATTERN.sub(" ", text)
    text = BASE64_PATTERN.sub(" ", text)
    return text


def format_text(text: str) -> str:
    text = normalize_newlines(text)
    text = POINTS_PATTERN.sub(" ", text)
    text = "\n".join(WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines())
    text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def contains_cjk(text: str) -> bool:
    return bool(CJK_PATTERN.search(text))


def contains_other_non_english(text: str) -> bool:
    return bool(OTHER_NON_ENGLISH_PATTERN.search(text))


def contains_non_english_script(text: str) -> bool:
    return contains_cjk(text) or contains_other_non_english(text)


def _classify_letter(codepoint: int) -> str:
    """Classify a Unicode letter codepoint into a script bucket."""
    if codepoint <= 0x024F:
        return "latin"
    if 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF:
        return "latin"
    if (0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0x2E80 <= codepoint <= 0x2EFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2A6DF):
        return "cjk"
    if 0x3040 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "cjk"
    if 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF:
        return "cjk"
    return "other"


@dataclass(slots=True)
class ScriptProfile:
    total_letters: int
    latin_count: int
    cjk_count: int
    other_count: int

    @property
    def latin_ratio(self) -> float:
        return self.latin_count / self.total_letters if self.total_letters else 1.0

    @property
    def cjk_ratio(self) -> float:
        return self.cjk_count / self.total_letters if self.total_letters else 0.0

    @property
    def other_ratio(self) -> float:
        return self.other_count / self.total_letters if self.total_letters else 0.0


def analyze_script_profile(text: str) -> ScriptProfile:
    """Analyze the script distribution of letter characters in text."""
    latin = 0
    cjk = 0
    other = 0
    for char in text:
        cat = unicodedata.category(char)
        if not cat.startswith("L"):
            continue
        bucket = _classify_letter(ord(char))
        if bucket == "latin":
            latin += 1
        elif bucket == "cjk":
            cjk += 1
        else:
            other += 1
    return ScriptProfile(
        total_letters=latin + cjk + other,
        latin_count=latin,
        cjk_count=cjk,
        other_count=other,
    )


def should_filter_short_question(text: str) -> bool:
    words = [token for token in re.split(r"\s+", text.strip()) if token]
    return len(words) < 5 and len(text.strip()) < 20


def extract_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for match in OPTION_LINE_PATTERN.finditer(normalize_newlines(question)):
        options[match.group(1).upper()] = match.group(2).strip()
    if not options:
        for match in LATEX_OPTION_PATTERN.finditer(question):
            letter = (match.group(1) or match.group(2) or "").upper()
            value = match.group(3).strip() if match.group(3) else ""
            if letter and value:
                options[letter] = value
    return options


def normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def preprocess_for_minhash(text: str) -> str:
    normalized = normalize_answer_text(text)
    no_punct = PUNCTUATION_PATTERN.sub(" ", normalized)
    return re.sub(r"\s+", " ", no_punct).strip()


def extract_answer_letter(text: str) -> str:
    match = ANSWER_LETTER_PATTERN.search(text.strip())
    if not match:
        return ""
    return match.group(1).upper()


def strip_option_lines(question: str) -> str:
    stripped = OPTION_LINE_PATTERN.sub("", normalize_newlines(question))
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def looks_like_compact_options(question: str) -> bool:
    normalized = normalize_newlines(question)
    matches = COMPACT_OPTION_PATTERN.findall(normalized)
    if len(matches) >= 3:
        return True
    marker_count = len(re.findall(r"\([A-F]\)|\b[A-F][\).:]", normalized, flags=re.IGNORECASE))
    return marker_count >= 3


def is_unconvertible_mcq(question: str, answer_option_value: str) -> tuple[bool, str]:
    lowered_question = normalize_answer_text(question)
    if (WHICH_OF_PATTERN.search(lowered_question) or WHICH_OF_THESE_PATTERN.search(lowered_question)) and "such that" not in lowered_question:
        return True, "blocked_pattern"
    if SELECT_ALL_PATTERN.search(lowered_question):
        return True, "select_all_pattern"
    if NEGATIVE_CHOICE_PATTERN.search(lowered_question):
        return True, "negative_choice_pattern"
    has_roman_or_numbered_markers = bool(ROMAN_MARKER_PATTERN.search(question))
    if has_roman_or_numbered_markers and (WHICH_OF_PATTERN.search(lowered_question) or WHICH_OF_THESE_PATTERN.search(lowered_question)):
        return True, "enumeration_reference_pattern"
    lowered_option = normalize_answer_text(answer_option_value)
    if NONE_OR_ALL_ABOVE_PATTERN.search(lowered_option):
        return True, "meta_option_pattern"
    if CROSS_OPTION_REFERENCE_PATTERN.search(lowered_option):
        return True, "cross_option_reference_pattern"
    if INDETERMINATE_OPTION_PATTERN.search(lowered_option):
        return True, "indeterminate_option_pattern"
    return False, ""


def ngrams(text: str, n: int = 3) -> set[str]:
    normalized = normalize_answer_text(text)
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def majority_vote(values: list[str]) -> tuple[str, int]:
    counts = Counter(item for item in values if item)
    if not counts:
        return "", 0
    winner, count = counts.most_common(1)[0]
    return winner, count
