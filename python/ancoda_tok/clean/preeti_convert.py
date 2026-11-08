"""Unicode→Preeti conversion table and functions.

Source: https://github.com/globalpolicy/UnicodeToPreeti
Commit: c1d6e3080976d7113c24ab5b59d06f937a20949d
License: MIT (Copyright (c) 2017 Global Policy)

The upstream file I/O CLI is omitted; the mapping table and conversion
logic are kept for synthetic Preeti positives in the legacy-font classifier.
"""

from __future__ import annotations

# MIT: Copyright (c) 2017 Global Policy
UNICODE_TO_PREETI: dict[str, str] = {
    "अ": "c",
    "आ": "cf",
    "ा": "f",
    "इ": "O",
    "ई": "O{",
    "र्": "{",
    "उ": "p",
    "ए": "P",
    "े": "]",
    "ै": "}",
    "ो": "f]",
    "ौ": "f}",
    "ओ": "cf]",
    "औ": "cf}",
    "ं": "+",
    "ँ": "F",
    "ि": "l",
    "ी": "L",
    "ु": "'",
    "ू": '"',
    "क": "s",
    "ख": "v",
    "ग": "u",
    "घ": "3",
    "ङ": "ª",
    "च": "r",
    "छ": "5",
    "ज": "h",
    "झ": "´",
    "ञ": "`",
    "ट": "6",
    "ठ": "7",
    "ड": "8",
    "ढ": "9",
    "ण": "0f",
    "त": "t",
    "थ": "y",
    "द": "b",
    "ध": "w",
    "न": "g",
    "प": "k",
    "फ": "km",
    "ब": "a",
    "भ": "e",
    "म": "d",
    "य": "o",
    "र": "/",
    "रू": "?",
    "ृ": "[",
    "ल": "n",
    "व": "j",
    "स": ";",
    "श": "z",
    "ष": "if",
    "ज्ञ": "1",
    "ह": "x",
    "१": "!",
    "२": "@",
    "३": "#",
    "४": "$",
    "५": "%",
    "६": "^",
    "७": "&",
    "८": "*",
    "९": "(",
    "०": ")",
    "।": ".",
    "्": "\\",
    "ऊ": "pm",
    "-": " ",
    "(": "-",
    ")": "_",
}


def normalize_unicode(unicodetext: str) -> str:
    index = -1
    normalized = ""
    while index + 1 < len(unicodetext):
        index += 1
        character = unicodetext[index]
        try:
            try:
                if character != "र" and (
                    unicodetext[index + 1] == "्"
                    and unicodetext[index + 2] != " "
                    and unicodetext[index + 2] != "।"
                    and unicodetext[index + 2] != ","
                ) and unicodetext[index + 2] != "र":
                    if UNICODE_TO_PREETI[character] in list("wertyuxasdghjkzvn"):
                        normalized += chr(ord(UNICODE_TO_PREETI[character]) - 32)
                        index += 1
                        continue
                    if character == "स":
                        normalized += ":"
                        index += 1
                        continue
                    if character == "ष":
                        normalized += "i"
                        index += 1
                        continue
                if (
                    unicodetext[index - 1] != "र"
                    and character == "्"
                    and unicodetext[index + 1] == "र"
                ):
                    if (
                        unicodetext[index - 1] != "ट"
                        and unicodetext[index - 1] != "ठ"
                        and unicodetext[index - 1] != "ड"
                    ):
                        normalized += "|"
                        index += 1
                        continue
                    normalized += "«"
                    index += 1
                    continue
            except IndexError:
                pass
            normalized += character
        except KeyError:
            normalized += character
    return normalized.replace("त|", "q")


def convert_unicode_to_preeti(unicodetext: str) -> str:
    """Convert a Unicode Nepali string to a Preeti-encoded Latin string."""
    normalizedunicodetext = normalize_unicode(unicodetext)
    converted = ""
    index = -1
    while index + 1 < len(normalizedunicodetext):
        index += 1
        character = normalizedunicodetext[index]
        if character == "\ufeff":
            continue
        try:
            try:
                if normalizedunicodetext[index + 1] == "ि":
                    if character == "q":
                        converted += "l" + character
                    else:
                        converted += "l" + UNICODE_TO_PREETI[character]
                    index += 1
                    continue

                if normalizedunicodetext[index + 2] == "ि":
                    if character in list("WERTYUXASDGHJK:ZVN"):
                        if normalizedunicodetext[index + 1] != "q":
                            converted += (
                                "l" + character + UNICODE_TO_PREETI[normalizedunicodetext[index + 1]]
                            )
                            index += 2
                            continue
                        if normalizedunicodetext[index + 1] == "q":
                            converted += "l" + character + normalizedunicodetext[index + 1]
                            index += 2
                            continue

                if normalizedunicodetext[index + 1] == "्" and character == "र":
                    if normalizedunicodetext[index + 3] in "ाोौेैी":
                        converted += (
                            UNICODE_TO_PREETI[normalizedunicodetext[index + 2]]
                            + UNICODE_TO_PREETI[normalizedunicodetext[index + 3]]
                            + "{"
                        )
                        index += 3
                        continue
                    if normalizedunicodetext[index + 3] == "ि":
                        converted += (
                            UNICODE_TO_PREETI[normalizedunicodetext[index + 3]]
                            + UNICODE_TO_PREETI[normalizedunicodetext[index + 2]]
                            + "{"
                        )
                        index += 3
                        continue
                    converted += UNICODE_TO_PREETI[normalizedunicodetext[index + 2]] + "{"
                    index += 2
                    continue

                if normalizedunicodetext[index + 3] == "ि":
                    if normalizedunicodetext[index + 2] in "|«":
                        if character in list("WERTYUXASDGHJK:ZVNIi"):
                            converted += (
                                "l"
                                + character
                                + UNICODE_TO_PREETI[normalizedunicodetext[index + 1]]
                                + normalizedunicodetext[index + 2]
                            )
                            index += 3
                            continue
            except IndexError:
                pass
            converted += UNICODE_TO_PREETI[character]
        except KeyError:
            converted += character

    converted = converted.replace("Si", "I")
    converted = converted.replace("H`", "1")
    converted = converted.replace("b\\w", "4")
    converted = converted.replace("z|", ">")
    converted = converted.replace("/'", "?")
    converted = converted.replace('/"', "¿")
    converted = converted.replace("Tt", "Q")
    converted = converted.replace("b\\lj", "lå")
    converted = converted.replace("b\\j", "å")
    converted = converted.replace("0f\\", "0")
    converted = converted.replace("`\\", "~")
    return converted
