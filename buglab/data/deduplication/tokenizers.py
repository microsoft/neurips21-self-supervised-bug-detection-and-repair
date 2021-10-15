import keyword
import logging
from tokenize import NAME, STRING, tokenize
from typing import List, TypedDict

from buglab.utils import detect_encoding_and_open


class TokenizedFileData(TypedDict):
    filename: str
    tokens: List[str]


def python_dedup_tokenize_file(filepath: str, all_tokens: bool = False) -> TokenizedFileData:
    tokens = []
    try:
        with open(filepath, "rb") as f:
            for toknum, tokval, _, _, _ in tokenize(f.readline):
                if all_tokens or toknum in {NAME, STRING}:
                    if not keyword.iskeyword(tokval):
                        tokens.append(tokval)
    except Exception as e:
        logging.error("Error tokenizing %s because %s" % (filepath, e))
    return dict(filename=filepath, tokens=tokens)
