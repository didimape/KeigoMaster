# -----------------------
# JAPANESE TOKENIZER
# -----------------------
# Japanese text does not contain spaces,
# so we use Fugashi to split sentences
# with Tagger()

from fugashi import Tagger

tagger = Tagger()


def tokenize_japanese(text):
    """
    Tokenize Japanese text into words.
    """

    words = []

    for word in tagger(text):
        words.append(word.surface)

    return words