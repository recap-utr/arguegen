import itertools
import typing as t

import lemminflect
from nltk import pos_tag, word_tokenize

ADDITIONAL_INFLECTIONS = {"prove": ["proven"]}


def _lemma_parts(text: str, pos: str) -> t.List[str]:
    tokens: t.List[str] = word_tokenize(text)  # type: ignore

    *parts, tail = tokens
    parts.append(lemminflect.getLemma(tail, pos)[0])

    return parts


def _inflect(text: str, pos: str) -> t.Tuple[str, t.FrozenSet[str]]:
    """Return the lemma of `text` and all inflected forms of `text`."""

    lemma_parts = _lemma_parts(text, pos)
    lemma = " ".join(lemma_parts)
    *lemma_prefixes, lemma_suffix = lemma_parts
    lemma_prefix = " ".join(lemma_prefixes)

    inflections = frozenset(
        itertools.chain(*lemminflect.getAllInflections(lemma_suffix, pos).values())
    )

    if not inflections:
        inflections = frozenset(
            itertools.chain(
                *lemminflect.getAllInflectionsOOV(lemma_suffix, pos).values()
            )
        )

    forms = set()
    forms.add(lemma)

    for inflection in inflections:
        form = " ".join([lemma_prefix, inflection])
        forms.add(form.strip())

    if additional_inflections := ADDITIONAL_INFLECTIONS.get(lemma):
        forms.update(additional_inflections)

    return lemma, frozenset(forms)


def inflect_concept(
    text: str, pos_tags: t.Union[str, t.Iterable[t.Optional[str]]]
) -> t.Tuple[str, t.FrozenSet[str]]:
    if isinstance(pos_tags, str):
        return _inflect(text, pos_tags)

    lemmas = set()
    forms = set()

    for pos in pos_tags:
        if pos is None:
            # The pos tag (index 1) of the last token (index -1) is used.
            pos = pos_tag(text, tagset="universal")[-1][1]

        pos_lemma, pos_forms = inflect_concept(text, pos)
        lemmas.add(pos_lemma)
        forms.update(pos_forms)

    # assert len(lemmas) == 1

    # Currently, PROPN is removed from the list of possible keywords.
    # Thus, len(query.pos_tags) == 1 in all cases.
    lemma = next(iter(lemmas))

    if not text in forms:
        raise RuntimeError(
            f"{text=} not in {forms=}. You should set ADDITIONAL_INFLECTIONS['{lemma}'] = ['{text}']"
        )

    return lemma, frozenset(forms)
