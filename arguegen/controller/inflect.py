import itertools
import typing as t
from collections import defaultdict

import immutables
import lemminflect
from nltk import pos_tag, word_tokenize
from arguegen.model.config import Config

config = Config.instance()


def _lemma_parts(text: str, pos: str) -> t.List[str]:
    tokens: t.List[str] = text.split()  # type: ignore

    *parts, tail = tokens
    parts.append(lemminflect.getLemma(tail, pos)[0])

    return parts


def _inflect(text: str, pos: str) -> t.Tuple[str, t.Dict[str, t.List[str]]]:
    """Return the lemma of `text` and all inflected forms of `text`."""

    lemma_parts = _lemma_parts(text, pos)
    lemma = " ".join(lemma_parts)
    *lemma_prefixes, lemma_suffix = lemma_parts
    lemma_prefix = " ".join(lemma_prefixes)

    inflection_map = lemminflect.getAllInflections(lemma_suffix, pos)

    if not inflection_map:
        inflection_map = lemminflect.getAllInflectionsOOV(lemma_suffix, pos)

    forms = defaultdict(list)
    forms["X"].append(lemma)

    for pos_tag, inflections in inflection_map.items():
        for inflection in inflections:
            form = " ".join([lemma_prefix, inflection])
            forms[pos_tag].append(form.strip())

    if additional_inflections := config["nlp"]["inflections"].get(lemma):
        for pos_tag, inflections in additional_inflections.items():
            forms[pos_tag].extend(inflections)

    return lemma, forms


def make_immutable(
    forms: t.Mapping[str, t.Iterable[str]], invert: bool
) -> immutables.Map[str, t.Tuple[str, ...]]:
    if invert:
        inverted = defaultdict(list)

        for key, values in forms.items():
            for value in values:
                inverted[value].append(key)

        return make_immutable(inverted, invert=False)

    return immutables.Map({key: tuple(values) for key, values in forms.items()})


def inflect_concept(
    text: str, pos_tags: t.Union[str, t.Iterable[t.Optional[str]]]
) -> t.Tuple[
    str, immutables.Map[str, t.Tuple[str, ...]], immutables.Map[str, t.Tuple[str, ...]]
]:
    if isinstance(pos_tags, str):
        lemma, forms = _inflect(text, pos_tags)
        return lemma, make_immutable(forms, True), make_immutable(forms, False)

    lemmas = set()
    form2pos = defaultdict(list)
    pos2form = defaultdict(list)

    for pos in pos_tags:
        if pos is None:
            # The pos tag (index 1) of the last token (index -1) is used.
            pos = pos_tag(text, tagset="universal")[-1][1]

        _lemma, _form2pos, _pos2form = inflect_concept(text, pos)
        lemmas.add(_lemma)

        for _pos_tag, _forms in _pos2form.items():
            pos2form[_pos_tag].extend(_forms)

        for _form, _pos_tags in _form2pos.items():
            form2pos[_form].extend(_pos_tags)

    # assert len(lemmas) == 1

    # Currently, PROPN is removed from the list of possible keywords.
    # Thus, len(query.pos_tags) == 1 in all cases.
    lemma = next(iter(lemmas))

    if not text in form2pos:
        raise RuntimeError(
            f"{text=} not in {form2pos=}. You should update config['nlp']['inflections']['{lemma}']."
        )

    return lemma, make_immutable(form2pos, False), make_immutable(pos2form, False)
