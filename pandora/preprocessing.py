#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from operator import itemgetter
from collections import Counter
import jsonpickle
import numpy as np

from sklearn.feature_extraction import DictVectorizer

import pandora.utils as utils
from pandora.utils import PAD, BOS, EOS, UNK
from pandora.encoding import PandoraLabelEncoder as LabelEncoder


def index_characters(tokens, v2u=False, min_freq=5):
    """Creates a character index for representing
       tokens at the character level. All tokens
       will be lowercased.

    Parameters
    ===========
    tokens : list of str
        A list of tokens, on the basis of which the
        index will be constructed.
    focus_repr = str ('recurrent', 'convolutional')
        Which representation model will be used. in
        the case of recurrent representation, the
        special symbols <pad>, <eos>, <bos> are added to the
        character vocabulary.
    v2u : bool (default: False)
        Whether to squash the 'v' and 'u' characters
        to the same index. Useful for some historic
        languages.

    Returns
    ===========
    char_vocab : tuple
        A sorted tuple with all unique characters.
    char_lookup : dict
        An lookup dict of characters, where each character
        points to its index in `char_vocab`.

    """
    if v2u:
        vocab = Counter([ch
                         for tok in tokens
                         for ch in tok.lower().replace('v', 'u')])
    else:
        vocab = Counter([ch for tok in tokens for ch in tok.lower()])

    vocab = set([t for t, v in vocab.most_common()
                 if v >= min_freq])

    vocab = vocab.union({PAD, EOS, BOS, UNK})

    char_vocab = tuple(sorted(vocab))
    char_lookup = {}
    for char in char_vocab:
        char_lookup[char] = len(char_lookup)

    return char_vocab, char_lookup


def vectorize_tokens(tokens, token_char_lookup, focus_repr,
                     max_len=15, v2u=False):

    """Converts tokens to a tensor-representation
       at the character level. All tokens will be
       lowercased.

    Parameters
    ===========
    tokens : list of str
        A list of tokens to convert to a 3D tensor-
        representation.
    char_vector_dict : dict
        A dict used for indexing the characters:
        {char1 : idx1, char2 : idx2, ...}.
    max_len : int (default = 15)
        The length to which all tokens will be uni-
        formized (through right-side truncation or
        right)side padding with zeros).
    focus_repr = str ('recurrent', 'convolutional')
        Which representation model will be used. in
        the case of recurrent representation, the
        following special symbols <pad>, <bos>, <eos>,
        are added to the character vocabulary.
    v2u : bool (default: False)
        Whether to squash the 'v' and 'u' characters
        to the same index. Useful for some historic
        languages.

    Returns
    ===========
    X : array-like (float32)
        A 3D tensor-representation of the tokens,
        with shape:
        (nb tokens, max len, nb characters).

    """
    X = []
    for token in tokens:
        token = token.lower()
        if v2u:
            token.lower().replace('v', 'u')

        x = vectorize_token(seq=token,
                            char_lookup=token_char_lookup,
                            max_len=max_len,
                            focus_repr=focus_repr)
        X.append(x)

    return np.asarray(X, dtype='int32')


def vectorize_lemmas(lemmas, char_vector_dict,
                     max_len=15, categorical=False):

    """Converts lemmas to a tensor-representation
       at the character level. All lemmas will be
       lowercased.

    Parameters
    ===========
    lemmas : list of str
        A list of lemmas to convert to a 3D tensor-
        representation.
    char_vector_dict : dict
        A dict used for indexing the characters:
        {char1 : idx1, char2 : idx2, ...}.
    max_len : int (default = 15)
        The length to which all lemmas will be uni-
        formized (through right-side truncation or
        right-side padding with zeros).
    focus_repr = str ('recurrent', 'convolutional')
        Which representation model will be used. in
        the case of recurrent representation, the
        following <pad>, <eos> and <bos> symbols are
        added to the character vocabulary.

    Returns
    ===========
    X : array-like (float32)
        A 3D tensor-representation of the lemmas,
        with shape:
        (nb lemmas, max len, nb characters).

    """
    X = []
    for lemma in lemmas:
        lemma = lemma.lower()
        x = vectorize_lemma(seq=lemma,
                            char_vector_dict=char_vector_dict,
                            max_len=max_len, categorical=categorical)
        X.append(x)

    X = np.asarray(X, dtype='float32')

    return X


def vectorize_token(seq, char_lookup,
                    max_len, focus_repr):
    """Converts a single token to a matrix-
       representation at the character level.
       All characters will be lowercased.

    Parameters
    ===========
    seq : str
        A string representing the token.
    char_vector_dict : dict
        A dict used for indexing the characters:
        {char1 : idx1, char2 : idx2, ...}.
    max_len : int (default = 15)
        The length to which all tokens will be uni-
        formized (through right-side truncation or
        right-side padding with zeros).
    focus_repr = str ('recurrent', 'convolutional')
        Which representation model will be used. in
        the case of recurrent representation, the
        following <pad>, <eos> and <bos> symbols are
        added to the character vocabulary.

    Returns
    ===========
    X : array-like (float32)
        A 2D tensor-representation of the lemmas,
        with shape:
        (max len, nb characters).

    Notes
    ===========
    In the case of recurrent representation, the
    input sequence of characters will be reversed.

    """

    seq = seq[:(max_len - 2)]
    seq = [BOS] + list(seq) + [EOS]

    ints = []
    for char in seq:
        try:
            ints.append(char_lookup[char])
        except KeyError:
            ints.append(char_lookup[UNK])

    while len(ints) < max_len:
        ints.append(char_lookup[PAD])

    return ints


def vectorize_lemma(seq, char_vector_dict, max_len, categorical=False):

    """Converts a single lemma to a matrix-
       representation at the character level.
       All characters will be lowercased.

    Parameters
    ===========
    seq : str
        A string representing the lemma.
    char_vector_dict : dict
        A dict used for indexing the characters:
        {char1 : idx1, char2 : idx2, ...}.
    max_len : int (default = 15)
        The length to which the lemma will be uni-
        formized (through right-side truncation or
        right-side padding with zeros).
    Returns
    ===========
    X : array-like (float32)
        A 2D tensor-representation of the lemmas,
        with shape:
        (max_len, nb characters).

    """
    # cut, if needed:
    seq = seq[:(max_len - 2)]
    seq = [BOS] + list(seq) + [EOS]

    # pad, if needed:
    while len(seq) < max_len:
        seq += [PAD]

    seq_X = []

    filler = np.zeros(len(char_vector_dict), dtype='float32')
    for char in seq:
        char_idx = char_vector_dict.get(char, char_vector_dict[UNK])
        if categorical:
            f = filler.copy()
            f[char_idx] = 1
            seq_X.append(f)
        else:
            seq_X.append(char_idx)
    return np.array(seq_X)


def parse_morphs(morph):
    """Parses the strings representing morphological
       tags into a series of dictionaries, e.g.
       [
        gender=NEUTER|case=NOMINATIVE|number=SINGULAR|degree=POSITIVE,
        number=SINGULAR|person=PERSON_3|mood=INDICATIVE|voice=ACTIVE|tense=PRESENT,
         _,
         gender=MASCULINE|case=ACCUSATIVE|number=PLURAL,
         ...
        ]

    Parameters
    ===========
    morph : list of str
        A list of strings with morphological tags.
        Conventions:
            * key and value linked by '='
            * tags separated by pipes ('|')
    char_vector_dict : dict
        A dict used for indexing the characters:
        {char1 : idx1, char2 : idx2, ...}.
    max_len : int (default = 15)
        The length to which the lemma will be uni-
        formized (through right-side truncation or
        right-side padding with zeros).

    Returns
    ===========
    morph_dicts : list of dicts
        A list of dictionaries representing the
        morphological tags for each item.

    """

    morph_dicts = []
    for ml in morph:
        d = {}
        try:
            for a in ml.split('|'):
                k, v = a.split('=')
                d[k] = v
        except ValueError:
            pass
        morph_dicts.append(d)

    return morph_dicts


class Preprocessor(object):
    """
    Takes care of all preprocessing for Pandora,
    including creating the one-index for context
    tokens (embeddings) and creating the character-
    level representations of input and output labels.
    """

    def __init__(self, categorical=False):
        self.categorical = categorical
        self.max_lemma_len = None
        self.max_token_len = None
        self.focus_repr = None
        self.token_char_vocab, self.token_char_lookup = None, None
        self.known_tokens = None
        self.include_lemma, self.known_lemmas = None, None
        self.lemma_char_vocab, self.lemma_char_lookup = None, None
        self.min_lem_cnt, self.lemma_encoder = None, None
        self.pos_encoder = None
        self.morph_encoder = None
        self.nb_morph_cats, self.morph_idxs = None, None
        self.include_morph = None
        self.include_pos = None
        self.lemma_char_dict = None

    def fit(self, tokens, lemmas, pos, morph, include_lemma,
            include_morph, focus_repr, max_token_len=None,
            min_lem_cnt=1, max_lemma_len=None):

        """Fits the prepocessor on annotated materials.
           * Although tokens, lemmas and pos are optional,
             at least one of them must be included. Tokens
             are non-optional.
           * Out of vocabulary items are modelled using
             the '<UNK>' symbol.


        Parameters
        ===========
        tokens : list of str
            A list of tokens.
        lemmas : list of str (optional)
            A list of lemmas
        pos : list of str (optional)
            A list of part-of-speech tags.
        morph : list of str (optional)
            A list of morphological tags.
        include_lemma : str ('generate' or 'label')
            Indicates whether lemmas will be
            obtained through classification ('label')
            or character-level generation ('generate')
        include_morph : str ('label' or 'multilabel')
            Indicate whether the morphological prediction
            uses hardcare single-label classification,
            or a subtag level multilabel approach.
        focus_repr : str (one of: 'recurrent', 'convolutional')
            Which representation model will be used:
            concolutional filters ('convolutional') or
            a bidirectional LSTM ('recurrent').
        max_token_len : int
            The length (in characters) to which all tokens
            will be uniformized (through padding and cutting).
            (Note that the maximum length of the lemmas is
            automatically inferred from the maximum lemma
            length observed.)
        min_lem_cnt : int (default: 1)
            The minimum number of attestions a lemma label
            have to be assigned its own classification label
            in the case of `include_lemma` = 'label'.

        Returns
        ===========
            Itself.
        """
        if max_token_len:
            self.max_token_len = max_token_len
        else:
            self.max_token_len = len(max(tokens, key=len)) + 1

        self.focus_repr = focus_repr

        # fit focus tokens:
        self.token_char_vocab, self.token_char_lookup = \
            index_characters(tokens)
        self.known_tokens = set(tokens)

        # fit lemmas:
        if lemmas:
            self.include_lemma = include_lemma
            self.known_lemmas = set(lemmas)

            if max_lemma_len:
                self.max_lemma_len = max_lemma_len
            else:
                self.max_lemma_len = len(max(lemmas, key=len)) + 1

            if include_lemma == 'generate':
                self.lemma_char_vocab, self.lemma_char_lookup = \
                    index_characters(lemmas)
            elif include_lemma == 'label':
                self.min_lem_cnt = min_lem_cnt
                cnt = Counter(lemmas)
                trunc_lems = \
                    [k for k, v in cnt.items() if v >= self.min_lem_cnt]
                self.lemma_encoder = LabelEncoder()
                self.lemma_encoder.fit(trunc_lems + ['<UNK>'])

        # fit pos labels:
        if pos:
            self.include_pos = True
            self.pos_encoder = LabelEncoder()
            self.pos_encoder.fit(pos + ['<UNK>'])

        if morph:
            self.include_morph = include_morph
            if self.include_morph == 'label':
                self.morph_encoder = LabelEncoder()
                self.morph_encoder.fit(morph + ['<UNK>'])
            elif self.include_morph == 'multilabel':
                # fit morph analysis:
                morph_dicts = parse_morphs(morph)
                self.morph_encoder = DictVectorizer(sparse=False)
                self.morph_encoder.fit(morph_dicts)
            self.deduct_secondary_informations_from_morph()

        return self

    def deduct_secondary_informations_from_morph(self):
        """ Deducts internal var values such as morph_idxs from self.morph_encoder

        Used for load and fit
        """
        if self.include_morph == 'label':
            self.nb_morph_cats = len(self.morph_encoder.classes_)
        elif self.include_morph == 'multilabel':
            self.nb_morph_cats = len(self.morph_encoder.feature_names_)
            self.morph_idxs = {}
            for i, feat_name in enumerate(
                    self.morph_encoder.feature_names_):
                label, _ = feat_name.strip().split('=')
                try:
                    self.morph_idxs[label].add(i)
                except KeyError:
                    self.morph_idxs[label] = set()
                    self.morph_idxs[label].add(i)

    def transform(self, tokens=None, lemmas=None, pos=None, morph=None):
        """ Transforms a list of corresponding tokens,
            lemmas, pos tags and morph tags to the correct
            feature representations. This method will encode
            out-of-vocabulary items using special symbols to
            prevent sublt leakage from test to train material.

        Parameters
        ===========
        tokens : list of str
            A list of tokens.
        lemmas : list of str (optional)
            A list of lemmas
        pos : list of str (optional)
            A list of part-of-speech tags.
        morph : list of str (optional)
            A list of morphological tags.


        Returns
        ===========
            returnables : dict
            Depending on the preprocessor's parametrization,
            a dict with keys {'X_focus' : token representation,
                              'X_lemma' : lemma representation,
                              'X_pos' : pos representation,
                              'X_morph' : pos representation}
        """

        # vectorize focus tokens:
        X_focus = vectorize_tokens(
            tokens=tokens,
            token_char_lookup=self.token_char_lookup,
            max_len=self.max_token_len,
            focus_repr=self.focus_repr)

        returnables = {'X_focus': X_focus}

        if lemmas and self.include_lemma:
            # vectorize lemmas:
            if self.include_lemma == 'generate':
                X_lemma = vectorize_lemmas(
                    lemmas=lemmas,
                    char_vector_dict=self.lemma_char_lookup,
                    max_len=self.max_lemma_len,
                    categorical=self.categorical)

            elif self.include_lemma == 'label':
                X_lemma = [l if l in self.lemma_encoder.classes_ else '<UNK>'
                           for l in lemmas]
                X_lemma = self.lemma_encoder.transform(X_lemma)

                if self.categorical:
                    X_lemma = utils.to_categorical(
                        X_lemma, num_classes=len(self.lemma_encoder.classes_))

            returnables['X_lemma'] = X_lemma

        if pos:
            # vectorize pos:
            X_pos = [p if p in self.pos_encoder.classes_ else '<UNK>'
                     for p in pos]
            X_pos = self.pos_encoder.transform(X_pos)

            if self.categorical:
                X_pos = utils.to_categorical(
                    X_pos, num_classes=len(self.pos_encoder.classes_))

            returnables['X_pos'] = X_pos

        if morph:
            # vectorize morph:
            if self.include_morph == 'label':
                X_morph = [m if m in self.morph_encoder.classes_ else '<UNK>'
                           for m in morph]
                X_morph = self.morph_encoder.transform(X_morph)

                if self.categorical:
                    X_morph = utils.to_categorical(
                        X_morph, num_classes=len(self.morph_encoder.classes_))

                returnables['X_morph'] = X_morph

            elif self.include_morph == 'multilabel':
                morph_dicts = parse_morphs(morph)
                X_morph = self.morph_encoder.transform(morph_dicts)
                returnables['X_morph'] = X_morph

        return returnables

    def fit_transform(self, tokens, lemmas, pos, morph):

        """Commodity function, equivalent to:
            self.transform(self.fit(...))
        """

        self.fit(tokens, lemmas, pos, morph)
        return self.transform(tokens, lemmas, pos, morph)

    def inverse_transform_lemmas(self, predictions):
        """Converts the model's lemma predictions to
           a human-readable list of lemma labels.
            * In the case of include_lemma = 'generate',
              a matrix-like representation of lemma outputs
              (generated at the character level) is converted
              to strings. Technical note: no beam-search is applied;
              we apply a hardcore maxarg. See code for details.
            * In the case of include_lemma = 'label',
              an index of lemma predictions is converted
              to their corresponding labels.


        Parameters
        ===========
        predictions : list
            - list of lemma-indices (`include_lemma` = 'label')
            - list of lemma-matrices at character-level (when
              `include_lemma` = 'generate').


        Returns
        ===========
            The lemma predictions as a list of strings.
        """

        pred_lemmas = []
        if self.include_lemma == 'generate':
            for pred in predictions:
                pred_lem = ''
                for positions in pred:
                    # winning position
                    top_idx = np.argmax(positions)
                    # look up corresponding char
                    c = self.lemma_char_vocab[top_idx]
                    if c in (BOS, EOS):
                        continue
                    # truncate once padding is generated
                    if c == PAD:
                        break
                    else:
                        pred_lem += c  # add character
                pred_lemmas.append(pred_lem)

        elif self.include_lemma == 'label':
            predictions = np.argmax(predictions, axis=1)
            pred_lemmas = self.lemma_encoder.inverse_transform(predictions)

        return pred_lemmas

    def inverse_transform_pos(self, predictions):
        """Converts index-predictions of part of speech tags
           and converts them to string labels.

        Parameters
        ===========
        predictions : list of ints
            A iterable of indices corresponding to pos prediction labels.

        Returns
        ===========
            The pos predictions as a list of strings.
        """
        predictions = np.argmax(predictions, axis=1)
        return self.pos_encoder.inverse_transform(predictions)

    def inverse_transform_morph(self, predictions, threshold=.5):
        """Converts morphological predictions to a list of human-
           readable morphological tags.

        Parameters
        ===========
        predictions : list
            A iterable of indices corresponding to morphological
            prediction labels.
        threshold : float (default = .5)
            In the case of the 'multilabel' setup, only labels with
            a softmax probability >= `threshold` are included.

        Returns
        ===========
            The morphological predictions as a list of strings.
        """
        if self.include_morph == 'label':
            predictions = np.argmax(predictions, axis=1)
            return self.morph_encoder.inverse_transform(predictions)
        elif self.include_morph == 'multilabel':
            morphs = []
            for pred in predictions:
                m = []
                for label, idxs in self.morph_idxs.items():
                    scores = ((pred[idx], idx) for idx in idxs)
                    max_score = max(scores, key=itemgetter(0))
                    if max_score[0] >= threshold:
                        f = self.morph_encoder.feature_names_[max_score[1]]
                        m.append(f)
                if m:
                    morphs.append('|'.join(m))
                else:
                    morphs.append('_')
            return morphs

    def save(self, model_dir):
        if self.lemma_encoder:
            self.lemma_encoder.save(
                p=os.sep.join((model_dir, 'lemma_enc.txt')))
        if self.pos_encoder:
            self.pos_encoder.save(
                p=os.sep.join((model_dir, 'pos_enc.txt')))
        if self.morph_encoder:
            if self.include_morph == "label":
                self.morph_encoder.save(
                    p=os.sep.join((model_dir, 'morph_enc.txt')))
            else:
                with open(os.sep.join((model_dir, 'morph_enc.json')), "w") as f:
                    f.write(jsonpickle.dumps(self.morph_encoder))

        if self.token_char_vocab:  # TODO: currently this always evaluates true
            path = os.sep.join((model_dir, 'token_char_lookup.txt'))
            with open(path, 'w') as f:
                for idx, char in enumerate(self.token_char_vocab):
                    f.write('\t'.join((char, str(idx)))+'\n')
        if self.include_lemma == 'generate':
            path = os.sep.join((model_dir, 'lemma_char_lookup.txt'))
            with open(path, 'w') as f:
                for c, idx in self.lemma_char_lookup.items():
                    f.write('\t'.join((c, str(idx)))+'\n')

    def load(self, model_dir, max_token_len, focus_repr,
             max_lemma_len, include_lemma, include_pos, include_morph=False):

        self.max_token_len = max_token_len
        self.max_lemma_len = max_lemma_len
        self.focus_repr = focus_repr
        self.include_lemma = include_lemma
        self.include_pos = include_pos
        self.include_morph = include_morph

        self.token_char_lookup = {}
        path = os.sep.join((model_dir, 'token_char_lookup.txt'))
        with open(path, "r") as f:
            for line in f.readlines():
                c, idx = line.strip().split()
                self.token_char_lookup[c] = int(idx)
        self.token_char_vocab = tuple(sorted(self.token_char_lookup.keys()))

        if self.include_pos:
            self.pos_encoder = LabelEncoder()
            self.pos_encoder.load(p=os.sep.join((model_dir, 'pos_enc.txt')))

        if self.include_morph:
            if self.include_morph == 'label':
                self.morph_encoder = LabelEncoder()
                self.morph_encoder.load(p=os.sep.join((model_dir, 'morph_enc.txt')))
            elif self.include_morph == 'multilabel':
                with open(os.sep.join((model_dir, 'morph_enc.json')), "r") as f:
                    self.morph_encoder = jsonpickle.loads(f.read())
            self.deduct_secondary_informations_from_morph()

        if self.include_lemma == 'label':
            self.lemma_encoder = LabelEncoder()
            self.lemma_encoder.load(
                p=os.sep.join((model_dir, 'lemma_enc.txt')))
        elif self.include_lemma == 'generate':
            self.lemma_char_lookup = {}
            path = os.sep.join((model_dir, 'lemma_char_lookup.txt'))
            for line in open(path, 'r'):
                c, idx = line.strip().split()
                self.lemma_char_lookup[int(idx)] = c
            self.lemma_char_vocab = tuple(sorted(self.lemma_char_lookup.values()))
            filler = np.zeros(len(self.lemma_char_vocab), dtype='float32')
            self.lemma_char_dict = {}
            for idx, char in enumerate(self.lemma_char_vocab):
                ph = filler.copy()
                ph[idx] = 1
                self.lemma_char_dict[char] = ph

        if include_morph:
            self.include_morph = include_morph
