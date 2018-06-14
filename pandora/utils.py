#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import codecs
import re
import configparser as ConfigParser

import numpy as np

PAD, EOS, BOS, UNK = '<PAD>', '<EOS>', '<BOS>', '<UNK>'


def load_annotated_dir(directory='directory', format='tab',
                       extensions='.txt', nb_instances=None,
                       include_lemma=True, include_morph=True,
                       include_pos=True, col_pos=None,
                       col_lemma=None, col_morph=None):

    """Loads annotated data files from a (single) directory.

    Parameters
    ===========
    directory : str (default: 'directory')
        Path to the directory to load.
    format : str (default: tab)
        The annotation format used (see notes).
        Currently supported:
            - 'tab': tab-separated format
            - 'conll': the connl-shared task format
    ext = str (default: '.txt')
        Only filenames ending in this extension will be
        loaded.
    nb_instances : int, optional (default: None)
        Max number of instances to load *from each file*
        under dir. This is a useful cutoff for development
        purposes. All instances will be loaded if `None`.
    include_lemma : bool (default: True)
        Whether to include lemma labels
    include_pos : bool (default: True)
        Whether to include part-of-speech labels
    include_morph : bool (default: True)
        Whether to include morphological labels

    Returns
    ===========
    instances : dict
        A dict representing all instances. Values under at
        least one of the following keys can be zipped to
        get the information for each instance:
        {'token': [t1, t2, t3], 'lemma': [l1, l2, l3],
         'pos': [p1, p2, p3], 'morph': [m1, m2, m3]}

        The `token` values will *always* be included.

    Notes
    ===========
    Supported input data formats are described below.
    """
    instances = {'token': []}
    if include_lemma:
        instances['lemma'] = []
    if include_pos:
        instances['pos'] = []
    if include_morph:
        instances['morph'] = []

    for root, dirs, files in os.walk(directory):
        for name in sorted(files):
            filepath = os.path.join(root, name)

            if not filepath.endswith(extensions):
                continue

            insts = load_annotated_file(filepath=filepath,
                                        format=format,
                                        nb_instances=nb_instances,
                                        include_lemma=include_lemma,
                                        include_morph=include_morph,
                                        include_pos=include_pos,
                                        col_pos = col_pos,
                                        col_lemma= col_lemma,
                                        col_morph = col_morph)

            instances['token'].extend(insts['token'])
            if include_lemma:
                instances['lemma'].extend(insts['lemma'])
            if include_pos:
                instances['pos'].extend(insts['pos'])
            if include_morph:
                instances['morph'].extend(insts['morph'])

    return instances


def load_annotated_file(filepath='text.txt', format='tab',
                        nb_instances=None, include_lemma=True,
                        include_morph=True, include_pos=True,
                        col_pos = None, col_lemma = None, col_morph=None):

    """Loads the annotated instances from a single file.

    Parameters
    ===========
    filpath : str (default: 'text.txt')
        Path to the file to load.
    format : str (default: tab)
        The annotation format used (see notes).
        Currently supported:
            - 'tab': tab-separated format
            - 'conll': the connl-shared task format
    nb_instances : int, optional (default: None)
        Max number of instances to load from the file.
        Cutoff for development purposes. All instances
        will be loaded if `None`.
    include_lemma : bool (default: True)
        Whether to include lemma labels
    include_pos : bool (default: True)
        Whether to include part-of-speech labels
    include_morph : bool (default: True)
        Whether to include morphological labels

    Returns
    ===========
    instances : dict
        A dict representing all instances. Values under at
        least of the following keys can be zipped to get the
        information for each token:
        {'token': [t1, t2, t3], 'lemma': [l1, l2, l3],
         'pos': [p1, p2, p3], 'morph': [m1, m2, m3]}

        The `token` values will *always* be included.

    Notes
    ===========
    Supported input data formats are described below. All files
    must be encoded using strict utf-8 only. The following
    formats are supported:
    - `connl` format:
        See http://universaldependencies.org/docs/format.html.
    - `tab` separated format (exactly 4 columns):
        * E.g.
            si   si  CON _
            autem   autem   CON _
            illis   ille    PRO gender=COMMON|case=DATIVE|number=PLURAL
            adhuc   adhuc   ADV degree=POSITIVE
            vita    vita    NN  gender=FEMININE|case=NOMINATIVE|number=SINGULAR

        * Morphological tags (e.g.) must be separated by pipes.
        * Please provide dummy fillers for each columns if you
          lack certain information; tokens must be included:
          E.g.
            si   _  CON _
            autem   _   CON _
            illis   _    PRO _
            adhuc   _   ADV _
            vita    _    NN  _
    """

    instances = {'token': []}
    if include_lemma:
        instances['lemma'] = []
    if include_pos:
        instances['pos'] = []
    if include_morph:
        instances['morph'] = []
    if format == 'conll':
        with codecs.open(filepath, 'r', 'utf8') as f:
            for line in f.readlines():
                line = line.strip()

                if line:
                    try:
                        idx, tok, _, lem, _, pos, morph = \
                            line.split()[:7]
                        print(idx, tok, lem, pos)
                        if include_lemma:
                            lem = lem.lower().strip().replace(' ', '')
                        tok = tok.strip().replace('~', '').replace(' ', '')
                        instances['token'].append(tok)
                        if include_lemma:
                            instances['lemma'].append(lem)
                        if include_pos:
                            instances['pos'].append(pos)
                        if include_morph:
                            instances['morph'].append(morph)
                    except ValueError:
                        pass
                if nb_instances:
                    if len(instances) >= nb_instances:
                        break

    elif format == 'tab':
        # Defaults
        if col_lemma is None:
            col_lemma = 1
        if col_pos is None:
            col_pos = 2
        if col_morph is None:
            col_morph = 3
        # Set from config file, if available
        
        with codecs.open(filepath, 'r', 'utf8') as cod:
            for line in cod:
                line = line.strip()
                if line and not line[0] == '@':
                    try:
                        comps = line.split()
                        tok = comps[0]
                        if include_lemma:
                            lem = comps[col_lemma].lower().strip()
                        if include_pos:
                            pos = comps[col_pos]
                        if include_morph:
                            morph = '|'.join(comps[col_morph].split('|'))
                        tok = tok.strip().replace('~', '').replace(' ', '')
                        instances['token'].append(tok)
                        if include_lemma:
                            instances['lemma'].append(lem)
                        if include_pos:
                            instances['pos'].append(pos)
                        if include_morph:
                            instances['morph'].append(morph)
                    except:
                        print(filepath, ':', line)
                if nb_instances:
                    if len(instances['token']) >= nb_instances:
                        break

    else:
        raise NotImplementedError('Unsupported file format: must be\
                                  one of: conll, tab')

    return instances


def load_unannotated_file(filepath='test.txt',
                          nb_instances=None,
                          tokenized_input=False):
    """Loads unannotated instances from a single file.

    Parameters
    ===========
    filepath : str (default: 'text.txt')
        Path to the unannotated file to load.
    nb_instances : int, optional (default: None)
        Max number of instances to load from the file.
        Cutoff for development purposes. All instances
        will be loaded if `None`.
    tokenized_input : bool (default: False)
        Whether the input data is already tokenized.
        * If `True`, one token per line is expected.
        * Else input file is tokenized using nltk's
          generic `nltk.tokenize.wordpunct_tokenize`


    Returns
    ===========
    tokens : list
        A list of the tokens.

    """
    if tokenized_input:
        tokens = []
        for line in codecs.open(filepath, 'r', 'utf8'):
            line = line.strip()
            if line:
                tokens.append(line)
            if nb_instances:
                nb_instances -= 1
                if nb_instances <= 0:
                    break
        return tokens
    else:
        from nltk.tokenize import wordpunct_tokenize
        W = re.compile(r'\s+')
        with codecs.open(filepath, 'r', 'utf8') as f:
            text = W.sub(f.read(), ' ')
        tokens = wordpunct_tokenize(text)
        if nb_instances:
            return tokens[:nb_instances]
        else:
            return tokens


def stats(tokens, lemmas, known):

    """Prints simple stats for a list of
    tokens and lemmas.

    Parameters
    ===========
    tokens : list-like
        The tokens.
    lemmas : list-like
        A list of correpsonding lemma labels
    known : set-like
        A iterable of lemmas considered 'known',
        i.e. seen during training

    """

    print('Nb of tokens:', len(tokens))
    print('Nb of unique tokens:', len(set(tokens)))
    cnt = sum([1.0 for k in tokens if k not in known]) / len(tokens) * 100.0
    print('Nb of unseen tokens:', cnt)
    print('Nb of unique lemmas: ', len(set(lemmas)))


def get_param_dict(p):
    """Loads and parses a parameter file.

    Parameters
    ===========
    p : str
        The path to the parameter file.

    Returns
    ===========
    param_dict : dict
        * A dict the parameters.
        * Instances of 'True' and 'False' are
        casted to bools.
    """
    def read_sections(config):
        param_dict = dict()
        for section in config.sections():
            for name, value in config.items(section):
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                elif value == 'None':
                    value = None
                param_dict[name] = value
        return param_dict

    try:
        config = ConfigParser.ConfigParser()
        config.read(p)
        return read_sections(config)
    except Exception as e:
        raise ValueError(
            "Couldn't read config file: %s. Exception: %s" % (p, str(e)))


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    (Borrowed from keras to avoid a direct dependency.)

    Parameters
    ===========
    y : 1d array
        class vector to be converted into a matrix
        (integers from 0 to num_classes).
    num_classes : int (optional)
        Total number of distinct classes. Will be
        inferred if not provided.

    Returns
    ===========
    categorical : 2D array of shape (nb_samples, nb_classes)
        A binary matrix representation of the input.
    """

    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
