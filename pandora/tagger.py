#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import shutil

import numpy as np

import editdistance

import pandora.utils as utils
import pandora.evaluation as evaluation
from pandora.impl import KerasModel, PyTorchModel
from pandora.preprocessing import Preprocessor
from pandora.pretraining import Pretrainer
from pandora.logger import Logger


MODELS = {KerasModel.CONFIG_KEY: KerasModel, PyTorchModel.CONFIG_KEY: PyTorchModel}


class Tagger():
    def __init__(self,
                 config_path=None,
                 nb_encoding_layers=1,
                 nb_dense_dims=30,
                 char_embed_dim=50,
                 batch_size=100,
                 nb_left_tokens=2,
                 nb_right_tokens=2,
                 nb_embedding_dims=150,
                 model_dir='new_model',
                 postcorrect=True,
                 include_token=True,
                 include_context=True,
                 include_lemma=True,
                 include_pos=True,
                 include_morph=True,
                 include_dev=True,
                 include_test=True,
                 nb_filters=100,
                 filter_length=3,
                 focus_repr='recurrent',
                 dropout_level=.1,
                 load=False,
                 nb_epochs=15,
                 min_token_freq_emb=5,
                 halve_lr_at=10,
                 max_token_len=None,
                 max_lemma_len=None,
                 min_lem_cnt=1,
                 overwrite=None,
                 lr=0.01,
                 model='Keras',
                 test_batch_size=None,
                 col_pos=None,
                 col_lemma=None,
                 col_morph=None):

        # initialize:
        self.setup = False
        self.curr_nb_epochs = 0

        self.train_tokens, self.dev_tokens, self.test_tokens = None, None, None
        self.train_lemmas, self.dev_lemmas, self.test_lemmas = None, None, None
        self.train_pos, self.dev_pos, self.test_pos = None, None, None
        self.train_morph, self.dev_morph, self.test_morph = None, None, None
        self.embed_tokens = None

        self.logger = Logger()  # Default logger uses shell

        self.test_batch_size = test_batch_size or batch_size

        if MODELS[model] is None:
            raise ValueError("Couldn't load implementation {}".format(model))

        if config_path is not None:
            param_dict = utils.get_param_dict(config_path)
        elif load is True:
            if model_dir:
                config_path = os.sep.join((model_dir, 'config.txt'))
                param_dict = utils.get_param_dict(config_path)
                print('Using params from config file: %s' % config_path)
            else:
                raise ValueError(
                    'To load a tagger you, must specify a model_name!')
        else:
            param_dict = {}

        self.nb_encoding_layers = \
            int(param_dict.get('nb_encoding_layers', nb_encoding_layers))
        self.nb_dense_dims = \
            int(param_dict.get('nb_dense_dims', nb_dense_dims))
        self.batch_size = int(param_dict.get('batch_size', batch_size))
        self.nb_epochs = int(param_dict.get('nb_epochs', nb_epochs))
        self.nb_left_tokens = \
            int(param_dict.get('nb_left_tokens', nb_left_tokens))
        self.nb_right_tokens = \
            int(param_dict.get('nb_right_tokens', nb_right_tokens))
        self.nb_context_tokens = self.nb_left_tokens + self.nb_right_tokens
        self.nb_embedding_dims = \
            int(param_dict.get('nb_embedding_dims', nb_embedding_dims))
        self.model_dir = param_dict.get('model_dir', model_dir)
        self.postcorrect = bool(param_dict.get('postcorrect', postcorrect))
        self.nb_filters = int(param_dict.get('nb_filters', nb_filters))
        self.filter_length = \
            int(param_dict.get('filter_length', filter_length))
        self.focus_repr = param_dict.get('focus_repr', focus_repr)
        self.dropout_level = \
            float(param_dict.get('dropout_level', dropout_level))
        self.include_token = param_dict.get('include_token', include_token)
        self.include_context = \
            param_dict.get('include_context', include_context)
        self.include_lemma = param_dict.get('include_lemma', include_lemma)
        self.include_pos = param_dict.get('include_pos', include_pos)
        self.include_morph = param_dict.get('include_morph', include_morph)
        self.include_dev = param_dict.get('include_dev', include_dev)
        self.include_test = param_dict.get('include_test', include_test)
        self.min_token_freq_emb = \
            int(param_dict.get('min_token_freq_emb', min_token_freq_emb))
        self.halve_lr_at = int(param_dict.get('halve_lr_at', halve_lr_at))
        self.lr = float(param_dict.get('lr', lr))
        self.max_token_len = param_dict.get('max_token_len', max_token_len)
        if self.max_token_len is not None:
            self.max_token_len = int(self.max_token_len)

        self.max_lemma_len = param_dict.get('max_lemma_len', max_lemma_len)
        if self.max_lemma_len is not None:
            self.max_lemma_len = int(self.max_lemma_len)

        self.min_lem_cnt = int(param_dict.get('min_lem_cnt', min_lem_cnt))
        self.char_embed_dim = \
            int(param_dict.get('char_embed_dim', char_embed_dim))
        self.curr_nb_epochs = int(param_dict.get('curr_nb_epochs', 0))
        self.test_batch_size = int(param_dict.get('test_batch_size', self.test_batch_size))
        self.model = param_dict.get('model', model)

        if overwrite is not None:
            # Overwrite should be a dict of attributes to update the tagger
            for key, value in overwrite.items():
                self.__setattr__(key, value)

        # create a models directory if it isn't there already
        if not os.path.isdir(self.model_dir):
            os.mkdir(model_dir)

        if load:
            self.load()

    @staticmethod
    def setup_from_disk(
            config_path,
            train_data=None, dev_data=None, test_data=None, verbose=False,
            load=False, embed=False, **kwargs
    ):
        """ Command to load the whole tagger through a config.txt file and the path to some data

        :param config_path:
        :param train_data:
        :param dev_data:
        :param test_data:
        :param verbose:
        :param load:
        :param embed:
        :return:
        :rtype: Tagger
        """
        if verbose:
            print('::: started :::')

        orig_params = utils.get_param_dict(config_path)
        params = {
            k: v
            for k, v in orig_params.items()
            if v is not None and k not in ["pretrainer_nb_workers"]
        }
        params['config_path'] = config_path
        params.update({
            k: v
            for k, v in kwargs.items()
            if v is not None and k not in ["pretrainer_nb_workers"]
        })

        nb_pretrainer_workers = int(kwargs.get(
            "pretrainer_nb_workers",
            orig_params.get(
                "pretrainer_nb_workers",
                Pretrainer.DEFAULT_NB_WORKERS
            )
        ))
        if "col_pos" not in params:
            params["col_pos"] = None
        if "col_lemma" not in params:
            params["col_lemma"] = None
        if "col_morph" not in params:
            params["col_morph"] = None

        if verbose:
            print("::: Loaded Config :::")
            for k, v in params.items():
                print("\t{} : {}".format(k, v))

        train_data = utils.load_annotated_dir(
            directory=train_data,
            format='tab',
            extensions=('.tab','.tsv'),
            include_pos=params['include_pos'],
            include_lemma=params['include_lemma'],
            include_morph=params['include_morph'],
            nb_instances=None,
            col_pos = int(params["col_pos"]) if params["col_pos"] is not None else None,
            col_lemma = int(params["col_lemma"]) if params["col_lemma"] is not None else None,
            col_morph = int(params["col_morph"]) if params["col_morph"] is not None else None
        )

        if not len(train_data.keys()) or \
                not len(train_data[list(train_data.keys())[0]]):
            raise ValueError('No training data loaded...')

        data_sets = dict(
            train_data=train_data,
        )

        if dev_data:
            dev_data = utils.load_annotated_dir(
                directory=dev_data,
                format='tab',
                extensions=('.tab','.tsv'),
                include_pos=params['include_pos'],
                include_lemma=params['include_lemma'],
                include_morph=params['include_morph'],
                nb_instances=None,
                col_pos = int(params["col_pos"]) if params["col_pos"] is not None else None,
                col_lemma = int(params["col_lemma"]) if params["col_lemma"] is not None else None,
                col_morph = int(params["col_morph"]) if params["col_morph"] is not None else None
            )
            if not len(dev_data.keys()) or \
                    not len(dev_data[list(dev_data.keys())[0]]):
                raise ValueError('No dev data loaded...')
            data_sets["dev_data"] = dev_data

        if test_data:
            test_data = utils.load_annotated_dir(
                directory=test_data,
                format='tab',
                extensions=('.tab','.tsv'),
                include_pos=params['include_pos'],
                include_lemma=params['include_lemma'],
                include_morph=params['include_morph'],
                nb_instances=None,
                col_pos = int(params["col_pos"]) if params["col_pos"] is not None else None,
                col_lemma = int(params["col_lemma"]) if params["col_lemma"] is not None else None,
                col_morph = int(params["col_morph"]) if params["col_morph"] is not None else None
            )
            if not len(test_data.keys()) or \
                    not len(test_data[list(test_data.keys())[0]]):
                raise ValueError('No test data loaded...')
            data_sets["test_data"] = test_data

        if embed:
            embed_data = utils.load_annotated_dir(
                embed,
                format='tab',
                extensions=('.tab','.tsv'),
                include_pos=False,
                include_lemma=False,
                include_morph=False,
                nb_instances=None
            )
            if not len(embed_data.keys()) or \
                    not len(embed_data[list(embed_data.keys())[0]]):
                raise ValueError('No embeddings data loaded...')
            data_sets["embed_data"] = embed_data

        if load:
            if verbose:
                print('::: loading model :::')
            tagger = Tagger(load=True, model_dir=params['model_dir'])
            if tagger.config_path == os.sep.join((tagger.model_dir, 'config.txt')):
                shutil.copy(tagger.config_path, os.sep.join((tagger.model_dir, 'config_original.txt')))
                if verbose:
                    print('Warning: current config file will be overwritten. Saving it to config_original.txt')
            tagger.setup_to_train(
                build=False,
                nb_pretrainer_workers=nb_pretrainer_workers,
                **data_sets
            )
            tagger.curr_nb_epochs = int(params['curr_nb_epochs'])
            if verbose:
                print("restart from epoch " + str(tagger.curr_nb_epochs) + "...")
            tagger.setup = True
        else:
            tagger = Tagger(**{k: v for k, v in params.items()})
            tagger.setup_to_train(
                nb_pretrainer_workers=nb_pretrainer_workers,
                **data_sets
            )

        return tagger

    def load(self):
        print('Re-loading preprocessor...')
        self.preprocessor = Preprocessor(categorical=self.model == 'Keras')
        self.preprocessor.load(model_dir=self.model_dir,
                               max_token_len=self.max_token_len,
                               max_lemma_len=self.max_lemma_len,
                               focus_repr=self.focus_repr,
                               include_lemma=self.include_lemma,
                               include_pos=self.include_pos,
                               include_morph=self.include_morph)

        print('Re-loading pretrainer...')
        self.pretrainer = Pretrainer()
        self.pretrainer.load(model_dir=self.model_dir,
                             nb_left_tokens=self.nb_left_tokens,
                             nb_right_tokens=self.nb_right_tokens)

        print('Re-building model...')
        self.model = MODELS[self.model].load(
            self.model_dir,
            include_lemma=self.include_lemma,
            include_pos=self.include_pos,
            include_morph=self.include_morph)

        if self.include_lemma:
            print('Loading known lemmas...')
            lemmas_path = os.sep.join((self.model_dir, 'known_lemmas.txt'))
            with open(lemmas_path, 'r') as f:
                self.known_lemmas = set([l.strip() for l in f.readlines()])

    def setup_to_train(self, train_data=None, dev_data=None, test_data=None, embed_data=None,
                       build=True, nb_pretrainer_workers=None):
        if build:
            # create a model directory:
            if os.path.isdir(self.model_dir):
                shutil.rmtree(self.model_dir)
            os.mkdir(self.model_dir)

        self.train_tokens = train_data['token']
        if self.include_test:
            self.test_tokens = test_data['token']
        if self.include_dev:
            self.dev_tokens = dev_data['token']

        if self.include_lemma:
            self.train_lemmas = train_data['lemma']
            self.known_lemmas = set(self.train_lemmas)
            if self.include_dev:
                self.dev_lemmas = dev_data['lemma']
            if self.include_test:
                self.test_lemmas = test_data['lemma']

        if self.include_pos:
            self.train_pos = train_data['pos']
            if self.include_dev:
                self.dev_pos = dev_data['pos']
            if self.include_test:
                self.test_pos = test_data['pos']

        if self.include_morph:
            self.train_morph = train_data['morph']
            if self.include_dev:
                self.dev_morph = dev_data['morph']
            if self.include_test:
                self.test_morph = test_data['morph']

        if embed_data:
            self.embed_tokens = embed_data['token']
            #TODO: concatenate with train tokens ?
            #in that case:
            #self.embed_tokens = dict(embed_data['tokens'].items() + train_data['token'].items() )

        self.preprocessor = Preprocessor(categorical=self.model == 'Keras')
        self.preprocessor.fit(
            tokens=self.train_tokens,
            lemmas=self.train_lemmas,
            pos=self.train_pos,
            morph=self.train_morph,
            include_lemma=self.include_lemma,
            include_morph=self.include_morph,
            max_token_len=self.max_token_len,
            max_lemma_len=self.max_lemma_len,
            focus_repr=self.focus_repr,
            min_lem_cnt=self.min_lem_cnt)

        self.max_token_len = self.preprocessor.max_token_len
        self.max_lemma_len = self.preprocessor.max_lemma_len
        self.pretrainer = Pretrainer(nb_left_tokens=self.nb_left_tokens,
                                     nb_right_tokens=self.nb_right_tokens,
                                     size=self.nb_embedding_dims,
                                     minimum_count=self.min_token_freq_emb,
                                     nb_workers=nb_pretrainer_workers)

        if self.embed_tokens is not None:
            self.pretrainer.fit(tokens=self.embed_tokens)
        else:
            self.pretrainer.fit(tokens=self.train_tokens)

        train_transformed = self.preprocessor.transform(
            tokens=self.train_tokens,
            lemmas=self.train_lemmas,
            pos=self.train_pos,
            morph=self.train_morph)

        if self.include_dev:
            dev_transformed = self.preprocessor.transform(
                tokens=self.dev_tokens,
                lemmas=self.dev_lemmas,
                pos=self.dev_pos,
                morph=self.dev_morph)

        if self.include_test:
            test_transformed = self.preprocessor.transform(
                tokens=self.test_tokens,
                lemmas=self.test_lemmas,
                pos=self.test_pos,
                morph=self.test_morph)

        self.train_X_focus = train_transformed['X_focus']
        if self.include_dev:
            self.dev_X_focus = dev_transformed['X_focus']
        if self.include_test:
            self.test_X_focus = test_transformed['X_focus']

        if self.include_lemma:
            self.train_X_lemma = train_transformed['X_lemma']
            if self.include_dev:
                self.dev_X_lemma = dev_transformed['X_lemma']
            if self.include_test:
                self.test_X_lemma = test_transformed['X_lemma']

        if self.include_pos:
            self.train_X_pos = train_transformed['X_pos']
            if self.include_dev:
                self.dev_X_pos = dev_transformed['X_pos']
            if self.include_test:
                self.test_X_pos = test_transformed['X_pos']

        if self.include_morph:
            self.train_X_morph = train_transformed['X_morph']
            if self.include_dev:
                self.dev_X_morph = dev_transformed['X_morph']
            if self.include_test:
                self.test_X_morph = test_transformed['X_morph']

        self.train_contexts = self.pretrainer.transform(
            tokens=self.train_tokens)
        if self.include_dev:
            self.dev_contexts = self.pretrainer.transform(
                tokens=self.dev_tokens)
        if self.include_test:
            self.test_contexts = self.pretrainer.transform(
                tokens=self.test_tokens)
        try:
            del train_transformed
        except:
            pass
        try:
            del dev_transformed
        except:
            pass
        try:
            del test_transformed
        except:
            pass

        print('Building model...')
        nb_tags = None
        try:
            nb_tags = len(self.preprocessor.pos_encoder.classes_)
        except AttributeError:
            pass
        nb_morph_cats = None
        try:
            nb_morph_cats = self.preprocessor.nb_morph_cats
        except AttributeError:
            pass
        max_token_len, token_char_dict = None, None
        try:
            max_token_len = self.preprocessor.max_token_len
            token_char_dict = self.preprocessor.token_char_lookup
        except AttributeError:
            pass
        max_lemma_len, lemma_char_dict = None, None
        try:
            max_lemma_len = self.preprocessor.max_lemma_len
            lemma_char_dict = self.preprocessor.lemma_char_lookup
        except AttributeError:
            pass
        nb_lemmas = None
        try:
            nb_lemmas = len(self.preprocessor.lemma_encoder.classes_)
        except AttributeError:
            pass

        self.model = MODELS[self.model](
            token_len=max_token_len,
            token_char_vector_dict=token_char_dict,
            lemma_len=max_lemma_len,
            nb_tags=nb_tags,
            nb_morph_cats=nb_morph_cats,
            lemma_char_vector_dict=lemma_char_dict,
            nb_encoding_layers=self.nb_encoding_layers,
            nb_dense_dims=self.nb_dense_dims,
            nb_embedding_dims=self.nb_embedding_dims,
            nb_train_tokens=len(self.pretrainer.train_token_vocab),
            nb_context_tokens=self.nb_context_tokens,
            pretrained_embeddings=self.pretrainer.pretrained_embeddings,
            include_token=self.include_token,
            include_context=self.include_context,
            include_lemma=self.include_lemma,
            include_pos=self.include_pos,
            include_morph=self.include_morph,
            nb_filters=self.nb_filters,
            filter_length=self.filter_length,
            focus_repr=self.focus_repr,
            dropout_level=self.dropout_level,
            nb_lemmas=nb_lemmas,
            char_embed_dim=self.char_embed_dim,
            batch_size=self.batch_size,
            lr=self.lr)

        self.model.print_summary()

        self.setup = True
        self.save()

    def train(self, nb_epochs=None):
        if nb_epochs:
            self.nb_epochs = nb_epochs
        for i in range(self.nb_epochs):
            scores = self.epoch()
        return scores

    def print_stats(self):
        print('Train stats:')
        utils.stats(tokens=self.train_tokens,
                    lemmas=self.train_lemmas,
                    known=self.preprocessor.known_tokens)
        print('Test stats:')
        utils.stats(tokens=self.test_tokens,
                    lemmas=self.test_lemmas,
                    known=self.preprocessor.known_tokens)

    def test(self, multilabel_threshold=0.5):
        """ Run tests evaluation with prediction

        :param multilabel_threshold:
        :return: Score Dictionary
        """
        if not self.include_test:
            raise ValueError(
                'Please do not call .test() if no test data is available.')

        score_dict = {}

        # get test predictions:
        test_in = {}
        if self.include_token:
            test_in['focus_in'] = self.test_X_focus
        if self.include_context:
            test_in['context_in'] = self.test_contexts

        test_preds = self.model.predict(test_in, batch_size=self.test_batch_size)

        if isinstance(test_preds, np.ndarray):
            test_preds = [test_preds]

        if self.include_lemma:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(
                predictions=test_preds['lemma_out'])
            if self.postcorrect:
                for i in range(len(pred_lemmas)):
                    if pred_lemmas[i] not in self.known_lemmas:
                        pred_lemmas[i] = min(
                            self.known_lemmas,
                            key=lambda x: editdistance.eval(x, pred_lemmas[i]))

            score_dict['test_lemma'] = evaluation.single_label_accuracies(
                gold=self.test_lemmas,
                silver=pred_lemmas,
                test_tokens=self.test_tokens,
                known_tokens=self.preprocessor.known_tokens,
                print_scores=False)

        if self.include_pos:
            pred_pos = self.preprocessor.inverse_transform_pos(
                predictions=test_preds['pos_out'])
            score_dict['test_pos'] = evaluation.single_label_accuracies(
                gold=self.test_pos,
                silver=pred_pos,
                test_tokens=self.test_tokens,
                known_tokens=self.preprocessor.known_tokens,
                print_scores=False)

        if self.include_morph:
            pred_morph = self.preprocessor.inverse_transform_morph(
                predictions=test_preds['morph_out'],
                threshold=multilabel_threshold)
            if self.include_morph == 'label':
                score_dict['test_morph'] = evaluation.single_label_accuracies(
                    gold=self.test_morph,
                    silver=pred_morph,
                    test_tokens=self.test_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)
            elif self.include_morph == 'multilabel':
                score_dict['test_morph'] = evaluation.multilabel_accuracies(
                    gold=self.test_morph,
                    silver=pred_morph,
                    test_tokens=self.test_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

        return score_dict

    def save(self):
        # save architecture:
        self.model.save(self.model_dir)

        # save preprocessor:
        self.preprocessor.save(self.model_dir)
        self.pretrainer.save(self.model_dir)

        if self.include_lemma:
            lemmas_path = os.sep.join((self.model_dir, 'known_lemmas.txt'))
            with open(lemmas_path, 'w') as f:
                f.write('\n'.join(sorted(self.known_lemmas)))

        self.save_params()

    def save_params(self):
        """ Write the params to self.model_dir """
        with open(os.sep.join((self.model_dir, 'config.txt')), 'w') as F:
            F.write('# Parameter file\n\n[global]\n')
            F.write('nb_encoding_layers = '+str(self.nb_encoding_layers)+'\n')
            F.write('nb_dense_dims = '+str(self.nb_dense_dims)+'\n')
            F.write('batch_size = '+str(self.batch_size)+'\n')
            F.write('nb_left_tokens = '+str(self.nb_left_tokens)+'\n')
            F.write('nb_right_tokens = '+str(self.nb_right_tokens)+'\n')
            F.write('nb_embedding_dims = '+str(self.nb_embedding_dims)+'\n')
            F.write('model_dir = '+str(self.model_dir)+'\n')
            F.write('postcorrect = '+str(self.postcorrect)+'\n')
            F.write('nb_filters = '+str(self.nb_filters)+'\n')
            F.write('filter_length = '+str(self.filter_length)+'\n')
            F.write('focus_repr = '+str(self.focus_repr)+'\n')
            F.write('dropout_level = '+str(self.dropout_level)+'\n')
            F.write('include_token = '+str(self.include_context)+'\n')
            F.write('include_context = '+str(self.include_context)+'\n')
            F.write('include_lemma = '+str(self.include_lemma)+'\n')
            F.write('include_pos = '+str(self.include_pos)+'\n')
            F.write('include_morph = '+str(self.include_morph)+'\n')
            F.write('include_dev = '+str(self.include_dev)+'\n')
            F.write('include_test = '+str(self.include_test)+'\n')
            F.write('nb_epochs = '+str(self.nb_epochs)+'\n')
            F.write('halve_lr_at = '+str(self.halve_lr_at)+'\n')
            F.write('max_token_len = '+str(self.max_token_len)+'\n')
            F.write('max_lemma_len = '+str(self.max_lemma_len)+'\n')
            F.write('min_token_freq_emb = '+str(self.min_token_freq_emb)+'\n')
            F.write('min_lem_cnt = '+str(self.min_lem_cnt)+'\n')
            F.write('char_embed_dim = '+str(self.char_embed_dim)+'\n')
            F.write('test_batch_size = '+str(self.test_batch_size)+'\n')
            F.write('model = '+str(self.model.CONFIG_KEY)+'\n')
            if self.pretrainer:
                F.write('pretrainer_nb_workers = '+str(self.pretrainer.nb_workers)+'\n')
            if hasattr(self, "curr_nb_epochs"):
                F.write('curr_nb_epochs = '+str(self.curr_nb_epochs)+'\n')

    def epoch(self, autosave=True, eval_test=False):
        if not self.setup:
            raise ValueError(
                'Not set up yet... Call Tagger.setup_to_train() first.')

        # update nb of epochs ran so far:
        self.curr_nb_epochs += 1
        print("-> Epoch ", self.curr_nb_epochs, "...")

        if self.curr_nb_epochs and self.halve_lr_at:
            # update learning rate at specific points:
            if self.curr_nb_epochs % self.halve_lr_at == 0:
                self.model.adjust_lr()

        # get inputs and outputs straight:
        train_in, train_out = {}, {}
        if self.include_token:
            train_in['focus_in'] = self.train_X_focus
        if self.include_context:
            train_in['context_in'] = self.train_contexts
        if self.include_lemma:
            train_out['lemma_out'] = self.train_X_lemma
        if self.include_pos:
            train_out['pos_out'] = self.train_X_pos
        if self.include_morph:
            train_out['morph_out'] = self.train_X_morph

        self.model.epoch(train_in, train_out)

        def run_eval():
            score_dict = self.eval(train_in=train_in)
            if eval_test is True:
                score_dict.update(self.test())
            return score_dict

        score_dict = self.logger.epoch(self.curr_nb_epochs, run_eval)

        if autosave:
            self.save()

        return score_dict

    def eval(self, train_in):
        """ Evaluate current epoch

        :param train_in: Data from training
        :return:
        """

        # get train preds:
        train_preds = self.model.predict(train_in, batch_size=self.test_batch_size)
        if isinstance(train_preds, np.ndarray):
            train_preds = [train_preds]

        # get dev preds:
        dev_preds = None
        if self.include_dev:
            dev_in = {}
            if self.include_token:
                dev_in['focus_in'] = self.dev_X_focus
            if self.include_context:
                dev_in['context_in'] = self.dev_contexts

            dev_preds = self.model.predict(dev_in, batch_size=self.test_batch_size)

            if isinstance(dev_preds, np.ndarray):
                dev_preds = [dev_preds]

        return self.get_score_dict(train_preds, dev_preds)

    def get_score_dict(self, train_preds, dev_preds=None):
        """ Get score dict given train and dev preds

        :param train_preds: Training predictions
        :param dev_preds: Development predictions
        :return: Score dictionary
        """

        score_dict = {}
        if self.include_lemma:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(
                predictions=train_preds['lemma_out'])
            score_dict['train_lemma'] = evaluation.single_label_accuracies(
                gold=self.train_lemmas,
                silver=pred_lemmas,
                test_tokens=self.train_tokens,
                known_tokens=self.preprocessor.known_tokens,
                print_scores=False)

            if self.include_dev:
                pred_lemmas = self.preprocessor.inverse_transform_lemmas(
                    predictions=dev_preds['lemma_out'])
                score_dict['dev_lemma'] = evaluation.single_label_accuracies(
                    gold=self.dev_lemmas,
                    silver=pred_lemmas,
                    test_tokens=self.dev_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

                if self.postcorrect:
                    for i in range(len(pred_lemmas)):
                        if pred_lemmas[i] not in self.known_lemmas:
                            pred_lemmas[i] = min(
                                self.known_lemmas,
                                key=lambda x: editdistance.eval(x, pred_lemmas[i]))

                    score_dict['dev_lemma_postcorrect'] = evaluation.single_label_accuracies(
                        gold=self.dev_lemmas,
                        silver=pred_lemmas,
                        test_tokens=self.dev_tokens,
                        known_tokens=self.preprocessor.known_tokens)

        if self.include_pos:

            pred_pos = self.preprocessor.inverse_transform_pos(
                predictions=train_preds['pos_out'])
            score_dict['train_pos'] = evaluation.single_label_accuracies(
                gold=self.train_pos,
                silver=pred_pos,
                test_tokens=self.train_tokens,
                known_tokens=self.preprocessor.known_tokens,
                print_scores=False)

            if self.include_dev:
                pred_pos = self.preprocessor.inverse_transform_pos(
                    predictions=dev_preds['pos_out'])
                score_dict['dev_pos'] = evaluation.single_label_accuracies(
                    gold=self.dev_pos,
                    silver=pred_pos,
                    test_tokens=self.dev_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

        if self.include_morph:

            pred_morph = self.preprocessor.inverse_transform_morph(
                predictions=train_preds['morph_out'])

            if self.include_morph == 'label':
                score_dict['train_morph'] = evaluation.single_label_accuracies(
                    gold=self.train_morph,
                    silver=pred_morph,
                    test_tokens=self.train_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

            elif self.include_morph == 'multilabel':
                score_dict['train_morph'] = evaluation.multilabel_accuracies(
                    gold=self.train_morph,
                    silver=pred_morph,
                    test_tokens=self.train_tokens,
                    known_tokens=self.preprocessor.known_tokens,
                    print_scores=False)

            if self.include_dev:

                pred_morph = self.preprocessor.inverse_transform_morph(
                    predictions=dev_preds['morph_out'])

                if self.include_morph == 'label':
                    score_dict['dev_morph'] = evaluation.single_label_accuracies(
                        gold=self.train_morph,
                        silver=pred_morph,
                        test_tokens=self.dev_tokens,
                        known_tokens=self.preprocessor.known_tokens,
                        print_scores=False)

                elif self.include_morph == 'multilabel':
                    score_dict['dev_morph'] = evaluation.multilabel_accuracies(
                        gold=self.train_morph,
                        silver=pred_morph,
                        test_tokens=self.dev_tokens,
                        known_tokens=self.preprocessor.known_tokens,
                        print_scores=False)

        return score_dict

    def annotate(self, tokens):
        X_focus = self.preprocessor.transform(tokens=tokens)['X_focus']
        X_context = self.pretrainer.transform(tokens=tokens)

        new_in = {}
        if self.include_token:
            new_in['focus_in'] = X_focus
        if self.include_context:
            new_in['context_in'] = X_context

        preds = self.model.predict(new_in)
        if isinstance(preds, np.ndarray):
            preds = [preds]

        annotation_dict = {'tokens': tokens}
        if self.include_lemma:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(
                predictions=preds['lemma_out'])
            annotation_dict['lemmas'] = pred_lemmas
            if self.postcorrect:
                for i in range(len(pred_lemmas)):
                    if pred_lemmas[i] not in self.known_lemmas:
                        pred_lemmas[i] = min(
                            self.known_lemmas,
                            key=lambda x: editdistance.eval(x, pred_lemmas[i]))
                annotation_dict['postcorrect_lemmas'] = pred_lemmas

        if self.include_pos:
            pred_pos = self.preprocessor.inverse_transform_pos(
                predictions=preds['pos_out'])
            annotation_dict['pos'] = pred_pos

        if self.include_morph:
            pred_morph = self.preprocessor.inverse_transform_morph(
                predictions=preds['morph_out'])
            annotation_dict['morph'] = pred_morph

        return annotation_dict
