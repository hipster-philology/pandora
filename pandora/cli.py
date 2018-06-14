from __future__ import print_function
import argparse
import pandora.utils
from pandora.logger import Logger
from pandora.tagger import Tagger
import os
import shutil
import codecs
import re


def train_func(config, train, dev=None, test=None, load=False, verbose=True, first=1, each=1, eval_file=None, no_shell=False, **kwargs):
    """ Main CLI Interface (training)

    :param config: Path to retrieve configuration file
    :type config: str
    :param train: Path to directory containing dev files
    :type train: str
    :param dev: Path to directory containing test files
    :type dev: str
    :param test: Path to directory containing train files
    :type test: str
    :param embed: Path to directory containing files for embeddings
    :type embed: str
    :param load: Whether to load an existing model to train on top of it (default: False)
    :type load: bool
    :param nb_epochs: Number of epoch
    :type nb_epochs: int
    :param verbose: (Overwrite the next few) Print only the first and last if False
    :param first: Evaluate first N epochs
    :param each: Evaluate each Nth epoch
    :param eval_file: Store evaluation into a file
    :param no_shell: Do not print to shell
    :param kwargs: Other arguments
    :type kwargs: dict
    :return:
    """
    tagger = Tagger.setup_from_disk(config, train, dev, test, verbose=True, load=load, **kwargs)

    nb_epochs = tagger.nb_epochs

    # Set up the logger
    logger_params = dict(
        shell=not no_shell,
        file=eval_file,
        first=first,
        nb_epochs=nb_epochs,
        each=each
    )
    if verbose is False:
        #  Print each total number of epoch + 1 to not print any
        logger_params = dict(shell=True, file=logger_params["file"], first=1, nb_epochs=nb_epochs, each=nb_epochs+1)
    tagger.logger = Logger(**logger_params)

    for i in range(nb_epochs):
        tagger.epoch(autosave=True, eval_test=tagger.include_test)

    tagger.save()
    print('::: ended :::')


def cli_train():
    parser = argparse.ArgumentParser(description="Training interface of Pandora")
    parser.add_argument("config", help="Path to retrieve configuration file")
    parser.add_argument("--dev", help="Path to directory containing dev files")
    parser.add_argument("--test", help="Path to directory containing test files")
    parser.add_argument("--train", help="Path to directory containing train files")
    parser.add_argument("--embed", help="Path to directory containing files for the embeddings")
    parser.add_argument("--nb_epochs", help="Number of epoch", type=int)
    parser.add_argument(
        "--load",
        dest="load",
        action="store_true",
        default=False,
        help="Whether to load an existing model to train on top of it (default: False)"
    )
    parser.add_argument(
        "--silent",
        dest="verbose",
        action="store_false",
        default=True,
        help="Stop printing results at each epoch"
    )
    parser.add_argument("--each", help="Print evaluation each X epochs", type=int, default=1)
    parser.add_argument("--first", help="Print first X epochs", type=int, default=1)
    parser.add_argument("--eval_file", help="Save evaluations to a CSV", type=str, default=None)
    parser.add_argument("--no_shell", help="Do not print evaluation", default=False, action="store_true")

    train_func(**vars(parser.parse_args()))


tokenize = re.compile("\s")


def tag_dir(model, input_dir, output_dir, tokenized_input, string=None, **kwargs):
    """ Tag a directory of texts

    :param model: Path to a model file
    :param input_dir: Path to a directory containing text files
    :param output_dir: Path to output tagged text files
    """
    print('::: started :::')

    tagger = Tagger(load=True, model_dir=model, overwrite=kwargs)
    print('Tagger loaded, now annotating...')

    orig_path = input_dir
    new_path = output_dir

    for filename in os.listdir(orig_path):
        if not filename.endswith('.txt'):
            continue

        print('\t +', filename)
        unseen_tokens = pandora.utils.load_unannotated_file(
            os.path.join(orig_path, filename),
            nb_instances=None,
            tokenized_input=tokenized_input
        )

        annotations = tagger.annotate(unseen_tokens)
        keys = list(annotations.keys())
        print("Keys :" + "\t".join(keys))
        with codecs.open(os.path.join(new_path, filename + ".tsv"), 'w', 'utf8') as f:
            f.write("\t".join(keys) + "\n")
            for x in zip(*tuple([annotations[k] for k in keys])):
                f.write('\t'.join(list(x)) + '\n')

    print('::: ended :::')


def tag_string(model, input_dir, output_dir=None, string=None, **kwargs):
    """ Tag a directory of texts

    :param model: Path to a model file
    :param input_dir: Untokenized string to tag
    """

    print('::: started :::')

    tagger = Tagger(load=True, model_dir=model, overwrite=kwargs)

    print('Tagger loaded, now annotating...')

    unseen_tokens = tokenize.split(input_dir)
    print(unseen_tokens)

    annotations = tagger.annotate(unseen_tokens)

    keys = list(annotations.keys())
    print("--------------------")
    print('\t'.join(keys))
    print("--------------------")
    for x in zip(*tuple([annotations[k] for k in keys])):
        print('\t'.join(list(x)))

    print('::: ended :::')


def cli_tagger():
    """ Functions that runs and takes console input for tagging"""
    parser = argparse.ArgumentParser(description="Training interface of Pandora")
    parser.add_argument("model", help="Path to model")
    parser.add_argument(
        "--string",
        action="store_true", default=False,
        help="Tag a string instead of a directory [Shell Mode]"
    )
    parser.add_argument("--input", dest="input_dir", help="Path of the input folder")
    parser.add_argument("--output", dest="output_dir", help="Path of the output folder")
    parser.add_argument(
        "--disable-post-correction",
        dest="postcorrect",
        help="Disable post correction",
        action="store_false",
        default=True
    )
    parser.add_argument(
        "--tokenized_input",
        dest="tokenized_input",
        action="store_true",
        default=False,
        help="specify if the input is already tokenized (default: False)"
    )

    args = parser.parse_args()
    if args.string:
        tag_string(**vars(args))
    else:
        tag_dir(**vars(args))


if __name__ == '__main__':
    cli_train()
