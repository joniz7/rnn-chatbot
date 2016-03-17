
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
#import tarfile

from tensorflow.python.platform import gfile
from six.moves import urllib

#from autocorrect import spell

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_DOTS = "_DOTS"
_IGNORE = "_IGNORE"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _DOTS, _IGNORE]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
DOTS_ID = 4
IGNORE_ID = 5

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\":;)(])")
_DIGIT_RE = re.compile(r"\d")


# TODO add def to check if training data at given directory exists

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
            tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is 
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later 
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="r") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print(" processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                    if word.strip() != _IGNORE:
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")

def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, 
                tokenizer=None, normalize_digits=True, correct_spelling=False):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into 
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2, 
    "a": 4, "dog": 7} this function will return [1, 2, 4, 7].

    Args:
      sentence: a string, the sentence to convert to token-ids.
      vocabulary: a dictonary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
#    def spell_and_replace(word):
#        if vocabulary.has_key(word):
#            return word
#        return spell(word)

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)

    if normalize_digits:
        words = [re.sub(_DIGIT_RE, "0", w) for w in words]

#    if correct_spelling:
#        words = [spell_and_replace(w) for w in words]

    return [vocabulary.get(w, UNK_ID) for w in words]

    #if not normalize_digits:
    #    return [vocabulary.get(w, UNK_ID) for w in words]
    ## Normalize digits by 0 before looking words up in the vocabulary
    #return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path, 
            tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the detailw of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print(" tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer, 
                                normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_dialogue_data(data_dir, vocabulary_size, part=None):
    """Load dialogue data from data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory where the data sets are stored. Assumes data is 
        called; 'train-data.utte', 'train-data.resp', 'valid-data.utte' and
        'valid-data.resp' for input and output for the training and validation
        data sets.
      utte_vocabulary_size: size of the utterance vocabulary to create and use.
      resp_vocabulary_size: size of the response vocabulary to create and use.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for utterance training data-set,
        (2) path to the token-ids for response training data-set,
        (3) path to the token-ids for utterance validation/development data-set,
        (4) path to the token-ids for response validation/development data-set,
        (5) path to the utterance vocabulary file,
        (6) path to the response vocabulary file.
    """
    train_path, valid_path = check_dialogue_sets(data_dir)
    #  Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    create_vocabulary(vocab_path, train_path+".data", vocabulary_size)

    # Create token ids for the training data.
    train_ids_path = train_path + (".ids%d.data" % vocabulary_size)
    data_to_token_ids(train_path + ".data", train_ids_path, vocab_path)

    if part:
        train_ids_path = train_ids_path + "-dir/part%d"%part

    # Create token ids for the development data.
    valid_ids_path = valid_path + (".ids%d.data" % vocabulary_size)
    data_to_token_ids(valid_path + ".data", valid_ids_path, vocab_path)

    return (train_ids_path, valid_ids_path, vocab_path)


def check_dialogue_sets(directory):
    """Check that the training sets are in the specified directory.

    Args:
      directory: the directory where the dialogue sets should be.

    Returns:
      train_path:
      valid_path:

    Raises:
      ValueError: if files are missing in the provided directory.
    """
    train_name = "train-data"
    valid_name = "valid-data"
    suffix = ".data"
    train_path = os.path.join(directory, train_name)
    valid_path = os.path.join(directory, valid_name)

    if not (gfile.Exists(train_path + suffix)):
        raise ValueError("Training file %s not found.", train_path + suffix)
    if not (gfile.Exists(valid_path + suffix)):
        raise ValueError("Validation file %s not found.", valid_path + suffix)

    return (train_path, valid_path)