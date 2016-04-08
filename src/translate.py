# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/pdf/1412.2007v2.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


#from tensorflow.models.rnn.translate import data_utils
#from tensorflow.models.rnn.translate import seq2seq_model
from tensorflow.python.platform import gfile
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import ops

import data_utils
import seq2seq_model
import parser


execfile("parser.py")

tf.app.flags.DEFINE_float("initial_accumulator_value", 0.1, 
                          "Starting value for the accumulators in Adagrad, must be positive")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("embedding_dimensions", 50, "Dimension of the embedding vectors")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 30000, "Size of our vocabulary")
tf.app.flags.DEFINE_integer("num_samples", 512, "Number of samples")
tf.app.flags.DEFINE_string("data_dir", "../data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "../data", "Training directory.")
tf.app.flags.DEFINE_string("train_data_part", None, "What part of the training data to be used.")
tf.app.flags.DEFINE_string("checkpoint_dir", "../checkpoints", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_valid_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("summary_path", "../data/summaries",
                            "Directory for summaries")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("embedding_path", "../data/embeddings%d.txt"%tf.app.flags.FLAGS.vocab_size, "The path for the file with initial embeddings")
tf.app.flags.DEFINE_float("max_running_time", 60, "The training will terminate after at most this many minutes.")
tf.app.flags.DEFINE_float("quest_drop_rate", 0.25, "The rate at which question marks will be dropped. Number between 0 and 1.")
tf.app.flags.DEFINE_float("excl_drop_rate", 0.25, "The rate at which exclamation markswill be dropped. Number between 0 and 1.")
tf.app.flags.DEFINE_float("period_drop_rate", 0.25, "The rate at which periods will be dropped. Number between 0 and 1.")
tf.app.flags.DEFINE_float("comma_drop_date", 0.25, "The rate at which commas will be dropped. Number between 0 and 1.")
tf.app.flags.DEFINE_float("dots_drop_rate", 0.25, "The rate at which the _DOTS tag will be dropped. Number between 0 and 1.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "The probability that dropout is NOT applied to a node.")
tf.app.flags.DEFINE_float("decode_randomness", 0.1, "Factor determining the randomness when producing the output. Should be a float in [0, 1]")
tf.app.flags.DEFINE_boolean("prettify_decoding", True, "If set, corrects spelling, randomizes numbers, generates a new output if output starts with _EOS and adds _UNK to to end of input")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_input_lengths = (25, 25)


def inject_embeddings(source_path):
  """Read embeddings from source and replace the current embeddings with this.

  Args:
    source_path : path to the file with the embeddings"""

  print("Load embedding from %s"%source_path)
  with tf.variable_scope("embedding_attention_seq2seq/embedding", reuse=True):
    with tf.Session() as sess:
      with ops.device("/cpu:0"):
        embedding = tf.get_variable("embedding")
        sess.run(embedding.assign(parseEmbeddings(source_path)))


def read_data(source_path, max_size=None):
  """Read data from source and target files and put into buckets.
      ASSUMES THERE ARE NO EMPTY CONVERSATIONS.

  Args:
    source_path: path to the files with token-ids for the source language.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = []
  with gfile.GFile(source_path, mode="r") as source_file:
    conversation_counter = 0
    line_discard_counter = 0
    conversation_discard_counter = 0
    lines_read = 0
    response = True
    def read_line():
      if lines_read % 10 == 0:
        print("  reading data line %d. Found conversations: %d, discarded conversations: %d, discarded lines: %d." 
              % (lines_read, conversation_counter, conversation_discard_counter, line_discard_counter))
        sys.stdout.flush()
      return source_file.readline()
    #END read_line()
    while response and (not max_size or lines_read < max_size):
      conversation = []
      contained_too_long_line = False

      #Check that we have no empty conversations.
      first_utt_id = data_utils.IGNORE_ID
      checked_once = False
      while first_utt_id == data_utils.IGNORE_ID:
        if checked_once:
          print("WARNING, found empty conversation. Check your parser, dude!")
        utterance = read_line()
        lines_read += 1
        utte_ids = [int(x) for x in utterance.split()]
        first_utt_id = utte_ids[0]
        checked_once = True

      response = read_line()
      lines_read += 1
      resp_ids = [int(x) for x in response.split()]

      while response and resp_ids[0] != data_utils.IGNORE_ID and (not max_size or lines_read < max_size):
        resp_ids.append(data_utils.EOS_ID)
        if len(utte_ids) >= _input_lengths[0] or len(resp_ids) >= _input_lengths[1]:
          contained_too_long_line = True

        conversation.append([utte_ids, resp_ids])

        utterance = response
        response = read_line()
        lines_read += 1
        utte_ids = [int(x) for x in utterance.split()]
        resp_ids = [int(x) for x in response.split()]
      #END WHILE
      if conversation != []: #Check that we even entered the while loop.
        if not contained_too_long_line:
          conversation_counter += 1
          data_set.append(conversation)
        else:
          conversation_discard_counter += 1
          line_discard_counter += len(conversation)
    #END WHILE
    print("Done reading data. Found conversations: %d, discarded conversations: %d, discarded lines: %d." 
      % (conversation_counter, conversation_discard_counter, line_discard_counter))
    return data_set
#END DEF


def create_model(session, forward_only, vocab, sample_output=False):
  """Create translation model and initialize or load parameters in session."""

  with tf.variable_scope("embedding_attention_seq2seq/embedding"):
    with ops.device("/cpu:0"):
      tf.get_variable("embedding", [FLAGS.vocab_size, FLAGS.embedding_dimensions])

  # what characters to randomly drop
  punct_marks = data_utils.sentence_to_token_ids(".?!,_DOTS", vocab)
  # list of drop rates with the same ordering as above
  mark_drop_rates = [FLAGS.period_drop_rate, FLAGS.quest_drop_rate, FLAGS.excl_drop_rate, FLAGS.comma_drop_date, FLAGS.dots_drop_rate]

  model = seq2seq_model.Seq2SeqModel(
      FLAGS.vocab_size, FLAGS.vocab_size, _input_lengths,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, use_lstm=True, num_samples=FLAGS.num_samples, 
      forward_only=forward_only, embedding_dimensions=FLAGS.embedding_dimensions,
      initial_accumulator_value=FLAGS.initial_accumulator_value, 
      dropout_keep_prob=FLAGS.dropout_keep_prob,
      punct_marks=punct_marks, mark_drop_rates=mark_drop_rates,
      sample_output=sample_output)

  #ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  #if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
  #  print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
  #  model.saver.restore(session, ckpt.model_checkpoint_path)
  #else:
  #  print("Created model with fresh parameters.")
  #  session.run(tf.initialize_all_variables())
  return model

def init_model(session, model):
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    print("Loaded model with validation error %.2f, global step %d" % 
          (model.best_validation_error.eval(), model.global_step.eval()))
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())

    with tf.variable_scope("embedding_attention_seq2seq/embedding", reuse=True): #Inject the embeddings
      with ops.device("/cpu:0"):
        embedding = tf.get_variable("embedding")
        session.run(embedding.assign(parseEmbeddings(FLAGS.embedding_path)))
  return model
  
def train():
  training_start_time = time.time()

  """Train a dialogue model using dialogue corpus."""
  # Prepare dialogue data.
  print("Preparing dialogue data in %s" % FLAGS.data_dir)

  train_path, dev_path, _ = data_utils.prepare_dialogue_data(
      FLAGS.data_dir, FLAGS.vocab_size, FLAGS.train_data_part)

  with tf.Session() as sess:
    # Load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d" % FLAGS.vocab_size)
    vocab, _ = data_utils.initialize_vocabulary(vocab_path)

    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False, vocab)

    # Read data
    print ("Reading validation data (limit: %d)."
           % FLAGS.max_valid_data_size)
    dev_set = read_data(dev_path, FLAGS.max_valid_data_size)
    print ("Reading training data (limit: %d)."
           % FLAGS.max_train_data_size)
    train_set = read_data(train_path, FLAGS.max_train_data_size)

    dev_set = dev_set[0]
    train_set = train_set[0]

    def eval_dev_set():
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          dev_set, _input_lengths[0], _input_lengths[1])
      _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, True, _input_lengths[0], _input_lengths[1])
      return eval_loss

    def perplexity(loss):
      ppx = math.exp(loss) if loss < 300 else float('inf')
      return ppx

    # Setting up summaries
    #buck_losses = tf.placeholder(tf.float32, shape=[len(_buckets)], name="buck_losses")#eval_dev_set()
    evaluation_losses = tf.placeholder(tf.float32, name="evaluation_losses")
    train_losses = tf.placeholder(tf.float32, name="train_losses")
    eval_ppx = tf.placeholder(tf.float32, name="eval_ppx")
    train_ppx = tf.placeholder(tf.float32, name="train_ppx")

    #eval_loss_summary = tf.histogram_summary("eval_bucket_losses", buck_losses)
    evaluation_losses_summary = tf.scalar_summary("evaluation_losses_summary",
          evaluation_losses)
    #learning_rate_summary = tf.scalar_summary("learning_rate", model.learning_rate)
    train_losses_summary = tf.scalar_summary("train_losses_avg", train_losses)
    eval_ppx_summary = tf.scalar_summary("eval_ppx_avg", eval_ppx)
    train_ppx_summary = tf.scalar_summary("train_ppx_avg", train_ppx)
    smoothed_train_losses_summary = tf.scalar_summary("smoothed_train_losses_summary", model.smoothed_train_error)
    smoothed_evaluation_losses_summary = tf.scalar_summary("smoothed_evaluation_losses_summary", model.smoothed_eval_error)
    best_validation_error_summary = tf.scalar_summary("best_validation_error", model.best_validation_error)
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.summary_path, sess.graph_def)

    print("Initializing model...")

    model = init_model(sess, model)

    print("Model initialized!")


    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    #train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
    #                       for i in xrange(len(train_bucket_sizes))]

    # The sizes of the buckets normalized to 1, such that: 
    # sum(eval_buckets_dist) == 1.0
    #eval_buckets_dist = [eval_bucket_sizes[i] / eval_total_size 
    #                      for i in xrange(len(eval_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0

    print("COMMENCE TRAINING!!!!!!")

    # create checkpoint_path
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "translate.ckpt")
    try:
      while ((time.time() - training_start_time)/60 < FLAGS.max_running_time ):
        """with tf.variable_scope("embedding_attention_seq2seq/embedding"):
          embedding = sess.run(tf.get_variable("embedding"))
          temp = embedding_ops.embedding_lookup(embedding, [6])
          print(embedding[6])
          print(temp)"""
        # Necessary when session is aborted out of sync with FLAGS.steps_per_checkpoint.
        current_step += 1
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        #random_number_01 = np.random.random_sample()
        #bucket_id = min([i for i in xrange(len(train_buckets_scale))
        #                 if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            train_set, _input_lengths[0], _input_lengths[1])
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, False, _input_lengths[0], _input_lengths[1])
        step_time += (time.time() - start_time)
        loss += step_loss #/ FLAGS.steps_per_checkpoint
        global_step = model.global_step.eval() ############################## START WORKING HERE TOMORROW!!!

        # Writes summaries and also a checkpoint if new best is found.
        if(global_step%FLAGS.steps_per_checkpoint == 0):
          loss = loss / current_step
          step_time = step_time / current_step

          current_evaluation_loss = eval_dev_set()
          
          lowest_valid_error = model.best_validation_error.eval()
          # Save model and error if new best is found
          if(current_evaluation_loss < lowest_valid_error):
            sess.run(model.best_validation_error.assign(current_evaluation_loss))
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          
          # Calculate new smoothed data points. Biased towards the latest data by smooth_ratio.
          def smooth_data(old_data, new_data):
            smooth_ratio = 0.9
            if global_step == FLAGS.steps_per_checkpoint:
              return new_data
            return old_data*smooth_ratio + new_data*(1 - smooth_ratio)

          sess.run(model.smoothed_train_error.assign(smooth_data(model.smoothed_train_error.eval(), loss)))
          sess.run(model.smoothed_eval_error.assign(smooth_data(model.smoothed_eval_error.eval(), current_evaluation_loss)))

          # Calculate summaries.
          current_eval_ppx = perplexity(current_evaluation_loss)
          current_train_ppx = perplexity(loss)
          feed = {eval_ppx: current_eval_ppx,
                  train_ppx: current_train_ppx,
                  evaluation_losses: current_evaluation_loss,
                  train_losses: loss}
          summary_str = sess.run(merged, feed_dict=feed)
          # Write all summaries.
          writer.add_summary(summary_str, global_step)

          # Print statistics for the previous epoch.
          print ("global step %d step-time %.2f training perplexity "
                 "%.2f evaluation perplexity %.2f" % (global_step,
                           step_time, current_train_ppx, current_eval_ppx))

          step_time, loss = 0.0, 0.0
          current_step = 0

          sys.stdout.flush()
      # END WHILE
      print("Training stopped after running out of time, at step %d" % model.global_step.eval())
      print("Saving model...")
      model.saver.save(sess, checkpoint_path, global_step=model.global_step)
      print("Model saved!")
    except KeyboardInterrupt:
      print("Training stopped at step %d"%model.global_step.eval())
      print("Saving model...")
      model.saver.save(sess, checkpoint_path, global_step=model.global_step)
      print("Model saved!")
      writer.flush()
      writer.close()


def decode():
  with tf.Session() as sess:
    # Load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    #_buckets = [(150, 150)]

    # Create model and load parameters.
    model = create_model(sess, True, vocab, sample_output=True)
    model = init_model(sess, model)
    
    model.batch_size = 1  # We decode one sentence at a time.

    def get_random_numbers(rand_level):
      out = []
      for j in xrange(len(decoder_inputs)):
        out.append([1 - np.random.uniform()*rand_level for i in xrange(model.source_vocab_size)])
      return out

    def replace_zeros_with_rands(input_str):
      new_str = ""
      for cha in input_str:
        if cha == '0':
          new_str += str(int(np.floor(np.random.uniform()*10)))
        else:
          new_str += cha
      return new_str

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Split at apostrophes and make everything lowercase.
      sentence = parser.splitApostrophe(sentence).lower()

      # Fix _DOTS here. Won't lower() on _DOTS.
      sentence = parser.removeStars(sentence)

      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(sentence, vocab, correct_spelling=FLAGS.prettify_decoding)

      # Add _UNK at end of sentence if flagged
      if FLAGS.prettify_decoding:
        if not token_ids[-1] in ([data_utils.UNK_ID, vocab["_DOTS"]] + [vocab[pm] for pm in parser.punctuationMarks]):
          token_ids.append(data_utils.UNK_ID)

      if len(token_ids) >= _buckets[-1][0]:
        print("tldr pls")
      else:
        # Which bucket does it belong to?
        bucket_id = min([b for b in xrange(len(_buckets))
                         if _buckets[b][0] > len(token_ids)])

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            [(token_ids, [])], _input_lengths[0], _input_lengths[1])

        #### Start while no response ####
        generate_new = True
        randomness = FLAGS.decode_randomness
        while(generate_new):
          generate_new = False
          random_numbers = get_random_numbers(randomness)
          # Get output logits for the sentence.
          _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True, random_numbers=random_numbers)
          # This is a greedy decoder - outputs are just argmaxes of output_logits.
          #outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

          # Throw away first random number and last logit in output_logits
          output_logits = output_logits[:-1]
          random_numbers = random_numbers[1:]
          outputs = [sample_output(output_logits[l], random_numbers[l]) for l in xrange(len(output_logits))]

          # Generate new response if prettifying and no response is given.
          # Do not generate new respone if response generation is deterministic.
          if(FLAGS.prettify_decoding and (not FLAGS.decode_randomness == 0.0) and outputs[0] == data_utils.EOS_ID):
            generate_new = True
            randomness *= 1.1 # Increase randomness a little bit so we won't get stuck in an infinite loop.
        #### End while no response ####

        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        # Print out French sentence corresponding to outputs.
        if FLAGS.prettify_decoding:
          # Join in a neat fashion if flag is set.
          s = ""
          tightJoinTokens = parser.getTightJoinTokens(vocab)
          for output in outputs:
            if output in tightJoinTokens:
              s = "".join([s, rev_vocab[output]])
            else:
              s = " ".join([s, rev_vocab[output]])
          s = replace_zeros_with_rands(s.replace("_DOTS", "..."))
          print(s)
        else:
          print(" ".join([rev_vocab[output] for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def sample_output(logit, rand, replace_UNK=True):
  logit = logit[0][:]
  smallest = np.argmin(logit)
  normed = logit - logit[smallest]
  noised = np.multiply(normed, rand)
  sorted_idx = np.argsort(noised)
  if replace_UNK and sorted_idx[-1] == data_utils.UNK_ID:
    return sorted_idx[-2]
  #print ("first logit: %.4f, first normed: %.4f, first noised: %.4f" % (logit[0], normed[0], noised[0]))
  return np.argmax(noised)


def self_test():
  return
#   """Test the translation model."""
#   with tf.Session() as sess:
#     print("Self-test for neural translation model.")
#     # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
#     model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
#                                        5.0, 32, 0.3, 0.99, num_samples=8)
#     sess.run(tf.initialize_all_variables())

#     # Fake data set for both the (3, 3) and (6, 6) bucket.
#     data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
#                 [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
#     for _ in xrange(5):  # Train the fake model for 5 steps.
#       bucket_id = random.choice([0, 1])
#       encoder_inputs, decoder_inputs, target_weights = model.get_batch(
#           data_set, bucket_id)
#       model.step(sess, encoder_inputs, decoder_inputs, target_weights,
#                  bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
