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
tf.app.flags.DEFINE_string("data_dir", "../data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "../data", "Training directory.")
tf.app.flags.DEFINE_string("checkpoint_dir", "../checkpoints", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("initial_steps", 0,#10000, 
                            "Guaranteed number of steps to train")
tf.app.flags.DEFINE_string("summary_path", "../data/summaries",
                            "Directory for summaries")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("embedding_path", "../data/embeddings%d.txt"%tf.app.flags.FLAGS.vocab_size, "The path for the file with initial embeddings")
tf.app.flags.DEFINE_float("patience_sensitivity", 0.995, 
                          "determines when an improvement/worsening is significant")
tf.app.flags.DEFINE_integer("max_patience", 120, 
                            "The number of checks where model performs worse before stopping")
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
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


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
  data_set = [[] for _ in _buckets]
  with gfile.GFile(source_path, mode="r") as source_file:
    utterance, response = source_file.readline(), source_file.readline()
    counter = 0
    while utterance and response and (not max_size or counter < max_size):
      counter += 1
      if counter % 100000 == 0:
        print("  reading data line %d" % counter)
        sys.stdout.flush()
      utte_ids = [int(x) for x in utterance.split()]
      resp_ids = [int(x) for x in response.split()]
      resp_ids.append(data_utils.EOS_ID)
      for bucket_id, (utte_size, resp_size) in enumerate(_buckets):
        if len(utte_ids) < utte_size and len(resp_ids) < resp_size:
          data_set[bucket_id].append([utte_ids, resp_ids])
          break
      utterance = response
      response = source_file.readline()
      if response and int(response.split()[0]) == data_utils.IGNORE_ID:
        utterance, response = source_file.readline(), source_file.readline()
  return data_set

"""data_set = [[] for _ in _buckets]
  with gfile.GFile(source_path, mode="r") as source_file:
    with gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set"""


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
      FLAGS.vocab_size, FLAGS.vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only, embedding_dimensions=FLAGS.embedding_dimensions,
      initial_accumulator_value=FLAGS.initial_accumulator_value,
      punct_marks=punct_marks, mark_drop_rates=mark_drop_rates,
      patience=FLAGS.max_patience, dropout_keep_prob=FLAGS.dropout_keep_prob, sample_output=sample_output)

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
    print("Loaded model with validation error %.2f, global step %d and patience %d" % 
          (model.best_validation_error.eval(), model.global_step.eval(), model.patience.eval()))
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
      FLAGS.data_dir, FLAGS.vocab_size)

  with tf.Session() as sess:
    # Load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d" % FLAGS.vocab_size)
    vocab, _ = data_utils.initialize_vocabulary(vocab_path)

    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False, vocab)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(dev_path)
    train_set = read_data(train_path, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    eval_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
    eval_total_size = float(sum(eval_bucket_sizes))

    def eval_dev_set():
      bucket_losses = []
      for bucket_id in xrange(len(_buckets)):
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          bucket_losses.append(eval_loss)
          #eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          #print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
      return bucket_losses

    def perplexity(loss):
      ppx = math.exp(loss) if loss < 300 else float('inf')
      return ppx

    # Setting up summaries
    buck_losses = tf.placeholder(tf.float32, shape=[len(_buckets)], name="buck_losses")#eval_dev_set()
    average_bucket_loss = tf.placeholder(tf.float32, name="average_bucket_loss")
    train_losses = tf.placeholder(tf.float32, name="train_losses")
    eval_ppx = tf.placeholder(tf.float32, name="eval_ppx")
    train_ppx = tf.placeholder(tf.float32, name="train_ppx")

    eval_loss_summary = tf.histogram_summary("eval_bucket_losses", buck_losses)
    eval_avg_loss_summary = tf.scalar_summary("eval_bucket_average_losses",
          average_bucket_loss)
    learning_rate_summary = tf.scalar_summary("learning_rate", model.learning_rate)
    train_avg_loss_summary = tf.scalar_summary("train_losses_avg", train_losses)
    eval_ppx_summary = tf.scalar_summary("eval_ppx_avg", eval_ppx)
    train_ppx_summary = tf.scalar_summary("train_ppx_avg", train_ppx)
    mean_train_err_summary = tf.scalar_summary("mean_train_err", model.mean_train_error)
    mean_eval_err_summary = tf.scalar_summary("mean_eval_err", model.mean_eval_error)
    best_validation_error_summary = tf.scalar_summary("best_validation_error", model.best_validation_error)
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.summary_path, sess.graph_def)

    print("Initializing model...")

    model = init_model(sess, model)

    print("Model initialized!")


    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # The sizes of the buckets normalized to 1, such that: 
    # sum(eval_buckets_dist) == 1.0
    eval_buckets_dist = [eval_bucket_sizes[i] / eval_total_size 
                          for i in xrange(len(eval_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    print("COMMENCE TRAINING!!!!!!")

    # create checkpoint_path
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "translate.ckpt")
    try:
      while ((model.patience.eval() > 0 or model.global_step.eval() < FLAGS.initial_steps) 
            and (time.time() - training_start_time)/60 < FLAGS.max_running_time ):
        """with tf.variable_scope("embedding_attention_seq2seq/embedding"):
          embedding = sess.run(tf.get_variable("embedding"))
          temp = embedding_ops.embedding_lookup(embedding, [6])
          print(embedding[6])
          print(temp)"""
        # Necessary when session is aborted out of sync with FLAGS.steps_per_checkpoint.
        current_step += 1
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            train_set, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss #/ FLAGS.steps_per_checkpoint
        global_step = model.global_step.eval()

        # Writes summaries and also a checkpoint if new best is found.
        if(global_step%FLAGS.steps_per_checkpoint == 0):
          loss = loss / current_step
          eval_losses = np.asarray(eval_dev_set())
          current_avg_buck_loss = 0.0
          for b in xrange(len(_buckets)):
            current_avg_buck_loss += eval_buckets_dist[b] * eval_losses[b]
          
          sess.run(model.decrement_patience_op)
          # Reset patience if no significant increase in error.
          lowest_valid_error = model.best_validation_error.eval()
          if(current_avg_buck_loss*FLAGS.patience_sensitivity < lowest_valid_error):
            sess.run(model.patience.assign(FLAGS.max_patience))
          # Save model and error if new best is found
          if(current_avg_buck_loss < lowest_valid_error):
            sess.run(model.best_validation_error.assign(current_avg_buck_loss))
            # Don't save model during initial_steps
            if(global_step > FLAGS.initial_steps):
              model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          
          # Calculate new means. Biased towards the latest data by 1/5.
          #current_check_step = global_step/FLAGS.steps_per_checkpoint
          old_train_mean = model.mean_train_error.eval()
          old_eval_mean = model.mean_eval_error.eval()
          old_modifier = 4/5 #current_check_step
          new_modifier = 1/5
          if global_step == FLAGS.steps_per_checkpoint:
            new_train_mean = loss
            new_eval_mean = current_avg_buck_loss
          else:
            new_train_mean = old_train_mean*old_modifier + loss*new_modifier #current_check_step
            new_eval_mean = old_eval_mean*old_modifier + current_avg_buck_loss*new_modifier #current_check_step
          #print("global step: %d, new train mean: %.4f, new eval mean: %.4f" % (global_step, new_train_mean, new_eval_mean))
          sess.run(model.mean_train_error.assign(new_train_mean))
          sess.run(model.mean_eval_error.assign(new_eval_mean))
          #print ("current step: %d, old train mean: %.4f, old eval mean: %.4f, old modifier: %.4f, new train mean: %.4f, new eval mean: %.4f" %
          #  (current_check_step, old_train_mean, old_eval_mean, old_modifier, new_train_mean, new_eval_mean))

          # Calculate summaries.
          current_eval_ppx = perplexity(current_avg_buck_loss)
          current_train_ppx = perplexity(loss)
          feed = {buck_losses: eval_losses, 
                  eval_ppx: current_eval_ppx,
                  train_ppx: current_train_ppx,
                  average_bucket_loss: current_avg_buck_loss,
                  train_losses: loss}
          summary_str = sess.run(merged, feed_dict=feed)
          # Write all summaries.
          writer.add_summary(summary_str, global_step)

          # Print statistics for the previous epoch.
          print ("global step %d learning rate %.4f step-time %.2f training perplexity "
                 "%.2f evaluation perplexity %.2f patience %d" % (global_step, model.learning_rate.eval(),
                           step_time, current_train_ppx, current_eval_ppx, model.patience.eval()))
          # Decrease learning rate if no improvement was seen over last 3 times.
          losses_lookback = 12
          if len(previous_losses) >= losses_lookback and loss > max(previous_losses[-losses_lookback:]):
            sess.run(model.learning_rate_decay_op)
          previous_losses.append(loss)

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

    _buckets = [(150, 150)]

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
            {bucket_id: [(token_ids, [])]}, bucket_id)

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
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
