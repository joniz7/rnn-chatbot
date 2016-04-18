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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
#from tensorflow.models.rnn import seq2seq
import seq2seq

#from tensorflow.models.rnn.translate import data_utils
import data_utils

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, source_vocab_size, target_vocab_size, input_lengths, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               use_lstm=False, num_samples=512, forward_only=False, embedding_dimensions=50, 
               initial_accumulator_value=0.1, dropout_keep_prob=1.0,
               punct_marks=[], mark_drop_rates=[], sample_output=False):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      input_lengths: a pair (I, O), where I specifies maximum input length
        that will be processed, and O specifies maximum output.
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.batch_size = batch_size
    #self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    #self.learning_rate_decay_op = self.learning_rate.assign(
    #    self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    self.embedding_dimensions=embedding_dimensions
    self.best_validation_error = tf.Variable(float('inf'), trainable=False)
    self.smoothed_train_error = tf.Variable(0.0, trainable=False)
    self.smoothed_eval_error = tf.Variable(0.0, trainable=False)
    #self.patience = tf.Variable(patience, trainable=False)
    #self.decrement_patience_op = self.patience.assign(self.patience - 1)
    self.punct_marks = punct_marks
    self.mark_drop_rates = mark_drop_rates

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      with tf.device("/cpu:0"):
        w = tf.get_variable("proj_w", [size, self.target_vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                            self.target_vocab_size)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    if use_lstm:
      single_cell = rnn_cell.BasicLSTMCell(size)
    else:
      single_cell = rnn_cell.GRUCell(size)

    dropout_single_cell = rnn_cell.DropoutWrapper(single_cell, 
                              output_keep_prob=dropout_keep_prob)
    cell = single_cell
    if not forward_only:
      cell = dropout_single_cell
    if num_layers > 1:
      cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
      if not forward_only:
        cell = rnn_cell.MultiRNNCell([dropout_single_cell] * num_layers)
    ### TODO check that dropout is done correctly. (between right layers)
    self.zero_state_f = cell.zero_state(batch_size, tf.float32)

    self.random_numbers = None
    if sample_output:
      self.random_numbers = tf.placeholder(tf.float32, shape=[None, self.source_vocab_size], name="random_numbers")
    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode, initial_state, sequence_lengths):
      return seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell, source_vocab_size,
          target_vocab_size, output_projection=output_projection,
          feed_previous=do_decode, embedding_dimension=embedding_dimensions, 
          sample_output=sample_output, random_numbers=self.random_numbers)

    # Feeds for inputs.
    self.initial_state_ph = tf.placeholder(tf.float32, shape=[batch_size, cell.state_size], name="initial_state")
    self.sequence_lengths_ph = tf.placeholder(tf.int32, shape=[batch_size], name="sequence_lengths")
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(input_lengths[0]):
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(input_lengths[1] + 1): #includes GO-symbol.
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    if forward_only:
      self.outputs, self.state, self.all_states = seq2seq_f(self.encoder_inputs[:input_lengths[0]], self.decoder_inputs[:input_lengths[1]], True, 
                                        self.initial_state_ph, self.sequence_lengths_ph)
      
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        self.outputs = [
            tf.matmul(output, output_projection[0]) + output_projection[1]
            for output in self.outputs
        ]
    else:
      self.outputs, self.state, self.all_states = seq2seq_f(self.encoder_inputs[:input_lengths[0]], self.decoder_inputs[:input_lengths[1]], False,
                                        self.initial_state_ph, self.sequence_lengths_ph)

    self.losses = seq2seq.sequence_loss(self.outputs, targets[:input_lengths[1]], self.target_weights[:input_lengths[1]],
              softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=initial_accumulator_value) # Changed from originally gradient descent.
      # opt = tf.train.GradientDescentOptimizer(self.learning_rate) ###############################################################################
      gradients = tf.gradients(self.losses, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                       max_gradient_norm)
      self.gradient_norms = norm
      self.updates = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           forward_only, encoder_size, decoder_size, encoder_input_lengths, initial_state=None, random_numbers=None):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A quadruple consisting of gradient norm (or None if we did not do backward),
      average perplexity, the outputs and the final state.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    #encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the given length,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the given length,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the given length,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Applies word drop on words in punct_marks
    unk_token = data_utils.UNK_ID
    for inputs in encoder_inputs:
      for i, num in enumerate(inputs):
        for j, mark in enumerate(self.punct_marks):
          if num == mark:
            rand = random.random()
            if rand < self.mark_drop_rates[j]:
                inputs[i] = unk_token

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    if initial_state is None:
      input_feed[self.initial_state_ph.name] = session.run(self.zero_state_f)
    else:
      input_feed[self.initial_state_ph.name] = initial_state
    input_feed[self.sequence_lengths_ph.name] = encoder_input_lengths

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates,  # Update Op that does SGD.
                     self.gradient_norms,  # Gradient norm.
                     self.losses,
                     self.state]#,
                     #self.all_states]  # Loss for this batch.
    else:
      output_feed = [self.losses, self.state]#, self.all_states]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[l])
      if random_numbers is not None:
        input_feed[self.random_numbers.name] = random_numbers

    for s in self.all_states:
      output_feed.append(s)
    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None, outputs[3], outputs[4:]#outputs[4]  # Gradient norm, loss, no outputs, final state, all states.
    else:
      return None, outputs[0], outputs[2:(decoder_size+2)], outputs[1], outputs[(decoder_size+2):]#outputs[2]  # No gradient norm, loss, outputs, final state, all states.

  def get_conversation_batch(self, data, prev_conv=None):
    """Get a batch and a list keeping track of the batched conversations. 
       Provide prev_conv with the last return value of 'conversations' 
       to fill in conversations that have ended, if this function has 
       been called previously.

    Args:
      data: contains lists of pairs of input and output data that we use 
        to create a batch.
      prev_conv: A list of size batch_size containing lists of 
        [input, output] pairs. None if function hasn't been called before.

    Returns:
      The triple (conversations, same_conv).
        conversations: The same type as prev_conv. Contains a list of 
          conversations, where each conversation has the remaining lines 
          after taking one batch.
        same_conv: A list of size batch_size containing integers
          where same_conv[i] == 0 if conversations[i] contains a new
          conversation, 1 otherwise.
        next_inputs: A list os length batch_size containing pairs of the 
          next inputs.
    """
    if prev_conv is None:
      prev_conv = []
      for i in xrange(self.batch_size):
        prev_conv.append(random.choice(data))
    same_conv = []
    next_inputs = []
    for i in xrange(self.batch_size):
      # Add a new random conversation if current one is empty.
      if len(prev_conv[i]) == 0:
        same_conv.append(0)
        prev_conv[i] = random.choice(data)
      else:
        same_conv.append(1)
      # Create a batch entry and remove first line from conversation.
      next_inputs.append(prev_conv[i][0])
      prev_conv[i] = prev_conv[i][1:]

    return (prev_conv, same_conv, next_inputs)


  def get_batch(self, data, encoder_size, decoder_size, batched_data=False):
    """Get a random batch of data, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: contains lists of pairs of input and output data that we use to 
        create a batch.
      batched_data: if True, data is of size batch_size, and all elements 
        should be used from it in order.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    #encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for bi in xrange(self.batch_size):
      if not batched_data:
        encoder_input, decoder_input = random.choice(data)
      else:
        encoder_input, decoder_input = data[bi]

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      #encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
      encoder_inputs.append(encoder_input + encoder_pad)
      print("    encoder inputs:")
      print(encoder_inputs)

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
      print("    decoder inputs:")
      print(decoder_inputs)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
