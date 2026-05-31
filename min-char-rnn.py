"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Per-timestep tensors are column vectors at time t:
  inputs_one_hot[t]    shape (vocab_size, 1)   -- one-hot encoding of the input char
  hidden_states[t]     shape (hidden_size, 1)  -- hidden state
  output_logits[t]     shape (vocab_size, 1)   -- unnormalized log-probabilities
  output_probs[t]      shape (vocab_size, 1)   -- softmax probabilities over the next char

Recurrence (forward):
  hidden_states[t] = tanh(
      weights_input_to_hidden  @ inputs_one_hot[t]    +
      weights_hidden_to_hidden @ hidden_states[t - 1] +
      bias_hidden
  )
  output_logits[t] = weights_hidden_to_output @ hidden_states[t] + bias_output
  output_probs[t]  = softmax(output_logits[t])
  loss_at_t        = -log(output_probs[t][target_index])
"""
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=2000,
                    help='training iterations (default: 2000)')
args = parser.parse_args()

# ----- data I/O ---------------------------------------------------------------
text = open('input.txt', 'r').read()                # entire training corpus as one string
unique_chars = list(set(text))                       # character vocabulary (order is arbitrary)
text_length, vocab_size = len(text), len(unique_chars)
print('data has %d characters, %d unique.' % (text_length, vocab_size))
char_to_index = { char: i for i, char in enumerate(unique_chars) }  # str -> int id
index_to_char = { i: char for i, char in enumerate(unique_chars) }  # int id -> str

# ----- hyperparameters --------------------------------------------------------
hidden_size = 10           # number of recurrent units in the hidden layer
sequence_length = 25       # backprop-through-time window: longer = more context but slower / harder to train
learning_rate = 1e-1       # Adagrad base step size

# ----- model parameters -------------------------------------------------------
# Small random init breaks symmetry; biases start at 0.
# Shapes are chosen so the matrix multiplies in the forward pass type-check.
weights_input_to_hidden  = np.random.randn(hidden_size, vocab_size)  * 0.01  # shape (hidden_size, vocab_size)
weights_hidden_to_hidden = np.random.randn(hidden_size, hidden_size) * 0.01  # shape (hidden_size, hidden_size); all-to-all recurrent connectivity
weights_hidden_to_output = np.random.randn(vocab_size, hidden_size)  * 0.01  # shape (vocab_size, hidden_size); read-out into per-char logits
bias_hidden = np.zeros((hidden_size, 1))                                      # shape (hidden_size, 1)
bias_output = np.zeros((vocab_size, 1))                                       # shape (vocab_size, 1)


def compute_loss_and_gradients(input_indices, target_indices, previous_hidden_state):
  """
  Forward + backward pass over ONE backprop-through-time window of length `sequence_length`.

  Args:
    input_indices:         list[int] of length sequence_length -- input char ids at each timestep
    target_indices:        list[int] of length sequence_length -- next-char ids the model should predict
    previous_hidden_state: ndarray (hidden_size, 1) -- hidden state inherited from the previous window
                           (this is what makes the recurrence carry information across windows)

  Returns:
    loss (float):   summed cross-entropy over the predictions in this window
    grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output,
    grad_bias_hidden, grad_bias_output: gradients of `loss` w.r.t. each parameter (same shape as the param)
    last hidden state (hidden_size, 1): hidden_states[sequence_length - 1], used to seed the next window
  """
  # Per-timestep caches kept around so we can reuse the activations during backprop.
  # Indexed by t (the time step within this window). hidden_states[-1] holds the carried-over state.
  inputs_one_hot, hidden_states, output_logits, output_probs = {}, {}, {}, {}
  hidden_states[-1] = np.copy(previous_hidden_state)
  loss = 0

  # ----- forward pass: t = 0, 1, ..., sequence_length - 1 ---------------------
  for t in range(len(input_indices)):
    # One-hot encode the input char so weights_input_to_hidden @ inputs_one_hot[t]
    # effectively selects column input_indices[t] of weights_input_to_hidden.
    inputs_one_hot[t] = np.zeros((vocab_size, 1))
    inputs_one_hot[t][input_indices[t]] = 1

    # Recurrent update. tanh squashes to (-1, 1) so the state can't blow up in the forward direction.
    hidden_states[t] = np.tanh(
        np.dot(weights_input_to_hidden,  inputs_one_hot[t]) +    # contribution of the new input
        np.dot(weights_hidden_to_hidden, hidden_states[t-1]) +   # contribution of the past (memory)
        bias_hidden
    )

    # Read out logits, then softmax to get a probability distribution over the vocab.
    output_logits[t] = np.dot(weights_hidden_to_output, hidden_states[t]) + bias_output
    output_probs[t]  = np.exp(output_logits[t]) / np.sum(np.exp(output_logits[t]))

    # Cross-entropy: penalize the negative log-probability the model assigned to the correct next char.
    # If the model is surprised (prob of target is small), the loss is large; if prob ~ 1, loss ~ 0.
    loss += -np.log(output_probs[t][target_indices[t], 0])

  # ----- backward pass: BPTT, t = sequence_length - 1, ..., 0 -----------------
  # We need d(loss)/d(param) for every learnable parameter. Each parameter is shared
  # across all timesteps, so its gradient is the *sum* of contributions from every t.
  # We accumulate those sums into these buffers, starting from zero.
  grad_weights_input_to_hidden  = np.zeros_like(weights_input_to_hidden)   # shape (hidden_size, vocab_size)
  grad_weights_hidden_to_hidden = np.zeros_like(weights_hidden_to_hidden)  # shape (hidden_size, hidden_size)
  grad_weights_hidden_to_output = np.zeros_like(weights_hidden_to_output)  # shape (vocab_size, hidden_size)
  grad_bias_hidden = np.zeros_like(bias_hidden)                            # shape (hidden_size, 1)
  grad_bias_output = np.zeros_like(bias_output)                            # shape (vocab_size, 1)

  # grad_hidden_next carries d(loss)/d(hidden_states[t+1]) backwards across timesteps.
  # At the last timestep there's no "future" contribution yet, so we start with zeros
  # and accumulate as we unroll backward.
  grad_hidden_next = np.zeros_like(hidden_states[0])

  for t in reversed(range(len(input_indices))):
    # ---- (1) gradient w.r.t. the output logits -----------------------------
    # For softmax + cross-entropy, the gradient w.r.t. the logits simplifies beautifully to:
    #     d(loss)/d(output_logits[t]) = output_probs[t] - one_hot(target_indices[t])
    # See http://cs231n.github.io/neural-networks-case-study/#grad for the derivation.
    grad_output = np.copy(output_probs[t])
    grad_output[target_indices[t]] -= 1              # subtract one-hot(target) from the prob vector

    # ---- (2) push that gradient into weights_hidden_to_output and bias_output ----
    # output_logits[t] = weights_hidden_to_output @ hidden_states[t] + bias_output
    #   => grad_weights_hidden_to_output += grad_output @ hidden_states[t].T   (outer product)
    #      grad_bias_output              += grad_output
    grad_weights_hidden_to_output += np.dot(grad_output, hidden_states[t].T)
    grad_bias_output              += grad_output

    # ---- (3) backprop into the hidden state hidden_states[t] ---------------
    # hidden_states[t] affects the loss two ways:
    #   (a) directly through output_logits[t]
    #         contribution: weights_hidden_to_output.T @ grad_output
    #   (b) indirectly through hidden_states[t+1] (via the recurrence)
    #         contribution carried in grad_hidden_next
    # Sum both -> the *total* gradient flowing back into hidden_states[t].
    grad_hidden = np.dot(weights_hidden_to_output.T, grad_output) + grad_hidden_next

    # ---- (4) backprop through the tanh nonlinearity ------------------------
    # hidden_states[t] = tanh(pre_activation),  tanh'(z) = 1 - tanh(z)^2 = 1 - hidden_states[t]^2.
    # So  d(loss)/d(pre_activation) = (1 - hidden_states[t]^2) * grad_hidden   (elementwise).
    grad_hidden_raw = (1 - hidden_states[t] * hidden_states[t]) * grad_hidden

    # ---- (5) push that into the pre-activation parameters ------------------
    # pre_activation = weights_input_to_hidden  @ inputs_one_hot[t]
    #                + weights_hidden_to_hidden @ hidden_states[t-1]
    #                + bias_hidden
    # so:
    #   grad_bias_hidden               += grad_hidden_raw
    #   grad_weights_input_to_hidden   += grad_hidden_raw @ inputs_one_hot[t].T   (outer product)
    #   grad_weights_hidden_to_hidden  += grad_hidden_raw @ hidden_states[t-1].T  (outer product)
    grad_bias_hidden              += grad_hidden_raw
    grad_weights_input_to_hidden  += np.dot(grad_hidden_raw, inputs_one_hot[t].T)
    grad_weights_hidden_to_hidden += np.dot(grad_hidden_raw, hidden_states[t-1].T)

    # ---- (6) carry the gradient into the previous timestep -----------------
    # hidden_states[t-1] influences hidden_states[t] only through
    #     pre_activation = ... + weights_hidden_to_hidden @ hidden_states[t-1] + ...
    # so this iteration's contribution to d(loss)/d(hidden_states[t-1]) is:
    #     weights_hidden_to_hidden.T @ grad_hidden_raw
    # Stash it so the next loop iteration (which handles t-1) can add it to its grad_hidden.
    grad_hidden_next = np.dot(weights_hidden_to_hidden.T, grad_hidden_raw)

  # Clip gradients elementwise to [-5, 5] to mitigate exploding gradients,
  # a well-known pathology of vanilla RNNs when backpropagating through many timesteps.
  for grad in [grad_weights_input_to_hidden, grad_weights_hidden_to_hidden,
               grad_weights_hidden_to_output, grad_bias_hidden, grad_bias_output]:
    np.clip(grad, -5, 5, out=grad)

  return (loss,
          grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output,
          grad_bias_hidden, grad_bias_output,
          hidden_states[len(input_indices) - 1])


def sample(hidden_state, seed_index, num_chars_to_sample):
  """
  Generate characters from the model, one at a time, by repeatedly:
    feed in current char -> compute next-char distribution -> draw a sample -> use it as the next input.

  hidden_state:         ndarray (hidden_size, 1) initial memory
                        (typically the most recent state from training).
  seed_index:           int, id of the very first input char used to "prime" the model.
  num_chars_to_sample:  how many chars to emit.
  """
  input_one_hot = np.zeros((vocab_size, 1))
  input_one_hot[seed_index] = 1
  sampled_indices = []
  for t in range(num_chars_to_sample):
    # Identical recurrence to the forward pass in compute_loss_and_gradients.
    hidden_state = np.tanh(
        np.dot(weights_input_to_hidden,  input_one_hot) +
        np.dot(weights_hidden_to_hidden, hidden_state) +
        bias_hidden
    )
    logits = np.dot(weights_hidden_to_output, hidden_state) + bias_output
    probs = np.exp(logits) / np.sum(np.exp(logits))

    # Sample a next-char index from the predicted distribution (stochastic, not argmax).
    next_char_index = np.random.choice(range(vocab_size), p=probs.ravel())

    # Feed the sampled char back in as the next input (one-hot it again).
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[next_char_index] = 1
    sampled_indices.append(next_char_index)
  return sampled_indices


# ----- training loop ----------------------------------------------------------
iteration, data_pointer = 0, 0

# Adagrad accumulators: same shape as each parameter. They keep a running sum of squared
# gradients, which is used to give each parameter its own adaptive (per-coordinate) learning rate.
mem_weights_input_to_hidden  = np.zeros_like(weights_input_to_hidden)
mem_weights_hidden_to_hidden = np.zeros_like(weights_hidden_to_hidden)
mem_weights_hidden_to_output = np.zeros_like(weights_hidden_to_output)
mem_bias_hidden = np.zeros_like(bias_hidden)
mem_bias_output = np.zeros_like(bias_output)

# A reasonable starting value: -log(1 / vocab_size) * sequence_length is the expected
# cross-entropy if the model is uniform over the vocab and we sum over the BPTT window.
# We track an exponential moving average so the printed loss isn't noisy window-to-window.
smooth_loss = -np.log(1.0 / vocab_size) * sequence_length
max_iterations = args.steps

while iteration < max_iterations:
  # Step the data pointer through the corpus in chunks of `sequence_length`.
  # If we run off the end (or we're on iteration 0), reset the hidden state and wrap to the start.
  if data_pointer + sequence_length + 1 >= len(text) or iteration == 0:
    previous_hidden_state = np.zeros((hidden_size, 1))   # reset RNN memory across wraps
    data_pointer = 0

  # Inputs and targets are the same window, shifted by one char.
  # For each t in [0, sequence_length), the model sees text[data_pointer + t]
  # and must predict text[data_pointer + t + 1].
  input_indices  = [char_to_index[char] for char in text[data_pointer    : data_pointer + sequence_length    ]]
  target_indices = [char_to_index[char] for char in text[data_pointer + 1: data_pointer + sequence_length + 1]]

  # Every 100 iterations, draw a 200-char sample from the model to see what it's learning.
  if iteration % 100 == 0:
    sampled_indices = sample(previous_hidden_state, input_indices[0], 200)
    sampled_text = ''.join(index_to_char[i] for i in sampled_indices)
    print('----\n %s \n----' % (sampled_text,))

  # Forward + backward over the window. previous_hidden_state is updated to the last
  # hidden state of *this* window, so the next iteration continues the recurrence smoothly.
  (loss,
   grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output,
   grad_bias_hidden, grad_bias_output,
   previous_hidden_state) = compute_loss_and_gradients(input_indices, target_indices, previous_hidden_state)

  # Exponential moving average of the loss for smoother printing.
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if iteration % 100 == 0:
    print('iter %d, loss: %f' % (iteration, smooth_loss))

  # ----- Adagrad parameter update --------------------------------------------
  # Adagrad accumulates squared gradients in `mem` and divides the learning rate by sqrt(mem).
  # Effect: coordinates that have had large/frequent gradients get smaller effective step sizes,
  # and rarely-updated coordinates get larger ones. The +1e-8 avoids div-by-zero.
  for param, grad, mem in zip(
      [weights_input_to_hidden, weights_hidden_to_hidden, weights_hidden_to_output, bias_hidden, bias_output],
      [grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output, grad_bias_hidden, grad_bias_output],
      [mem_weights_input_to_hidden,  mem_weights_hidden_to_hidden,  mem_weights_hidden_to_output,  mem_bias_hidden,  mem_bias_output]):
    mem += grad * grad
    param += -learning_rate * grad / np.sqrt(mem + 1e-8)

  data_pointer += sequence_length
  iteration    += 1

# Final sample after training finishes, plus the last smoothed loss.
sampled_indices = sample(previous_hidden_state, char_to_index[text[0]], 200)
sampled_text = ''.join(index_to_char[i] for i in sampled_indices)
print('----\n %s \n----' % (sampled_text,))
print('iter %d, loss: %f (done)' % (iteration, smooth_loss))

# Save trained parameters and vocab so we can inspect/visualize the model later.
np.savez(
    'model.npz',
    weights_input_to_hidden=weights_input_to_hidden,
    weights_hidden_to_hidden=weights_hidden_to_hidden,
    weights_hidden_to_output=weights_hidden_to_output,
    bias_hidden=bias_hidden,
    bias_output=bias_output,
    chars=np.array(unique_chars),
    hidden_size=np.array(hidden_size),
    vocab_size=np.array(vocab_size),
)
print('saved trained model to model.npz')
