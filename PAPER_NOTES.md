
# PAPER_NOTES.md

## Paper: Learning Representations by Back-Propagating Errors (Rumelhart, Hinton & Williams, 1986)

### 1. Central Claim

**The Main Point:**
The paper shows that neural networks with hidden layers can learn difficult problems using the backpropagation algorithm. Instead of telling the network what features to look for, it learns them by itself during training.

**Why it Works:**
The network first makes a prediction, checks how wrong it is, and then updates its weights to reduce the error. By repeating this process many times, it slowly gets better at making the correct predictions.

---

### 2. Core Architecture

I based my implementation on the method explained in the paper.

**Forward Pass:**
The input is passed through the hidden layer and then to the output layer. The network uses the Sigmoid activation function to calculate the output values.

**Backward Pass:**
After getting the output, the network compares it with the correct answer. It then sends the error backwards through the network and updates the weights using backpropagation so the next prediction is better.

**Weight Updates:**
The weights are updated after every training example using gradient descent. In my implementation, I also used momentum to help the network train more smoothly and converge faster.

---

### 3. Dataset and Evaluation

I tested my implementation using the same **6-bit Symmetry Detection** task used in the paper.

**Dataset:**
I generated all 64 possible 6-bit binary patterns. Out of these, only 8 patterns are perfectly symmetric, while the rest are not.

**Evaluation:**
The goal is to correctly identify whether a pattern is symmetric or not. I will measure the training loss and the final classification accuracy to see how well the network learns.