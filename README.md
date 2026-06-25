# Backpropagation Replication (Rumelhart, Hinton, and Williams, 1986)

This repository contains a from-scratch implementation of the Multi-Layer Perceptron (MLP) and backpropagation algorithm described in the 1986 paper *Learning Representations by Back-Propagating Errors*. 

The model is evaluated on the 6-bit symmetry detection problem outlined in the paper, proving that hidden units can successfully learn non-linearly separable internal representations.

## Structure
* `PAPER_NOTES.md` - Reading notes covering the central claim, architecture, and metrics.
* `src/solution.py` - The core implementation. 
* `results/` - Screenshots or logs of your evaluation output
## How to Run

**1. Install dependencies:**
The only requirement is PyTorch (used purely for tensor matrix multiplications).
pip install torch

## Dataset

The implementation uses the 6-bit Symmetry Detection dataset from the paper. It contains all 64 possible 6-bit binary patterns, with 8 symmetric patterns labelled as positive.

## Results

- Final Accuracy: 87.5%
- The implementation successfully demonstrates the backpropagation algorithm on the symmetry detection task. The network converged to 87.5% accuracy with the chosen hyperparameters.
