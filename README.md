# Representation Learning-based Network Intrusion Detection System (RL-NIDS)

This repository implements the RL-NIDS system proposed in the paper ["Representation learning-based network intrusion detection system by capturing explicit and implicit feature interactions"](https://www.sciencedirect.com/science/article/pii/S0167404821003273).

## Methodology

RL-NIDS is a novel network intrusion detection system that learns both explicit and implicit feature interactions through representation learning. The system consists of two main modules:

### 1. Feature Value Representation Learning (FVRL)

The FVRL module explicitly captures interactions between categorical feature values through:

1. **Value-Value Coupling Matrices**:
   - Constructs two matrices (M0 and Mc) that quantify:
     - Occurrence-based relationships (how often values appear together)
     - Co-occurrence-based relationships (how values influence each other's presence)

2. **Multi-grain Clustering**:
   - Performs k-means clustering on the coupling matrices with different cluster numbers
   - Captures feature interactions at multiple granularities

3. **Autoencoder**:
   - Compresses the high-dimensional cluster indicators
   - Learns compact feature value embeddings

### 2. Neural Network for Representation Learning (NNRL)

The NNRL module learns implicit feature interactions through:

1. **Deep Neural Network**:
   - Takes concatenated numeric features and FVRL embeddings as input
   - Learns hierarchical representations through multiple dense layers

2. **Hybrid Loss Function**:
   - Classification loss (cross-entropy): Ensures correct attack classification
   - Triplet loss: Handles class imbalance by learning discriminative representations
     - Forms triplets (anchor, positive, negative) based on class labels
     - Minimizes distance between same-class samples while maximizing distance to different-class samples

3. **Hard Triplet Mining**:
   - Focuses on "hard" triplets where the negative is close to the anchor
   - Improves decision boundaries for better intrusion detection

## Key Advantages

1. **Handles Heterogeneous Features**:
   - Effectively combines numeric and categorical features
   - Captures complex interactions between different feature types

2. **Addresses Class Imbalance**:
   - Triplet loss helps detect rare attack types
   - FVRL is unsupervised and less affected by label distribution

3. **Interpretable Representations**:
   - Explicit feature value embeddings provide insights into feature interactions
   - Learned representations improve performance across different classifiers

## Results

As reported in the original paper, RL-NIDS achieves:

| Dataset  | Accuracy | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| NSL-KDD  | 81.38%   | 83.69%    | 81.38% | 78.79%   |
| AWID     | 95.72%   | 93.98%    | 95.72% | 94.47%   |

The system outperforms state-of-the-art feature selection and deep learning methods, particularly for rare attack classes like R2L (43.6% F1 improvement over second-best method).