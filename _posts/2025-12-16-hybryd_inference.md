---
layout: post
title: The Best of Both Worlds: Hybrid Inference Combining Physics and Deep Learning
---

# Hybrid Inference: Combining Graphical Models with Graph Neural Networks

*A NeurIPS 2019 Paper by Garcia Satorras, Akata, and Welling*

## Introduction: Reconciling Classical and Learning-Based Approaches

Machine learning has traditionally operated within two distinct paradigms. **Graphical models** encode domain knowledge through structured representations of the data generating process. They leverage prior knowledge about the world and excel in low-data regimes due to strong inductive bias. Conversely, **neural networks** learn patterns directly from observations and can capture complex nonlinear relationships. However, they typically require substantial amounts of data to achieve good generalization.

A 2019 NeurIPS paper addresses the relationship between these approaches: **"Combining Generative and Discriminative Models for Hybrid Inference."** The authors propose a hybrid algorithm that integrates classical probabilistic inference with learned neural components. The method automatically balances between graphical inference and learned inference depending on data availability. Experimental results demonstrate performance improvements over either approach in isolation across multiple domains.

---

## Motivation: The Limitations of Pure Approaches

Consider the problem of trajectory estimation. A classical approach would use a Kalman filter, which combines a physics model with noisy measurements to estimate the true state. The Kalman filter is optimal when the model accurately reflects the underlying system dynamics—linear, Gaussian, and precisely specified.

Real-world systems rarely meet these assumptions. Unmodeled dynamics exist: wind effects in tracking problems, friction variations in mechanical systems, or complex environmental factors in robotics. A model-based approach trained on domain knowledge becomes suboptimal when these factors are significant.

The alternative approach—training a neural network end-to-end on data—bypasses explicit modeling but demands substantial training data. With limited observations, neural networks typically produce poor estimates due to underfitting.

This paper proposes an alternative formulation: rather than replacing the physics model with a neural network, the approach learns to correct the errors of the classical method. This residual learning strategy leverages the structure provided by domain knowledge while adding flexibility through data-driven components.

---

## Core Principle: Learning Residual Corrections

The fundamental insight underlying the hybrid approach is that residual errors have simpler structure than raw signals. Instead of learning complete dynamics:

$$\hat{x}_k^{\text{NN}} = f_{\text{NN}}(y)$$

the method learns only the correction term:

$$\hat{x}_k^{\text{Hybrid}} = \hat{x}_k^{\text{Physics}} + \epsilon_k$$

where $\epsilon_k$ represents a learned correction factor derived from data.

This formulation offers practical advantages. Residual signals are often approximately linear even when underlying dynamics are highly nonlinear. Neural networks can learn simpler patterns with substantially fewer training examples compared to learning full dynamics from scratch. The empirical results support this principle across multiple experimental settings.

---

## Technical Framework: Message Passing Integration

The paper formulates inference as iterative message passing on graphical model structures. This allows unified treatment of classical and learned components.

### Graphical Model Messages

Classical inference operates through three directional messages at each node. For a hidden Markov process, these derive from:

**Transition component:**
$$\mu_{x_{k-1} \to x_k} = -Q^{-1}(x_k - F x_{k-1})$$

**Backward smoothing:**
$$\mu_{x_{k+1} \to x_k} = F^T Q^{-1}(x_{k+1} - F x_k)$$

**Measurement component:**
$$\mu_{y_k \to x_k} = H^T R^{-1}(y_k - H x_k)$$

These messages encode structural knowledge from the generative model. In the Kalman filter case, they arise from linear Gaussian assumptions.

### Graph Neural Network Refinements

The hybrid architecture augments classical messages with learned refinements through a graph neural network:

1. **Compute learned messages** via learnable edge functions:
$$m_{k,n}^{(i)} = z_{k,n} \cdot f_e(h_{x_k}^{(i)}, h_{v_n}^{(i)}, \mu_{v_n \to x_k})$$

2. **Aggregate messages** across edges:
$$U_k^{(i)} = \sum_{v_n \neq x_k} m_{k,n}^{(i)}$$

3. **Update node representations** using gated recurrent units:
$$h_{x_k}^{(i+1)} = \text{GRU}(U_k^{(i)}, h_{x_k}^{(i)})$$

4. **Decode correction signal**:
$$\epsilon_k^{(i+1)} = f_{\text{dec}}(h_{x_k}^{(i+1)})$$

### Combined State Update

The algorithm iterates this process for a fixed number of steps. Each iteration combines classical and learned components:

$$x_k^{(i+1)} = x_k^{(i)} + \gamma \left( M_k^{(i)} + \epsilon_k^{(i+1)} \right)$$

The term $M_k^{(i)}$ represents classical messages, while $\epsilon_k^{(i+1)}$ denotes the learned correction. Unrolling this iteration produces a recurrent neural network that performs "learned belief propagation."

---

## Experimental Evaluation

The paper presents three experimental scenarios of increasing complexity:

### Experiment 1: Linear Dynamics with Model Mismatch

**Experimental setup:** Synthetic trajectories generated from physics including air drag:
$$\frac{\partial p}{\partial t} = v, \quad \frac{\partial v}{\partial t} = a - cv, \quad \frac{\partial a}{\partial t} = -\tau v$$

The graphical model incorporates only simple uniform motion assumptions, creating deliberate model mismatch.

**Empirical findings:**

Performance across different training set sizes shows distinct patterns:
- With approximately 100 training samples: Kalman filter achieves lower error (physics assumption remains approximately valid)
- With 1K to 10K samples: Hybrid method produces superior results
- With 100K samples: Error rates converge across methods, with hybrid maintaining a marginal advantage

The results indicate that the hybrid approach transitions smoothly between regimes, leveraging prior knowledge when data is limited and utilizing learning when data becomes abundant.

### Experiment 2: Lorenz Attractor System

The Lorenz system represents a challenging nonlinear, chaotic problem:

$$\frac{dz_1}{dt} = 10(z_2 - z_1)$$
$$\frac{dz_2}{dt} = z_1(28 - z_3) - z_2$$
$$\frac{dz_3}{dt} = z_1 z_2 - \frac{8}{3}z_3$$

The graphical model uses Taylor expansion approximations to approximate nonlinear dynamics.

**Sample efficiency comparison:**

| Method | Samples for MSE=0.1 | Samples for MSE=0.05 |
|--------|---------------------|----------------------|
| Pure GNN (no physics) | ~5,000 | ~90,000 |
| Hybrid (weak physics, J=1) | ~500 | ~5,000 |
| Hybrid (stronger physics, J=5) | ~400 | ~4,000 |

The hybrid approach achieves 10-20× sample efficiency improvement over pure neural learning through incorporation of physics priors.

Performance on a 50K-sample training trajectory:

- Raw observations (baseline): MSE = 0.2462
- Pure GNN approach: MSE = 0.0613
- Extended Kalman Smoother: MSE = 0.0372
- Hybrid method: MSE = **0.0169**

The superior performance indicates that residual error structure remains simpler than raw dynamics. The neural network learns the difference between classical estimates and true values more efficiently than learning full dynamics.

### Experiment 3: Real-World Robot Localization

The Michigan NCLT dataset provides real-world validation. A Segway robot with GPS traverses the University of Michigan campus, collecting noisy GPS signals and ground truth positions.

| Method | Mean Squared Error |
|--------|-------------------|
| Raw GPS observations | 3.4974 |
| Classical Kalman Smoother | 3.0099 |
| Pure GNN | 1.7929 |
| Hybrid model | **1.4109** |

The hybrid approach achieves a 53% reduction in error relative to the Kalman filter and 21% improvement over pure neural learning. This validates the approach on real-world data where model mismatch is inevitable.

---

## Technical Implementation

### Training Methodology

The model is trained end-to-end using a weighted loss function that emphasizes later iterations:

$$\text{Loss}(\Theta) = \sum_{i=1}^{N} w_i \cdot L(y_{\text{true}}, \hat{x}^{(i)})$$

where $w_i = i/N$ gives increasing weight to later iterations.

The training procedure consists of three stages:

1. **State initialization** at values maximizing observation likelihood (e.g., position initialized to observed position)
2. **Graphical model tuning** of hyperparameters—primarily noise covariance matrices, treated identically to classical Kalman filter calibration
3. **GNN component training** via backpropagation through unrolled iterations

### Implementation Details

Standard hyperparameters:
- Number of iterations: N = 50
- Hidden dimension: 48 features
- Optimization: Adam with learning rate $10^{-3}$
- Step size: $\gamma = 0.005$
- Message functions: 2-layer MLPs with LeakyReLU activations
- Recurrent units: GRU cells

Architectural choices: Edge-type-specific message functions distinguish transition edges from measurement edges. Translation invariance in trajectory problems is achieved by using positional differences rather than absolute positions.

---

## Generalizability and Scope

Current work focuses on sequential models (Hidden Markov Models). However, the framework generalizes beyond this setting. Message passing equations apply to arbitrary graph structures. By modifying edge and node configurations, the method extends to:

- Undirected graphical models
- General factor graphs
- Discrete variable systems

Future extensions could include learning graphical model structure, applying Bayesian treatments to prior-data balance, and extending to inference problems beyond trajectory estimation.

---

## Interpretation and Implications

The hybrid approach reconciles two research traditions. Classical methods provide interpretability and sample efficiency through structural assumptions. Modern neural networks provide flexibility and capacity through data-driven learning. Integration of these approaches offers practical advantages:

**Sample efficiency:** Incorporation of domain knowledge reduces data requirements substantially. In the Lorenz experiment, weak physics knowledge reduced sample requirements by an order of magnitude.

**Interpretability:** The graphical model backbone maintains interpretability compared to purely learned approaches, as components remain connected to domain knowledge.

**Robustness:** Physical constraints provide regularization effects beneficial for out-of-distribution generalization.

**Extensibility:** The framework naturally extends to different graphical model structures and problem domains without fundamental changes.

The work demonstrates that classical probabilistic inference and modern deep learning need not compete. Integration of both paradigms produces systems that leverage advantages of each approach.

---

## Limitations

The presented work has several scoping limitations:

- Current experiments restrict focus to sequential models
- Cross-validation for balancing components may not scale to large problem instances
- Computational cost analysis is not thoroughly developed
- Claims regarding interpretability require further investigation

Despite these limitations, the core contribution—demonstrating practical value of hybrid inference combining classical and neural approaches—remains significant.

---

## Conclusion

This work addresses the complementary nature of classical probabilistic methods and neural learning approaches. Rather than selecting between graphical models and neural networks, the hybrid framework integrates both components. Classical inference provides structure and sample efficiency; neural refinement adds flexibility.

Experimental validation across three domains—synthetic linear dynamics, chaotic nonlinear systems, and real robot localization—demonstrates consistent improvements over either pure approach. Performance gains range from marginal in data-rich regimes to substantial (10-20×) in data-limited settings.

The integration of physics-based and learning-based inference represents a practical direction for systems where domain knowledge exists alongside data. Such hybrid approaches may become increasingly important as the field matures beyond the historical binary choice between symbolic and neural methods.

---

## References

- **Original Paper:** Victor Garcia Satorras, Zeynep Akata, Max Welling. "Combining Generative and Discriminative Models for Hybrid Inference." NeurIPS 2019.
- **Code:** Available at https://github.com/vgsatorras/hybrid-inference
- **Related Work:** Graphical models, belief propagation, graph neural networks, amortized inference, neural message passing

---

*This post summarizes research published at NeurIPS 2019. Technical content and experimental results are drawn directly from the original paper.*