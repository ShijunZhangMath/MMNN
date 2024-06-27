<img width="600" alt="kan_plot" src="https://github.com/KindXiaoming/pykan/assets/23551623/a2d2d225-b4d2-4c1e-823e-bc45c7ea96f9">

# Multi-component and Multi-layer Neural Networks (MMNNs)

This is the github repo for the paper "Structured and Balanced Multi-component and Multi-layer Neural Networks". 

In this work, we design a balanced multi-component and multi-layer neural network (MMNN) structure to approximate functions with complex features with both accuracy and efficiency in terms of degrees of freedom and computation cost. The key idea is based on a multi-layer smooth decomposition each component of which can be approximated effectively by a single-layer network. Using balanced structures in the network reduces the need for large degrees of freedom compared to fully connected networks and makes the optimization/learning process more efficient. Extensive numerical experiments are presented to illustrate the effectiveness of MMNNs in approximating high oscillatory functions and its automatic adaptivity in capturing localized features. 

## Architecture of MMNNs

Each layer of MMNN is a (shallow) neural network of the form

$$ \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{A}\sigma(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}) + \boldsymbol{c}, $$

where $\boldsymbol{W} \in \mathbb{R}^{n \times d_{\text{in}}}$, $\boldsymbol{A} \in \mathbb{R}^{d_{\text{out}} \times n}$, and $n$ is the width of this network. Here, $\sigma : \mathbb{R} \rightarrow \mathbb{R}$ represents the activation function that can be applied elementwise to vector inputs. 

We call each element of $\boldsymbol{h}$, i.e., $\boldsymbol{h}[i]$ for $i = 1, 2, \ldots, d_{\text{out}}$, a component. Here are a few key features of $\boldsymbol{h}$:

1. Each component is viewed as a linear combination of basis functions $\sigma(\boldsymbol{W}[i, :] \cdot \boldsymbol{x} + \boldsymbol{b}[i])$, $i = 1, 2, \ldots, n$, which is a function in $\boldsymbol{x}$ as a whole.
2. Different components of $\boldsymbol{h}$ share the same set of basis with different coefficients $\boldsymbol{A}[i, :]$ and $\boldsymbol{c}[:]$.
3. Only $(\boldsymbol{A}, \boldsymbol{c})$ are trained while $(\boldsymbol{W}, \boldsymbol{b})$ are randomly assigned and fixed.
4. The output dimension $d_{\text{out}}$ and network width $n$ can be tuned according to the intrinsic dimension and complexity of the problem.

A MMNN is a multi-layer composition of $\boldsymbol{h}_i$, i.e.,

$$ \boldsymbol{h} = \boldsymbol{h}_m \circ \cdots \circ \boldsymbol{h}_2 \circ \boldsymbol{h}_1 $$

where each $\boldsymbol{h}_i : \mathbb{R}^{d_{i-1}} \to \mathbb{R}^{d_i}$ is a multi-component shallow network defined of width $n_i$, where

$$ d_0 = d_{\text{in}}, \quad d_1, \ldots, d_{m-1} \ll n_i, \quad d_m = d_{\text{out}} $$

The width of this MMNN is defined as $\max\{n_i : i = 1, 2, \ldots, m-1\}$, the rank as $\max\{d_i : i = 1, 2, \ldots, m-1\}$, and the depth as $m$. To simplify, we denote a network with width $w$, rank $r$, and depth $l$ using the compact notation $(w, r, l)$. See Figure 1(a) for an illustration of MMNN of size (4, 2, 2). 

In comparison, each layer in a typical deep FCNN takes the form $\sigma(\widetilde{\boldsymbol{W}}x)$, and each hidden neuron is individually a function of the input $x$ or each point $x \in \mathbb{R}^{d_{\text{in}}}$ is mapped to $\mathbb{R}^n$, where $n$ is the layer width. All weights $\widetilde{\boldsymbol{W}}$ are training parameters. In MMNN, each layer is composed of multiple components $\widetilde{\boldsymbol{A}}\sigma(\widetilde{\boldsymbol{W}}x)$. Each component is a linear combination of randomly parameterized hidden neurons $\sigma(\widetilde{\boldsymbol{W}}x)$, which can be more effectively and stably trained through $\widetilde{\boldsymbol{A}}$ as a smooth decomposition/transformation. Typically the number of components $d_{\text{out}}$ is (much) smaller than the layer width $n$ in our experiments.
In contrast, an FCNN $\phi$ can be expressed in the following composition form

$$ \phi = L_L \circ \sigma \circ L_{L-1} \circ \cdots \circ \sigma \circ L_1 \circ \sigma \circ L_0 $$

where $L_i$ is an affine linear map given by $L_i(y) = \boldsymbol{W}_i \cdot y + \boldsymbol{b}_i$. Readers are referred to Figure 1(b) for an illustration and also a comparison with the MMNN.

For very deep MMNNs, one can borrow ideas from ResNets to address the gradient vanishing issue, making training more efficient. Incorporating this idea, we propose a new architecture given by a multi-layer composition of $I + \boldsymbol{h}_i$, i.e.,

$$ \boldsymbol{h} = \boldsymbol{h}_m \circ (I + \boldsymbol{h}_{m-1}) \circ \cdots \circ (I + \boldsymbol{h}_3) \circ (I + \boldsymbol{h}_2) \circ \boldsymbol{h}_1 $$

where each $\boldsymbol{h}_i : \mathbb{R}^{d_{i-1}} \rightarrow \mathbb{R}^{d_i}$ is a multi-component shallow network defined in (1) with width $n_i$,

$$ d_0 = d_{\text{in}}, \quad d_1 = \cdots = d_{m-1} = r \ll n_i, \quad d_m = d_{\text{out}} $$

and $I$ is the identity map. We call this architecture ResMMNN.


<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

## Accuracy
**KANs have faster scaling than MLPs. KANs have better accuracy than MLPs with fewer parameters.**

Please set `torch.set_default_dtype(torch.float64)` if you want high precision.

**Example 1: fitting symbolic formulas**
<img width="1824" alt="Screenshot 2024-04-30 at 10 55 30" src="https://github.com/KindXiaoming/pykan/assets/23551623/e1fc3dcc-c1f6-49d5-b58e-79ff7b98a49b">

**Example 2: fitting special functions**
<img width="1544" alt="Screenshot 2024-04-30 at 11 07 20" src="https://github.com/KindXiaoming/pykan/assets/23551623/b2124337-cabf-4e00-9690-938e84058a91">

**Example 3: PDE solving**
<img width="1665" alt="Screenshot 2024-04-30 at 10 57 25" src="https://github.com/KindXiaoming/pykan/assets/23551623/5da94412-c409-45d1-9a60-9086e11d6ccc">

**Example 4: avoid catastrophic forgetting**
<img width="1652" alt="Screenshot 2024-04-30 at 11 04 36" src="https://github.com/KindXiaoming/pykan/assets/23551623/57d81de6-7cff-4e55-b8f9-c4768ace2c77">

## Interpretability
**KANs can be intuitively visualized. KANs offer interpretability and interactivity that MLPs cannot provide. We can use KANs to potentially discover new scientific laws.**

**Example 1: Symbolic formulas**
<img width="1510" alt="Screenshot 2024-04-30 at 11 04 56" src="https://github.com/KindXiaoming/pykan/assets/23551623/3cfd1ca2-cd3e-4396-845e-ef8f3a7c55ef">

**Example 2: Discovering mathematical laws of knots**
<img width="1443" alt="Screenshot 2024-04-30 at 11 05 25" src="https://github.com/KindXiaoming/pykan/assets/23551623/80451ac2-c5fd-45b9-89a7-1637ba8145af">

**Example 3: Discovering physical laws of Anderson localization**
<img width="1295" alt="Screenshot 2024-04-30 at 11 05 53" src="https://github.com/KindXiaoming/pykan/assets/23551623/8ee507a0-d194-44a9-8837-15d7f5984301">

**Example 4: Training of a three-layer KAN**

![kan_training_low_res](https://github.com/KindXiaoming/pykan/assets/23551623/e9f215c7-a393-46b9-8528-c906878f015e)



## Installation
Pykan can be installed via PyPI or directly from GitHub. 

**Pre-requisites:**

```
Python 3.9.7 or higher
pip
```

**Installation via github**

```
python -m venv pykan-env
source pykan-env/bin/activate  # On Windows use `pykan-env\Scripts\activate`
pip install git+https://github.com/KindXiaoming/pykan.git
```

**Installation via PyPI:**
```
python -m venv pykan-env
source pykan-env/bin/activate  # On Windows use `pykan-env\Scripts\activate`
pip install pykan
```
Requirements

```python
# python==3.9.7
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```

**Optional: Conda Environment Setup**
For those who prefer using Conda:
```
conda create --name pykan-env python=3.9.7
conda activate pykan-env
pip install git+https://github.com/KindXiaoming/pykan.git  # For GitHub installation
# or
pip install pykan  # For PyPI installation
```

## Computation requirements

Examples in [tutorials](tutorials) are runnable on a single CPU typically less than 10 minutes. All examples in the paper are runnable on a single CPU in less than one day. Training KANs for PDE is the most expensive and may take hours to days on a single CPU. We use CPUs to train our models because we carried out parameter sweeps (both for MLPs and KANs) to obtain Pareto Frontiers. There are thousands of small models which is why we use CPUs rather than GPUs. Admittedly, our problem scales are smaller than typical machine learning tasks, but are typical for science-related tasks. In case the scale of your task is large, it is advisable to use GPUs.

## Documentation
The documentation can be found [here](https://kindxiaoming.github.io/pykan/).

## Tutorials

**Quickstart**

Get started with [hellokan.ipynb](./hellokan.ipynb) notebook.

**More demos**

More Notebook tutorials can be found in [tutorials](tutorials).

## Advice on hyperparameter tuning
Many intuition about MLPs and other networks may not directy transfer to KANs. So how can I tune the hyperparameters effectively? Here is my general advice based on my experience playing with the problems reported in the paper. Since these problems are relatively small-scale and science-oriented, it is likely that my advice is not suitable to your case. But I want to at least share my experience such that users can have better clues where to start and what to expect from tuning hyperparameters.

* Start from a simple setup (small KAN shape, small grid size, small data, no reguralization `lamb=0`). This is very different from MLP literature, where people by default use widths of order `O(10^2)` or higher. For example, if you have a task with 5 inputs and 1 outputs, I would try something as simple as `KAN(width=[5,1,1], grid=3, k=3)`. If it doesn't work, I would gradually first increase width. If that still doesn't work, I would consider increasing depth. You don't need to be this extreme, if you have better understanding about the complexity of your task.

* Once an acceptable performance is achieved, you could then try refining your KAN (more accurate or more interpretable).

* If you care about accuracy, try grid extention technique. An example is [here](https://kindxiaoming.github.io/pykan/Examples/Example_1_function_fitting.html). But watch out for overfitting, see below.

* If you care about interpretability, try sparsifying the network with, e.g., `model.train(lamb=0.01)`. It would also be advisable to try increasing lamb gradually. After training with sparsification, plot it, if you see some neurons that are obvious useless, you may call `pruned_model = model.prune()` to get the pruned model. You can then further train (either to encourage accuracy or encouarge sparsity), or do symbolic regression.

* I also want to emphasize that accuracy and interpretability (and also parameter efficiency) are not necessarily contradictory, e.g., Figure 2.3 in [our paper](https://arxiv.org/pdf/2404.19756). They can be positively correlated in some cases but in other cases may dispaly some tradeoff. So it would be good not to be greedy and aim for one goal at a time. However, if you have a strong reason why you believe pruning (interpretability) can also help accuracy, you may want to plan ahead, such that even if your end goal is accuracy, you want to push interpretability first. 

* Once you get a quite good result, try increasing data size and have a final run, which should give you even better results!

Disclaimer: Try the simplest thing first is the mindset of physicists, which could be personal/biased but I find this mindset quite effective and make things well-controlled for me. Also, The reason why I tend to choose a small dataset at first is to get faster feedback in the debugging stage (my initial implementation is slow, after all!). The hidden assumption is that a small dataset behaves qualitatively similar to a large dataset, which is not necessarily true in general, but usually true in small-scale problems that I have tried. To know if your data is sufficient, see the next paragraph.

Another thing that would be good to keep in mind is that please constantly checking if your model is in underfitting or overfitting regime. If there is a large gap between train/test losses, you probably want to increase data or reduce model (`grid` is more important than `width`, so first try decreasing `grid`, then `width`). This is also the reason why I'd love to start from simple models to make sure that the model is first in underfitting regime and then gradually expands to the "Goldilocks zone".

## Citation
```python
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

## Contact
If you have any questions, please contact zmliu@mit.edu

## Author's note
I would like to thank everyone who's interested in KANs. When I designed KANs and wrote codes, I have math & physics examples (which are quite small scale!) in mind, so did not consider much optimization in efficiency or reusability. It's so honored to receive this unwarranted attention, which is way beyond my expectation. So I accept any criticism from people complaning about the efficiency and resuability of the codes, my apology. My only hope is that you find `model.plot()` fun to play with :).

For users who are interested in scientific discoveries and scientific computing (the orginal users intended for), I'm happy to hear your applications and collaborate. This repo will continue remaining mostly for this purpose, probably without signifiant updates for efficiency. In fact, there are already implmentations like [efficientkan](https://github.com/Blealtan/efficient-kan) or [fouierkan](https://github.com/GistNoesis/FourierKAN/) that look promising for improving efficiency.

For users who are machine learning focus, I have to be honest that KANs are likely not a simple plug-in that can be used out-of-the box (yet). Hyperparameters need tuning, and more tricks special to your applications should be introduced. For example, [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks) suggests that KANs should better be used in latent space (need embedding and unembedding linear layers after inputs and before outputs). [KANRL](https://github.com/riiswa/kanrl) suggests that some trainable parameters should better be fixed in reinforcement learning to increase training stability. The extra tricks required by KAN (e.g., grid updates and grid extension) beyond MLPs make it sometimes confusing on how to use them so we should be extra careful, e.g., [Prof. George Karniadakis' post on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7196684191479070721/) and [my response](https://www.linkedin.com/feed/update/urn:li:activity:7197097659017379840/) is an example.

The most common question I've been asked lately is whether KANs will be next-gen LLMs. I don't have good intuition about this. KANs are designed for applications where one cares about high accuracy and/or interpretability. We do care about LLM interpretability for sure, but interpretability can mean wildly different things for LLM and for science. Do we care about high accuracy for LLMs? I don't know, scaling laws seem to imply so, but probably not too high precision. Also, accuracy can also mean different things for LLM and for science. This subtlety makes it hard to directly transfer conclusions in our paper to LLMs, or machine learning tasks in general. However, I would be very happy if you have enjoyed the high-level idea (learnable activation functions on edges, or interacting with AI for scientific discoveries), which is not necessariy *the future*, but can hopefully inspire and impact *many possible futures*. As a physicist, the message I want to convey is less of "KANs are great", but more of "try thinking of current architectures critically and seeking fundamentally different alternatives that can do fun and/or useful stuff".

I would like to welcome people to be critical of KANs, but also to be critical of critiques as well. Practice is the only criterion for testing understanding (实践是检验真理的唯一标准). We don't know many things beforehand until they are really tried and shown to be succeeding or failing. As much as I'm willing to see success mode of KANs, I'm equally curious about failure modes of KANs, to better understand the boundaries. KANs and MLPs cannot replace each other (as far as I can tell); they each have advantages in some settings and limitations in others. I would be intrigued by a theoretical framework that encompasses both and could even suggest new alternatives (physicists love unified theories, sorry :).

