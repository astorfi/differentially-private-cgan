# Differentially Private Synthetic Medical Data Generation using Convolutional GANs

[![Name](https://img.shields.io/github/license/astorfi/differentially-private-cgan)](https://github.com/astorfi/differentially-private-cgan/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2010.03549-b31b1b.svg)](https://arxiv.org/abs/2012.11774)

This repository contains an implementation of "Differentially Private Synthetic Medical Data Generation using Convolutional GANs".


For a detailed description of the architecture please read [our paper](https://arxiv.org/abs/2012.11774). Using the code of this repository is allowed with **proper attribution**: Please cite the paper if you use the code from this repository in your work.

## Bibtex

    @article{torfi2020evaluation,
      title={On the Evaluation of Generative Adversarial Networks By Discriminative Models},
      author={Torfi, Amirsina and Beyki, Mohammadreza and Fox, Edward A},
      journal={arXiv preprint arXiv:2010.03549},
      year={2020}
    }



Table of contents
=================

<!--ts-->
   * [Paper Summary](#paper-summary)
   * [Running the Code](#Running-the-Code)
      * [Prerequisites](#Prerequisites)
   * [Collaborators](#Collaborators)
<!--te-->


## Paper Summary

<details>
<summary>Abstract</summary>

 *Deep learning models have demonstrated superior performance in several application problems, such as image classification and speech processing. However, creating a deep learning model using health record data requires addressing certain privacy challenges that bring unique concerns to researchers working in this domain. One effective way to handle such private data issues is to generate realistic synthetic data that can provide practically acceptable data quality and correspondingly the model performance. To tackle this challenge, we develop a differentially private framework for synthetic data generation using Rényi differential privacy. Our approach builds on convolutional autoencoders and convolutional generative adversarial networks to preserve some of the critical characteristics of the generated synthetic data. In addition, our model can also capture the temporal information and feature correlations that might be present in the original data. We demonstrate that our model outperforms existing state-of-the-art models under the same privacy budget using several publicly available benchmark medical datasets in both supervised and unsupervised settings.*

</details>


## Running the Code

### Prerequisites

* Pytorch
* CUDA [strongly recommended]

**NOTE:** PyTorch does a pretty good job in installing required packages but you should have installed CUDA according to PyTorch requirements.
Please refer to [this link](https://pytorch.org/) for further information.

## Collaborators

| [<img src="https://github.com/astorfi.png" width="100px;"/>](https://github.com/astorfi)<br/> [<sub>Amirsina Torfi</sub>](https://github.com/astorfi)
| --- |

<!-- ## Credit

This research conducted at [Virginia Tech](https://vt.edu/) under the supervision of [Dr. Edward A. Fox](http://fox.cs.vt.edu/foxinfo.html). -->
