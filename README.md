# Anomaly Detection

Implementation of various generative neural network models for anomaly detection in Julia, using the Flux framework.

## Models implemented:

| acronym | name | paper |
|---------|------|-------|
| AE | Autoencoder | Vincent, Pascal, et al. "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." Journal of Machine Learning Research 11.Dec (2010): 3371-3408. [link](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)|
| VAE | Variational Autoencoder | Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013). [link](arxiv.org/abs/1312.6114) |
| sVAE | symetric Variational Autoencoder | Pu, Yunchen, et al. "Symmetric variational autoencoder and connections to adversarial learning." arXiv preprint arXiv:1709.01846 (2017). [link](https://arxiv.org/abs/1709.01846) |
| GAN | Generative Adversarial Network | Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014. [link](http://papers.nips.cc/paper/5423-generative-adversarial-nets)|
| fmGAN | GAN with feature-matching loss | Salimans, Tim, et al. "Improved techniques for training gans." Advances in Neural Information Processing Systems. 2016. [link](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans) |

## Experiments:

Experiments are executed on the Loda (Lightweight on-line detector of anomalies) datasets that can be downloaded [here](http://webdav.agents.fel.cvut.cz/data/projects/stegodata/Loda.zip). Tha sampling method is based on [this paper](http://web.engr.oregonstate.edu/~tgd/publications/emmott-das-dietterich-fern-wong-systematic-construction-of-anomaly-detection-benchmarks-from-real-data-odd13.pdf). After downloading the datasets, you can create your own using the experiments/prepare_data.jl function. For experimental evaluation, you need the EvalCurves package:

`>> Pkg.clone("https://github.com/vitskvara/EvalCurves.jl.git")`