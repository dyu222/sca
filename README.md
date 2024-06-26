# SCA: Sparse Component Analysis

"Sparse component analysis" is a dimensionality reduction tool that aims to provide more interpretable low-D representations than PCA. Please see our [preprint](https://www.biorxiv.org/content/10.1101/2024.02.05.578988v1) for further details on the method.

## Installation and Dependencies

This package can be installed by: 
```buildoutcfg
git clone https://github.com/glaserlab/sca.git
cd sca
pip install -e .
```
This package requires python 3.6 or higher for a succesful installation.


## Getting started

Let's say we have a matrix **X** that contains the activity of *N* neurons over *T* time points (it is dimension *T* x *N*). We want to reduce the dimensionality to *K* instead of *N*.

To do this, we first import the necessary function and then run SCA:
```python
from sca.models import SCA
sca = SCA(n_components=K)
latent = sca.fit_transform(X)
```

Please see the example jupyter notebook **`Example_1pop.ipynb`** in the notebooks folder for further details. <br><br>

<!--- Additionally, the example notebook **`Example_2pops.ipynb`** demonstrates how to find interpretable low-D representations between two populations. <br> --->
