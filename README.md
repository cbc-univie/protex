protex
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/florianjoerg/protex/workflows/CI/badge.svg)](https://github.com/florianjoerg/protex/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/florianjoerg/protex/branch/main/graph/badge.svg?token=ddqu0BzewU)](https://codecov.io/gh/florianjoerg/protex)
[![Github release](https://badgen.net/github/release/florianjoerg/protex)](https://github.com/florianjoerg/protex/releases/)
[![GitHub license](https://img.shields.io/github/license/florianjoerg/protex?color=green)](https://github.com/florianjoerg/protex/blob/main/LICENSE)
[![GH Pages](https://github.com/florianj77/protex/actions/workflows/gh_pages.yml/badge.svg)](https://github.com/florianj77/protex/actions/workflows/gh_pages.yml)
[![docs stable](https://img.shields.io/badge/docs-stable-5077AB.svg?logo=read%20the%20docs)](https://florianjoerg.github.io/protex/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/protex.svg)](https://anaconda.org/conda-forge/protex)

[//]: <[![GitHub forks](https://img.shields.io/github/forks/florianj77/protex)](https://github.com/florianj77/protex/network)>
[//]: <[![Github tag](https://badgen.net/github/tag/florianj77/protex)](https://github.com/florianj77/protex/tags/)>
[//]: <[![GitHub issues](https://img.shields.io/github/issues/florianj77/protex?style=flat)](https://github.com/florianj77/protex/issues)>
[//]: <[![GitHub stars](https://img.shields.io/github/stars/florianj77/protex)](https://github.com/florianj77/protex/stargazers)>
[//]: <[![codecov](https://codecov.io/gh/florianj77/protex/branch/main/graph/badge.svg?token=ddqu0BzewU)](https://codecov.io/gh/florianj77/protex)>


Protex enables proton exchange in solvent molecules using openMM. The simulations are augmented by possible (de-)protonation events using a single topology approach with two different λ-states.[^1]

## Installation

protex can be easily installed from conda-forge:
```
conda install protex -c conda-forge
```
Alternatively clone this repo and install from source:
```
git clone https://github.com/florianjoerg/protex.git
cd protex
pip install .
```

## Usage
Please see the [documentation](https://florianjoerg.github.io/protex) for usage examples.

## Maintainers

- Florian Jörg <florian.joerg@univie.ac.at> (University of Vienna)
- Márta Gödeny (University of Vienna)

### Copyright

Copyright (c) 2023, Florian Joerg & Marcus Wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.

[^1]: Joerg F., Wieder M., Schröder C. *Frontiers in Chemistry: Molecular Liquids* (2023), 11, [DOI]( https://doi.org/10.3389/fchem.2023.1140896) 
