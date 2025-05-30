[//]: # (Badges)
[![CI](https://github.com/cbc-univie/protex/actions/workflows/CI.yaml/badge.svg)](https://github.com/cbc-univie/protex/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/florianjoerg/protex/branch/main/graph/badge.svg?token=ddqu0BzewU)](https://codecov.io/gh/florianjoerg/protex)
[![Github release](https://badgen.net/github/release/cbc-univie/protex)](https://github.com/cbc-univie/protex/releases/)
[![GitHub license](https://img.shields.io/github/license/florianjoerg/protex?color=green)](https://github.com/florianjoerg/protex/blob/main/LICENSE)
[![GH Pages](https://github.com/cbc-univie/protex/actions/workflows/gh_pages.yml/badge.svg)](https://github.com/cbc-univie/protex/actions/workflows/gh_pages.yml)
[![docs stable](https://img.shields.io/badge/docs-stable-5077AB.svg?logo=read%20the%20docs)](https://cbc-univie.github.io/protex/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/protex.svg)](https://anaconda.org/conda-forge/protex)

[//]: <[![GitHub Actions Build Status](https://github.com/cbc-univie/protex/workflows/CI/badge.svg)](https://github.com/cbc-univie/protex/actions?query=workflow%3ACI)>
[//]: <[![GitHub forks](https://img.shields.io/github/forks/florianj77/protex)](https://github.com/florianj77/protex/network)>
[//]: <[![Github tag](https://badgen.net/github/tag/florianj77/protex)](https://github.com/florianj77/protex/tags/)>
[//]: <[![GitHub issues](https://img.shields.io/github/issues/florianj77/protex?style=flat)](https://github.com/florianj77/protex/issues)>
[//]: <[![GitHub stars](https://img.shields.io/github/stars/florianj77/protex)](https://github.com/florianj77/protex/stargazers)>
[//]: <[![codecov](https://codecov.io/gh/florianj77/protex/branch/main/graph/badge.svg?token=ddqu0BzewU)](https://codecov.io/gh/florianj77/protex)>

<p align="center">
 <a href="https://florianjoerg.github.io/protex" target="_blank" rel="noopener noreferrer">
  <img src="https://github.com/florianjoerg/protex/blob/main/docs/assets/images/protex_logo.png" alt="Protex Logo"/>
 </a>
</p>

Protex enables proton exchange in solvent molecules using openMM. The simulations are augmented by possible (de-)protonation events using a single topology approach with two different λ-states.[^1]

## Installation

protex can be easily installed from conda-forge:
```
conda install protex -c conda-forge
```
Alternatively clone this repo and install from source:
```
git clone https://github.com/cbc-univie/protex.git
cd protex
pip install .
```

## Usage
Please see the [documentation](https://cbc-univie.github.io/protex/) for usage examples.

## Maintainers

- Florian Jörg <florian.joerg@univie.ac.at> (University of Vienna)
- Márta Gödeny (University of Vienna)

### Copyright

:copyright: 2024, Florian Joerg & Marcus Wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.

[^1]: Joerg F., Wieder M., Schröder C. *Frontiers in Chemistry: Molecular Liquids* (2023), 11, [DOI]( https://doi.org/10.3389/fchem.2023.1140896) 
