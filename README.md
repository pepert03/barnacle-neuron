[![made-with-python](https://img.shields.io/badge/Made%20with-Python-informational?style=flat-square)](https://www.python.org/)
[![made-with-pygame](https://img.shields.io/badge/Made%20with-Pygame-informational?style=flat-square)](https://www.pygame.org/news)


<div style="text-align:left"><img src="images/logo.jpg" height="300"></div>


# The Barnacle Neuron
### Basic implementation of a Neural Network from scratch

## Table of Contents

- [About this Project](#about-this-project)
    - [Detailed Description](#detailed-description)
    - [Dataset Creation](#dataset-creation)
    - [Basic Layers](#basic-layers)
    - [Neural Network](#neural-network)
    - [MNIST/LETTERS + Training + Result](#entrenamiento)
    - [Interface](#interfaz-grafica)

- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Usage](#usage)

- [Contact](#contact)

## About This Project

Notation:
- $\bar{X}$: Dataset input (matrix)
- $\bar{Y}$: Dataset output (matrix)

- $\bar{x}$: Input of the layer (vector)
- $\bar{y}$: Output of the layer (vector)
- $\bar{W}$: Weights of the layer (matrix)

- $\bar{a}$: Activation function (vector)

### Imputed Error (last layer)
$$ \frac{\partial{E}}{\partial{\bar{x}}} = \frac{2}{N}*(\bar{a}-\bar{y})\odot{\bar{a}'} $$

### Parameter update
$$ \frac{\partial{E}}{\partial{\bar{w}}} = \frac{\partial{E}}{\partial{\bar{x}}}\bar{x}^T $$

### Imputed Error (hidden layer)
$$ \frac{\partial{E}}{\partial{\bar{x}}} = W^t\frac{\partial{E}}{\partial{\bar{y}}} $$

## Getting Started

### Installation

## Usage


## Contact

- [José Ridruejo][email-pepe]
- [Sergio Herreros Pérez][email-gomi]


[email-gomi]: mailto:gomimaster1@gmail.com
[email-pepe]: mailto:pepert03@gmail.com
