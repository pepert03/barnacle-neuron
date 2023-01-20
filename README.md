[![made-with-python](https://img.shields.io/badge/Made%20with-Python-informational?style=flat-square)](https://www.python.org/)
[![made-with-pygame](https://img.shields.io/badge/Made%20with-Pygame-informational?style=flat-square)](https://www.pygame.org/news)


<span>
    <img src="https://media.istockphoto.com/id/481744887/es/vector/bernacle-de-historieta.jpg?s=612x612&w=0&k=20&c=B6CLMDs_ELFGzMQTYQ73P9xVe_XvFIe--j27TeAeZ28=" alt="Logo" height="300px">
</span>
<span>
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/3d/Neural_network.svg" alt="Logo" height="300px">
</span>
<span>
    <img src="https://static.wikia.nocookie.net/doblaje/images/9/90/Chico_Percebe_imagen.png/revision/latest?cb=20220918180154&path-prefix=es" alt="Logo" height="300px">
</span>


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