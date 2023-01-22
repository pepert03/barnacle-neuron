[![made-with-python](https://img.shields.io/badge/Made%20with-Python-informational?style=flat-square)](https://www.python.org/)
[![made-with-pygame](https://img.shields.io/badge/Made%20with-Pygame-informational?style=flat-square)](https://www.pygame.org/news)


<div style="text-align:left"><img src="images/logo.jpg" width="100%"></div>


# The Barnacle Neuron
### Basic implementation of a Neural Network from scratch

## Table of Contents

- [About this Project](#about-this-project)
    - [Detailed Description](#detailed-description)
    - [Dataset Creation](#dataset-creation)
    - [Neural Network](#neural-network)
    - [Basic Layers](#basic-layers)
    - [MNIST/LETTERS + Training + Result](#entrenamiento)
    - [Interface](#interfaz-grafica)

- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Usage](#usage)

- [Contact](#contact)

## About This Project

### Detailed Description
We aim to create a complete project from scratch, including the creation of the dataset, the implementation and mathematical derivation of the neural network and the creation of a graphical interface to test the results.

### Dataset Creation
Tocrate the dataset, we've made a pygame interface that allows us to draw letters and numbers and save them as images. 
In order to achive translation, rotation and scaling invariance, we've used data augmentation techniques to expand the dataset.

### Neural Network
We are going to use a modular approach to create the neural network. We are going to define 2 basic objects: the Layer and the NeuralNetwork.

### Basic Layers
The Layer class will consist of 3 main methods:
- **__init__**: This method will initialize the layer, depending on the type on layer, it will initialize the weights, the activation function, the kernels, etc.
- **forward**: This method will calculate the output of the layer given an input.
- **backward**: This method will calculate the imputed error of the layer given the imputed error of the next layer. It will also update the parameters of the layer if it is a trainable layer.

This layer will be the base class for the other layers. We will define 2 basic layers for this project: the Dense layer and the Activation layer.  
<br>
Now we proceed to define the formulas needed in forward and backward propagation for each layer.  

#### Notation:
- $\bar{X}$: Dataset input (matrix)
- $\bar{Y}$: Dataset output (matrix)

- $\bar{x}$: Input of the layer (vector)
- $\bar{y}$: Output of the layer (vector)

- $\bar{W}$: Weights of the layer (matrix)
- $\bar{a}$: Activation function of the layer (vector)

<br>

We will take advantage of the modular approach of the neural network to define the formulas for each layer.

1. **Dense Layer**

    - **Forward Propagation**:
    $$ \bar{y} = \bar{W}\bar{x} $$
    - **Backward Propagation**:


#### Imputed Error (last layer)
$$ \frac{\partial{E}}{\partial{\bar{x}}} = \frac{2}{N}*(\bar{a}-\bar{y})\odot{\bar{a}'} $$

#### Parameter update (dense layer)
$$ \frac{\partial{E}}{\partial{\bar{w}}} = \frac{\partial{E}}{\partial{\bar{x}}}\bar{x}^T $$

#### Imputed Error (dense layer)
$$ \frac{\partial{E}}{\partial{\bar{x}}} = W^t\frac{\partial{E}}{\partial{\bar{y}}} $$

#### Imputed Error (activation layer)
$$ \frac{\partial{E}}{\partial{\bar{x}}} = \frac{\partial{E}}{\partial{\bar{y}}}\odot{\bar{a}'} $$

## Getting Started


### Installation

## Usage

## Contact

- [José Ridruejo][email-pepe]
- [Sergio Herreros Pérez][email-gomi]


[email-gomi]: mailto:gomimaster1@gmail.com
[email-pepe]: mailto:pepert03@gmail.com
