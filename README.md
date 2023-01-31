[![made-with-python](https://img.shields.io/badge/Made%20with-Python-informational?style=flat-square )](https://www.python.org/)
[![made-with-pygame](https://img.shields.io/badge/Made%20with-Pygame-informational?style=flat-square )](https://www.pygame.org/news)
  
  
<div style="text-align:left"><img src="meta/logo.jpg" width="100%"></div>
  
  
#  The Barnacle Neuron
  
###  Basic implementation of a Neural Network from scratch
  
  
##  Table of Contents
  
  
- [About this Project](#about-this-project )
    - [Detailed Description](#detailed-description )
    - [Dataset Creation](#dataset-creation )
    - [Neural Network](#neural-network )
    - [Basic Layers](#basic-layers )
    - [MNIST/LETTERS + Training + Result](#entrenamiento )
    - [Interface](#interfaz-grafica )
  
  
- [Getting Started](#getting-started )
    - [Installation](#installation )
    - [Usage](#usage )
  
- [Contact](#contact )
  
##  About This Project
  
  
###  Detailed Description
  
We aim to create a complete project from scratch, including the creation of the dataset, the implementation and mathematical derivation of the neural network and the creation of a graphical interface to test the results.
  
###  Dataset Creation
  
To create the dataset, we've made a pygame interface that allows us to draw letters and numbers and save them as images. 
In order to achive translation, rotation and scaling invariance, we've used data augmentation techniques to expand the dataset.
  
###  Neural Network
  
We are going to use a modular approach to create the NeuralNetwork class. In our implementation, every single operation in the network will be its own Layer. This will simplify the implementation as well as the mathematical derivation of the formulas. Each layer will be able to do two main operations:  
  
1. **Forward Propagation**: Receives an input from the previous layer, performs the desired operation on it and returs the ouput.  
The first layer will recieve the input from the dataset or from a user provided input.
  
2. **Backward Propagation**: Recieves the imputed error of the following layer. That is, the error with respect to its input.  
The exception is the last layer, which will be a Loss Layer. It will calculate the imputed error using the loss function and the output of the network.
  
###  Basic Layers
  
The Layer class will be the base class for the other layers. We will define 3 type of layers for this project: Dense, Activation and Loss layers.
  
Now we proceed to define the formulas needed in forward and backward propagation for each layer.  
  
####  Notation:
  
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{X}"/>: Dataset input (matrix)
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{Y}"/>: Dataset output (matrix)
  
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;textbf{x}"/>: Input of the network (vector)
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;textbf{y}"/>: Output of the network (vector)
  
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{x}"/>: Input of the layer (vector)
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{y}"/>: Output of the layer (vector)
  
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{W}"/>: Weights of the Dense layer (matrix)
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{b}"/>: Bias of the Dense layer (vector)
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{a}(&#x5C;bar{x})"/>: Activation function of the layer (vector)
  
- <img src="https://latex.codecogs.com/gif.latex?n"/>: length of <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{x}"/>
- <img src="https://latex.codecogs.com/gif.latex?m"/>: length of <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{y}"/>
  
<br>
  
We will take advantage of the modular approach of the neural network to define the formulas for each layer. 
  
Using the chain rule, we will be able to derive the formulas for each layer in terms of the formulas of the previous layer.
  
1. **Dense Layer**
  
    - **Forward Propagation**:
  
    <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{equation}%20%20%20%20y_i%20=%20%20%20%20%20w_{i1}x_1%20+%20w_{i2}x_2%20+%20&#x5C;cdots%20+%20w_{in}x_n%20+%20b_i%20=%20%20%20%20%20&#x5C;sum_{j}w_{ij}x_j%20+%20b_i%20%20%20%20&#x5C;end{equation}"/></p>  
  
  
    <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{equation}%20%20%20%20&#x5C;bar{y}%20=%20&#x5C;bar{W}&#x5C;bar{x}%20+%20&#x5C;bar{b}%20%20%20%20&#x5C;end{equation}"/></p>  
  
  
    - **Backward Propagation**:   
        - **Parameter update**:
            For each weight <img src="https://latex.codecogs.com/gif.latex?w_{ij}"/>, we have to calculate the partial derivative of the error with respect to that weight. Using the chain rule, we have:
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{w_{ij}}}%20=%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{y_i}}}%20&#x5C;frac{&#x5C;partial{&#x5C;bar{y_i}}}{&#x5C;partial{w_{ij}}}%20=%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{y_i}}}&#x5C;bar{x_j}"/></p>  
  
            And for each bias <img src="https://latex.codecogs.com/gif.latex?b_i"/>:
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{b_i}}%20=%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{y_i}}}&#x5C;frac{&#x5C;partial{&#x5C;bar{y_i}}}{&#x5C;partial{b_i}}%20=%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{y_i}}}&#x5C;cdot{1}"/></p>  
  
            Extrapolating to matrix notation:
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{W}}}%20=%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{y}}}&#x5C;bar{x}^T"/></p>  
  
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{b}}}%20=%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{y}}}"/></p>  
  
            Then, we update the weights and the biases using the gradient descent algorithm:
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{W}%20=%20&#x5C;bar{W}%20-%20&#x5C;alpha&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{W}}}"/></p>  
  
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{b}%20=%20&#x5C;bar{b}%20-%20&#x5C;alpha&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{b}}}"/></p>  
  
            Where <img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha"/> is the learning rate.
  
  
        - **Imputed Error**:  
            This time, since <img src="https://latex.codecogs.com/gif.latex?x_i"/> is distributed in all the neurons of the next layer, we have to sum all the partial derivatives of the next layer with respect to <img src="https://latex.codecogs.com/gif.latex?x_i"/>:
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{x_i}}%20=%20&#x5C;sum_{j}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_j}}&#x5C;frac{&#x5C;partial{y_j}}{&#x5C;partial{x_i}}%20=%20&#x5C;sum_{j}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_j}}w_{ji}"/></p>  
  
            Extrapolating to matrix notation:  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{x}}}%20=%20&#x5C;begin{bmatrix}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{x_1}}%20&#x5C;&#x5C;&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{x_2}}%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{x_n}}&#x5C;end{bmatrix}%20=&#x5C;begin{bmatrix}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_1}}{w_{11}}+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_1}}{w_{21}}+&#x5C;cdots+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_1}}{w_{m1}}%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_2}}{w_{12}}+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_2}}{w_{22}}+&#x5C;cdots+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_2}}{w_{m2}}%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_m}}{w_{1n}}+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_m}}{w_{2n}}+&#x5C;cdots+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_m}}{w_{mn}}&#x5C;end{bmatrix}%20="/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{bmatrix}w_{11}&amp;w_{21}&amp;&#x5C;cdots&amp;w_{m1}%20&#x5C;&#x5C;%20w_{12}&amp;w_{22}&amp;&#x5C;cdots&amp;w_{m2}%20&#x5C;&#x5C;&#x5C;vdots&amp;&#x5C;vdots&amp;&#x5C;ddots&amp;&#x5C;vdots%20&#x5C;&#x5C;w_{1n}&amp;w_{2n}&amp;&#x5C;cdots&amp;w_{mn}&#x5C;end{bmatrix}&#x5C;begin{bmatrix}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_1}}%20&#x5C;&#x5C;&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_2}}%20&#x5C;&#x5C;&#x5C;vdots%20&#x5C;&#x5C;&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_m}}&#x5C;end{bmatrix}%20=%20W^t&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{y}}}"/></p>  
  
  
2. **Activation Layer**
    - **Activation Functions**: Here we'll use two activation functions:  
  
        - **Tanh**: 
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?a_i(&#x5C;bar{x})%20=%20&#x5C;frac{e^{x_i}-e^{-x_i}}{e^{x_i}+e^{-x_i}}"/></p>  
  
  
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{a_i(&#x5C;bar{x})}}{&#x5C;partial{x_j}}%20=%20&#x5C;left&#x5C;lbrace%201%20-%20a_i^2(&#x5C;bar{x})%20&#x5C;%20&#x5C;%20&#x5C;%20if%20&#x5C;%20&#x5C;%20%20i==j%20&#x5C;atop%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%200%20%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20otherwise%20%20&#x5C;right."/></p>  
  
  
        - **Softmax**: 
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?a_i(&#x5C;bar{x})%20=%20&#x5C;frac{e^{x_i}}{&#x5C;sum_{j}e^{x_j}}"/></p>  
  
  
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{a_i(&#x5C;bar{x})}}{&#x5C;partial{x_j}}%20=%20&#x5C;left&#x5C;lbrace%20a_i(&#x5C;bar{x})(1-a_i(&#x5C;bar{x}))%20&#x5C;%20&#x5C;%20&#x5C;%20if%20&#x5C;%20&#x5C;%20%20i==j%20&#x5C;atop%20&#x5C;%20&#x5C;%20&#x5C;%20-a_i(&#x5C;bar{x})a_j(&#x5C;bar{x})%20%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20&#x5C;%20otherwise%20%20&#x5C;right.%20=%20a_i(&#x5C;bar{x})(1%20&#x5C;lbrace%20i==j%20&#x5C;rbrace%20-a_j(&#x5C;bar{x}))"/></p>  
  
  
  
    - **Forward Propagation**:  
        <p align="center"><img src="https://latex.codecogs.com/gif.latex?y_i%20=%20a_i(&#x5C;bar{x})"/></p>  
  
        <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{y}%20=%20&#x5C;bar{a}(&#x5C;bar{x})"/></p>  
  
  
    - **Backward Propagation**:   
        - **Imputed Error**:
            For each neuron <img src="https://latex.codecogs.com/gif.latex?x_i"/>, we have to calculate the partial derivative of the error with respect to that neuron. Using the chain rule, we have:  
  
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{x_i}}%20=%20&#x5C;sum_{j}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_j}}&#x5C;frac{&#x5C;partial{y_j}}{&#x5C;partial{x_i}}%20=%20&#x5C;sum_{j}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_j}}&#x5C;frac{&#x5C;partial{a_j(&#x5C;bar{x})}}{&#x5C;partial{x_i}}"/></p>  
  
  
            Extrapolating to matrix notation:  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{x}}}%20=%20&#x5C;begin{bmatrix}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{x_1}}%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{x_2}}%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{x_n}}&#x5C;end{bmatrix}%20=&#x5C;begin{bmatrix}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_1}}&#x5C;frac{&#x5C;partial{a_1(&#x5C;bar{x})}}{&#x5C;partial{x_1}}+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_2}}&#x5C;frac{&#x5C;partial{a_2(&#x5C;bar{x})}}{&#x5C;partial{x_1}}+&#x5C;cdots+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_m}}&#x5C;frac{&#x5C;partial{a_m(&#x5C;bar{x})}}{&#x5C;partial{x_1}}%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_1}}&#x5C;frac{&#x5C;partial{a_1(&#x5C;bar{x})}}{&#x5C;partial{x_2}}+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_2}}&#x5C;frac{&#x5C;partial{a_2(&#x5C;bar{x})}}{&#x5C;partial{x_2}}+&#x5C;cdots+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_m}}&#x5C;frac{&#x5C;partial{a_m(&#x5C;bar{x})}}{&#x5C;partial{x_2}}%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_1}}&#x5C;frac{&#x5C;partial{a_1(&#x5C;bar{x})}}{&#x5C;partial{x_n}}+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_2}}&#x5C;frac{&#x5C;partial{a_2(&#x5C;bar{x})}}{&#x5C;partial{x_n}}+&#x5C;cdots+&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_m}}&#x5C;frac{&#x5C;partial%20a_m(&#x5C;bar%20x)%20}{&#x5C;partial%20x_n}&#x5C;end{bmatrix}"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?=%20&#x5C;begin{bmatrix}&#x5C;frac{&#x5C;partial{a_1(&#x5C;bar{x})}}{&#x5C;partial{x_1}}%20&amp;%20&#x5C;frac{&#x5C;partial{a_1(&#x5C;bar{x})}}{&#x5C;partial{x_2}}%20&amp;%20&#x5C;cdots%20&amp;%20&#x5C;frac{&#x5C;partial{a_1(&#x5C;bar{x})}}{&#x5C;partial{x_n}}%20&#x5C;&#x5C;&#x5C;frac{&#x5C;partial{a_2(&#x5C;bar{x})}}{&#x5C;partial{x_1}}%20&amp;%20&#x5C;frac{&#x5C;partial{a_2(&#x5C;bar{x})}}{&#x5C;partial{x_2}}%20&amp;%20&#x5C;cdots%20&amp;%20&#x5C;frac{&#x5C;partial{a_2(&#x5C;bar{x})}}{&#x5C;partial{x_n}}%20&#x5C;&#x5C;&#x5C;vdots%20&amp;%20&#x5C;vdots%20&amp;%20&#x5C;ddots%20&amp;%20&#x5C;vdots%20&#x5C;&#x5C;&#x5C;frac{&#x5C;partial{a_m(&#x5C;bar{x})}}{&#x5C;partial{x_1}}%20&amp;%20&#x5C;frac{&#x5C;partial{a_m(&#x5C;bar{x})}}{&#x5C;partial{x_2}}%20&amp;%20&#x5C;cdots%20&amp;%20&#x5C;frac{&#x5C;partial{a_m(&#x5C;bar{x})}}{&#x5C;partial{x_n}}&#x5C;end{bmatrix}&#x5C;begin{bmatrix}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_1}}%20&#x5C;&#x5C;&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_2}}%20&#x5C;&#x5C;&#x5C;vdots%20&#x5C;&#x5C;&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{y_m}}&#x5C;end{bmatrix}%20=J_a(&#x5C;bar{x})^T&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{y}}}"/></p>  
  
  
3. **Loss Layer**  
    Using the [error](#Neural-Network ) defined above, we can calculate the error of the network.
    - **Backward Propagation**:   
        - **Imputed Error**: 
            <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E(&#x5C;textbf{y},&#x5C;bar{x})}}{&#x5C;partial{x_i}}%20=%20-%20&#x5C;sum_j%20&#x5C;textbf{y}_j%20&#x5C;frac{&#x5C;partial{&#x5C;log{x_j}}}{&#x5C;partial{x_i}}%20=%20-%20&#x5C;frac{&#x5C;textbf{y}_i}{x_i}"/></p>  
  
            , where <img src="https://latex.codecogs.com/gif.latex?&#x5C;textbf{y}"/> is the real output of the network and <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{x}"/> is the output of the last layer of the network, i.e. the predicted output.  
  
    We can check that this works, calculating what would be the <img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{z}}}"/>, where <img src="https://latex.codecogs.com/gif.latex?z"/> is the input of the last activation layer, in our case, a softmax layer. For simplicity, lets call <img src="https://latex.codecogs.com/gif.latex?&#x5C;bar%20s%20(&#x5C;bar%20z)"/> the softmax function.
    <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{z_i}}%20=%20&#x5C;sum_j%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{s_j}}&#x5C;frac{&#x5C;partial{s_j}}{&#x5C;partial{z_i}}%20=%20&#x5C;sum_j%20&#x5C;left(-&#x5C;frac{&#x5C;textbf{y}_j}{s_j}&#x5C;right)(s_j(1&#x5C;lbrace%20i==j&#x5C;rbrace%20-s_i))"/></p>  
  
    <p align="center"><img src="https://latex.codecogs.com/gif.latex?=%20&#x5C;sum_j%20-&#x5C;textbf{y}_j(1%20&#x5C;lbrace%20i==j%20&#x5C;rbrace%20-s_i)%20=%20-&#x5C;textbf{y}_i%20+%20&#x5C;sum_j%20&#x5C;textbf{y}_j%20s_i%20=%20-&#x5C;textbf{y}_i%20+%20&#x5C;textbf{y}_i%20s_i%20=%20s_i%20-%20&#x5C;textbf{y}_i"/></p>  
  
  
    Vectorizing this, we get:  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{&#x5C;bar{z}}}%20=%20&#x5C;begin{bmatrix}&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{z_1}}%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{z_2}}%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20&#x5C;frac{&#x5C;partial{E}}{&#x5C;partial{z_n}}&#x5C;end{bmatrix}%20=%20&#x5C;begin{bmatrix}s_1%20-%20&#x5C;textbf{y}_1%20&#x5C;&#x5C;%20s_2%20-%20&#x5C;textbf{y}_2%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20s_n%20-%20&#x5C;textbf{y}_n&#x5C;end{bmatrix}%20=%20&#x5C;bar{s}%20-%20&#x5C;textbf{y}"/></p>  
  
  
##  Getting Started
  
Use the following instructions to get a copy of the project up and running on your local machine.
  
  
###  Installation
  
1. Install the required libraries
```
pip install -r requirements.txt
```
2. Clone the repository
```	
git clone https://github.com/pepert03/barnacle-neuron.git
```
  
###  Usage
  
  
 - **Dataset Creation**
    1. Run the dataset_ui.py file
    2. Draw a letter or a number
    3. Press the letter or number key to save the image
    4. Repeat the process for all the letters and numbers
    at least 100 images per letter/number.
    5. Press the escape key to exit the program
    6. Run the build_dataset.py file to create the dataset
    feedable to the neural network
    7. (Optional) Run the data_augmentation.py file to
    create more images for the dataset.
  
    <br>
- **Number / Letters Classification**
    * **GUI**
        1. Run the prediction_ui.py file
        2. Draw a letter or a number
        3. Number / letter will be highlited on the screen
        4. Press 'enter' to clear the screen and draw another
        5. Press the escape key to exit the program
    * **Console**
        1. Run the script predict.py. You can give a path to an image (png or jpg) or a path to a dataset (csv).  
        Arguments:
            - -i, --image: Path to the image to predict
            - -d, --dataset: Path to the dataset to predict
            - -m, --model: Path to the model to use
            - -o, --output: Path to the output file
            - -v, --verbose: Verbose mode  
  
  
##  Contact
  
  
- [José Ridruejo Tuñón][email-pepe]
- [Sergio Herreros Pérez][email-gomi]
  
  
[email-gomi]: mailto:gomimaster1@gmail.com
[email-pepe]: mailto:pepert03@gmail.com
  