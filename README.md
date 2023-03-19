# DL_Assignment1
## Author : Adarsh Gupta CS22M006
## Overview
The project is to train fashion-mnist and mnist dataset, which consists of 70000 images of size 28*28.     
Using various optimizers like SGD, momentum, rmsprop, adam, nadam, nag, etc. optimizers to train the dataset.

## Folder structure
* **optimizers_wandb.ipynb:** Contains code to run sweep with various hyperparameter configurations and plot confusion matrix. This also contain implementation of all the optimizers. It also contains all the functions which are required to train the model.
* **train.py:** It contains code to train the model, as well as code to support the training of model using command line interface. Used **ArgParse** to support this feature.
* **requirements.txt:** Contains all the libraries needed to run this project.
 
## Instructions to train and evaluate various models

1. Install the required libararies using following command

`
pip install -r requirements.txt
`

2. Run the train.py using command line and pass parameters which you want to set. Passing parameters is optional, if you don't want to pass parameters then it will take default hyperparameter values to train the model.
Here is one example of the command to run train.py and train the model.

`
python train.py -wp 'Assignment 1' -we 'cs22m006' -d 'mnist' -lr 0.01 -a 'sigmoid' -w_i 'random_uniform' -o 'rmsprop' -b 32 -w_d 0 -e 20 -sz 128 -nhl 4
`

3. After running the command, it will print accuracies and loss. It will also log accuracies and loss in wandb.

4. **Pass wandb entity as your wandb username only, otherwise it will give error.**

5. Here is the list of hyperparameter that you can pass to train the model.

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["sigmoid", "tanh", "reLU"] |
