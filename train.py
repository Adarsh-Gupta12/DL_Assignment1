import wandb
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
k = len(class_names)

def initializeWeightAndBias(layer_dims, init_mode):
  W = []
  bias = []
  np.random.seed(3)
  if(init_mode == "random_uniform" or init_mode == "random"):
    for layer_num in range(len(layer_dims)-1):
      W.append(np.random.uniform(-0.7, 0.7, (layer_dims[layer_num+1], layer_dims[layer_num])))
      bias.append((np.random.uniform(-0.7, 0.7, (layer_dims[layer_num+1],1))))
  elif(init_mode == "xavier"):
    for layer_num in range(len(layer_dims)-1):
      W.append(np.random.randn(layer_dims[layer_num+1],layer_dims[layer_num])*np.sqrt(2/(layer_dims[layer_num+1]+layer_dims[layer_num])))
      bias.append(np.random.randn(layer_dims[layer_num+1],1)*np.sqrt(2/(layer_dims[layer_num+1])))
  else: #if(init_mode == "random_normal"):
    for layer_num in range(len(layer_dims)-1):
      W.append(np.random.randn(layer_dims[layer_num+1], layer_dims[layer_num]))
      bias.append((np.random.randn(layer_dims[layer_num+1],1)))
  # print("inside", len(W), len(bias), len(layer_dims))
  return W, bias

def feedForward(W, bias, X, num_hidden_layers, layer_dims, activation_fun = "tanh"):
  preactivation = []
  activation = []
  activation.append(X.T)
  preactivation.append(X.T)
  for i in range(1, num_hidden_layers+1):
    preactivation.append(bias[i-1] + np.matmul(W[i-1], activation[(i-1)]))
    if(activation_fun == "sigmoid"):
      activation.append(sigmoid(preactivation[i]))
    elif(activation_fun == "tanh"):
      activation.append(tanh(preactivation[i]))
    elif(activation_fun == "reLU"):
      activation.append(reLU(preactivation[i]))
  preactivation.append(bias[-1] + np.dot(W[-1], activation[-1]))
  activation.append(softmax(preactivation[-1]))
  return activation[-1], activation, preactivation

def updateParam(W, gradientW, bias, gradientBias, learning_rate):
  for i in range(0, len(W)):
    W[i] = W[i] - learning_rate*gradientW[i]
    bias[i] = bias[i] - learning_rate*gradientBias[i]
  return W, bias

def updateParamMomentum(W, bias, gradientW, gradientBias, previous_updates_W, previous_updates_Bias, learning_rate, momentum):
  for idx in range(len(gradientW)):
    previous_updates_W[idx] = momentum*previous_updates_W[idx] + gradientW[idx]
    previous_updates_Bias[idx] = momentum*previous_updates_Bias[idx] + gradientBias[idx]
  for i in range(0, len(W)):
    W[i] = W[i] - learning_rate*gradientW[i]
    bias[i] = bias[i] - learning_rate*gradientBias[i]
  return W, bias
  

def updateParamRMS(W, gradientW, bias, gradientBias, learning_rate, v_W, v_bias, beta, epsilon):
  for idx in range(0, len(W)):
    v_W_t = beta*v_W[idx] + (1-beta)*np.multiply(gradientW[idx], gradientW[idx])
    v_bias_t = beta*v_bias[idx] + (1-beta)*np.multiply(gradientBias[idx], gradientBias[idx])
    W[idx] = W[idx] - learning_rate*gradientW[idx]/(np.sqrt(v_W_t)+epsilon)
    bias[idx] = bias[idx] - learning_rate*gradientBias[idx]/(np.sqrt(v_bias_t)+epsilon)
    v_W[idx] = v_W_t
    v_bias[idx] = v_bias_t
  return W, bias, v_W, v_bias

def updateParamAdam(W, bias, gradientW, gradientBias, v_W, v_bias, m_W, m_bias, t, learning_rate, beta1, beta2, epsilon):
  for i in range(0, len(W)):
    mdW = beta1*m_W[i] + (1-beta1)*gradientW[i]
    mdBias = beta1*m_bias[i] + (1-beta1)*gradientBias[i]
    vdW = beta2*v_W[i] + (1-beta2)*np.square(gradientW[i])
    vdBias = beta2*v_bias[i] + (1-beta2)*np.square(gradientBias[i])
    m_w_hat = mdW/(1.0 - beta1**t)
    v_w_hat = vdW/(1.0 - beta2**t)
    m_bias_hat = mdBias/(1.0 - beta1**t)
    v_bias_hat = vdBias/(1.0 - beta2**t)

    W[i] = W[i] - (learning_rate * m_w_hat)/np.sqrt(v_w_hat + epsilon)
    bias[i] = bias[i] - (learning_rate * m_bias_hat)/np.sqrt(v_bias_hat + epsilon)

    v_W[i] = vdW
    m_W[i] = mdW
    v_bias[i] = vdBias
    m_bias[i] = mdBias

    return W, bias, v_W, v_bias, m_W, m_bias

def sigmoid(X):
  return 1.0/(1.+np.exp(-X))

def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

def reLU(x):
  return np.maximum(0,x)

def reLU_derivative(x):
  return 1*(x>0) 

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return (1 - (np.tanh(x)**2))

def softmax(a):
  return np.exp(a)/np.sum(np.exp(a), axis=0)

def softmax_derivative(a):
  return softmax(a)*(1-softmax(a))

def backward_propogation(y_one_hot, x, y, W, bias, activation, preactivation, num_hidden_layers, batch_size, activation_fun, weight_decay, loss_fun):
  L = num_hidden_layers+1
  gradientPreactivation = []
  if(loss_fun == "cross_entropy"):
    gradientPreactivation.append(activation[L]-y_one_hot)
  else:
    gradientPreactivation.append((activation[L]-y_one_hot) * softmax_derivative(preactivation[L]))
  gradientWeight = []
  gradientBias = []
  for k in range(L, 0, -1):
    gradientWeight.append(np.matmul(gradientPreactivation[-1], activation[k-1].T)/batch_size + (weight_decay*W[k-1])/batch_size)
    gradientBias.append(np.sum(gradientPreactivation[-1], axis=1, keepdims=True)/batch_size)
    if k==1:
      break
    if(activation_fun == "sigmoid"):
      gradientPreactivation.append(np.multiply(np.matmul(W[k-1].T, gradientPreactivation[-1]), sigmoid_derivative(preactivation[k-1])))
    elif(activation_fun == "tanh"):
      gradientPreactivation.append(np.multiply(np.matmul(W[k-1].T, gradientPreactivation[-1]), tanh_derivative(preactivation[k-1])))
    if(activation_fun == "reLU"):
      gradientPreactivation.append(np.multiply(np.matmul(W[k-1].T, gradientPreactivation[-1]), reLU_derivative(preactivation[k-1])))
  return gradientWeight[::-1], gradientBias[::-1]

def cross_entropy(y, y_hat, W, weight_decay):
  loss = 0
  for i in range(len(y)):
    loss += -1.0*np.sum(y[i]*np.log(y_hat[i]))
  #L2 regularization
  acc = 0
  for i in range(len(W)):
    acc += np.sum(W[i]**2)
  loss += weight_decay*acc
  return loss

def mean_squared_error(y, y_hat, W, weight_decay):
  loss = 0.5 * np.sum((y-y_hat)**2)
  #L2 regularizaation
  acc = 0
  for i in range(len(W)):
    acc += np.sum(W[i]**2)
  loss += weight_decay*acc
  return loss

def calculate_accuracy_and_loss(W, bias, X, y, num_hidden_layers, layer_dims, activation_fun, weight_decay, loss_function, y_one_hot):
  hL, _, _ = feedForward(W, bias, X, num_hidden_layers, layer_dims, activation_fun)
  predictions = np.argmax(hL, axis = 0)
  acc = accuracy_score(y, predictions)*100
  if(loss_function == "cross_entropy"):
    loss = cross_entropy(y_one_hot, hL, W, weight_decay)
  else:
    loss = mean_squared_error(y_one_hot, hL, W, weight_decay)
  return acc, loss

def generate_one_hot(n, true_label):
  y_one_hot = np.zeros((10, n))
  for i in range(n):
    y_one_hot[true_label[i]][i] = 1
  return y_one_hot

def optimizers(trainX, trainy, validationX, validationy, testX, testy, wandb_project, wandb_entity, num_hidden_layers, neurons_in_each_layer, epochs, learning_rate, batch_size, init_mode, activation_fun, loss_function, optimizer, momentum, beta, beta1, beta2, weight_decay, epsilon):
  wandb.init(
      project=wandb_project,
      entity=wandb_entity,
      name="Assignment1_optimizer"
  )
  num_images = len(trainX)
  layer_dims = [trainX.shape[1]]
  for i in range(num_hidden_layers):
    layer_dims.append(neurons_in_each_layer)
  layer_dims.append(k)
  W, bias = initializeWeightAndBias(layer_dims, init_mode)

  y_one_hot, y_one_hot_val = generate_one_hot(num_images, trainy), generate_one_hot(len(validationy), validationy)
  
  v_W = [0]*(num_hidden_layers+1)
  v_bias, m_W, m_bias, gradientW, gradientBias, look_ahead_W, look_ahead_bias, previous_updates_W, previous_updates_Bias = v_W.copy(), v_W.copy(), v_W.copy(), v_W.copy(), v_W.copy(), v_W.copy(), v_W.copy(), v_W.copy(), v_W.copy()
  t = 1 #for adam
  run_name = "lr_{}_ac_{}_in_{}_op_{}_bs_{}_L2_{}_ep_{}_nn_{}_nh_{}".format(learning_rate, activation_fun, init_mode, optimizer, batch_size, weight_decay, epochs, neurons_in_each_layer, num_hidden_layers)
  y_pred = []
  for iterationNumber in range(epochs):
    loss=0
    val_loss = 0
    batch_count = batch_size
    for i in range(0, num_images, batch_size):
      if(i+batch_size >= num_images):
        batch_count = num_images-i

      if(optimizer == "nag"):
        for idx in range(len(W)):
          look_ahead_W[idx] = W[idx] - momentum * gradientW[idx]
          look_ahead_bias[idx] = bias[idx] - momentum * gradientBias[idx]

        hL, activation, preactivation = feedForward(look_ahead_W, look_ahead_bias, trainX[i:i+batch_count], num_hidden_layers, layer_dims, activation_fun)
        gradientW, gradientBias = backward_propogation(y_one_hot[:,i:i+batch_count], trainX[i:i+batch_count], trainy[i:i+batch_count], look_ahead_W, look_ahead_bias, activation, preactivation, num_hidden_layers, batch_size, activation_fun, weight_decay, loss_function)
        #update gradient
        W, bias = updateParam(W, gradientW, bias, gradientBias, learning_rate)

      elif(optimizer == "nadam"):
        for idx in range(len(W)):
          look_ahead_W[idx] = W[idx] - momentum * gradientW[idx]
          look_ahead_bias[idx] = bias[idx] - momentum * gradientBias[idx]

        hL, activation, preactivation = feedForward(look_ahead_W, look_ahead_bias, trainX[i:i+batch_count], num_hidden_layers, layer_dims, activation_fun)
        gradientW, gradientBias = backward_propogation(y_one_hot[:,i:i+batch_count], trainX[i:i+batch_count], trainy[i:i+batch_count], look_ahead_W, look_ahead_bias, activation, preactivation, num_hidden_layers, batch_size, activation_fun, weight_decay, loss_function)
        W, bias, v_W, v_bias, m_W, m_bias = updateParamAdam(W, bias, gradientW, gradientBias, v_W, v_bias, m_W, m_bias, t, learning_rate, beta1, beta2, epsilon)
        t += 1

      else:
        hL, activation, preactivation = feedForward(W, bias, trainX[i:i+batch_count], num_hidden_layers, layer_dims, activation_fun)

        gradientW, gradientBias = backward_propogation(y_one_hot[:,i:i+batch_count], trainX[i:i+batch_count], trainy[i:i+batch_count], W, bias, activation, preactivation, num_hidden_layers, batch_size, activation_fun, weight_decay, loss_function)
  
        if(optimizer == "sgd"):
          W, bias = updateParam(W, gradientW, bias, gradientBias, learning_rate)

        elif(optimizer == "momentum"):
          W, bias = updateParamMomentum(W, bias, gradientW, gradientBias, previous_updates_W, previous_updates_Bias, learning_rate, momentum)
        
        elif(optimizer == "rmsprop"):
          W, bias, v_W, v_bias = updateParamRMS(W, gradientW, bias, gradientBias, learning_rate, v_W, v_bias, beta, epsilon)

        elif(optimizer == "adam"):
          W, bias, v_W, v_bias, m_W, m_bias = updateParamAdam(W, bias, gradientW, gradientBias, v_W, v_bias, m_W, m_bias, t, learning_rate, beta1, beta2, epsilon)
          t += 1
      if(iterationNumber==epochs-1):
        for j in range(i, i+batch_count):
          y_pred.append(np.argmax(hL[:,(j-i)]))
    train_acc, loss = calculate_accuracy_and_loss(W, bias, trainX, trainy, num_hidden_layers, layer_dims, activation_fun, weight_decay, loss_function, y_one_hot)
    val_acc, val_loss = calculate_accuracy_and_loss(W, bias, validationX, validationy, num_hidden_layers, layer_dims, activation_fun, weight_decay, loss_function, y_one_hot_val)
    print("training_accuracy:", train_acc, "validation_accuracy:", val_acc, "training_loss:", loss/(len(trainX)), "validation loss:", val_loss/len(validationX), "epoch:", iterationNumber)
    wandb.log({"training_accuracy": train_acc, "validation_accuracy": val_acc, "training_loss": loss/(len(trainX)), "validation loss": val_loss/len(validationX), 'epoch': iterationNumber})
  wandb.run.name = run_name
  wandb.run.save()
  wandb.run.finish()
  calculateTestAccuracy(testX, testy, layer_dims, num_hidden_layers, neurons_in_each_layer, batch_size, W, bias, activation_fun)

def preProcessing(params):
  dataset = params.dataset
  if dataset == "fashion_mnist":
    (x_train, y_train), (testX, testy) = fashion_mnist.load_data()
  elif dataset == "mnist":
    (x_train, y_train), (testX, testy) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
  x_train = x_train/255.0
  testX = testX.reshape(testX.shape[0], testX.shape[1]*testX.shape[2])
  testX = testX/255.0
  trainX, validationX, trainy, validationy = train_test_split(x_train, y_train, random_state=104, test_size=0.1, shuffle=True)
  optimizers(trainX, trainy, validationX, validationy, testX, testy, params.wandb_project, params.wandb_entity, params.num_layers, params.hidden_size, params.epochs, params.learning_rate, params.batch_size, params.weight_init, params.activation, params.loss, params.optimizer, params.momentum, params.beta, params.beta1, params.beta2, params.weight_decay, params.epsilon)


def calculateTestAccuracy(testX, testy, layer_dims, num_hidden_layers, neurons_in_each_layer, batch_size, W, bias, activation_fun):
  batch_count = batch_size
  count = 0
  for i in range(0, len(testX), batch_size):
    if(i+batch_size>len(testX)):
      batch_count = len(testX)-i-1
    hL, activation, preactivation = feedForward(W, bias, testX[i:i+batch_count], num_hidden_layers, layer_dims, activation_fun)
    for j in range(i, i+batch_count):
      if(np.argmax(hL[:,(j-i)]) == testy[j]):
        count+=1
  print("Accuracy on test data", (100.0*count)/len(testX))

parser = argparse.ArgumentParser(description='calculate accuracy and loss for given hyperparameters')
parser.add_argument('-wp', '--wandb_project', type=str, help='wandb project name', default='Assignment 1')
parser.add_argument('-we', '--wandb_entity', type=str, help='wandb entity', default='cs22m006')
parser.add_argument('-d', '--dataset', type=str, help='dataset', default='fashion_mnist')
parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=20)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=32)
parser.add_argument('-l', '--loss', type=str, help='type of loss function', default='cross_entropy')
parser.add_argument('-o', '--optimizer', type=str, help='optimizer to be used', default='sgd')
parser.add_argument('-lr', '--learning_rate', type=int, help='learning rate', default=0.1)
parser.add_argument('-m', '--momentum', type=int, help='Momentum used by momentum and nag optimizers', default=0.5)
parser.add_argument('-beta', '--beta', type=int, help='Beta used by rmsprop optimizer', default=0.5)
parser.add_argument('-beta1', '--beta1', type=int, help='used by adam and nadam optimizers', default=0.9)
parser.add_argument('-beta2', '--beta2', type=int, help='used by adam and nadam optimizers', default=0.999)
parser.add_argument('-eps', '--epsilon', type=int, help='Epsilon used by optimizers', default=0.000001)
parser.add_argument('-w_d', '--weight_decay', type=int, help='Weight decay used by optimizers', default=0.05)
parser.add_argument('-w_i', '--weight_init', type=str, help='initialization mode', default='random_normal')
parser.add_argument('-nhl', '--num_layers', type=int, help='Number of hidden layers used in feedforward neural network', default=4)
parser.add_argument('-sz', '--hidden_size', type=int, help='Number of hidden neurons in a feedforward layer', default=128)
parser.add_argument('-a', '--activation', type=str, help='activation function', default='tanh')
params = parser.parse_args()
if __name__ == '__main__':
  preProcessing(params)