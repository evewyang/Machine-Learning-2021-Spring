import tensorflow as tf
import argparse
import utils
import numpy as np
import pandas as pd
import matplotlib.image as img
import torch
from PIL import Image as pil
import csv

# import wandb

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--nets', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

print(args)
# wandb.init(project="conv-nets", name=args.nets.lower())

model = utils.choose_nets(args.nets)

# cifar100 = tf.keras.datasets.cifar10
#######
def to_lower(arr):
    for i in range(0,len(arr)):
        arr[i] = arr[i].lower()
    return arr
def label_to_num(label_list):
    label_dict = {}
    labels = np.unique(label_list)
    for i in range (0, len(labels)):
        label_dict[labels[i]] = i
    for j in range (0, len(label_list)):
        label_list[j] = label_dict[label_list[j]]
    return label_dict, label_list
def attach(mat1,mat2):
    mat2 = np.reshape(mat2,(1,len(mat2),len(mat2[0]),len(mat2[0][0])))
    if (type(mat1) != np.ndarray):
        mat1 = np.array(mat1)
        mat1 = mat2
    else:
        mat1 = np.append(mat1, mat2, axis=0)
    return mat1
def resize_img(input_data, path_to_find, path_to_save, size = (32,32)):
    for i in range(0, len(input_data)):
        try:
            file = pil.open(path_to_find + input_data[i])
            file = file.resize(size)
            file = file.convert("RGB")
            file = file.save(path_to_save + input_data[i])
        except:
            print("Image %s cannot be openned."%(input_data[i]))
def get_pic_matrix(input_data, filepath):
    #initialize a new array to hold vectorized pics
    input_pics = []
    #two new list to record imgs' idx where they fail to open or process
    idx_exception = []
    #transfrom the corresponding pictures into matrices, then vectors.
    #for train_pics in X_train:
    for i in range(0, len(input_data)):
        try:
            image = img.imread(filepath + input_data[i])
        except:
            #record "throw-away" pics' idx
            idx_exception.append(i)
            continue
        input_pics = attach(input_pics,image)
    return input_pics, idx_exception
# Initialization
my_data = pd.read_csv('../facial_expressions-master/data/legend.csv').values
X_data = my_data[0:500 , 1]#。。。改data_range
y_data = my_data[0:500 , 2]#。。。改data_range
y_data = to_lower(y_data)
#y_data into int labels
y_dict, y_data = label_to_num(y_data)

# Spiliting Data 67-33 ratio as said by sir
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.33,random_state=0)
#process & convert pictures into matrices
my_path = "../facial_expressions-master/resized_images/"
X_train_pics, idx_train_exception = get_pic_matrix(X_train, my_path)
X_test_pics, idx_test_exception = get_pic_matrix(X_test, my_path)
#throw away unprocessable imgs' idx for y data
y_train = np.delete(y_train,idx_train_exception,axis = 0)
y_test = np.delete(y_test,idx_test_exception,axis = 0)
#######
x_train = torch.from_numpy(X_train_pics)
x_test = torch.from_numpy(X_test_pics)
y_train = torch.from_numpy(y_train.astype(np.uint8))
y_test = torch.from_numpy(y_test.astype(np.uint8))
#########
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(100).batch(args.batch_size)#。。。改shuffle
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(args.batch_size)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(args.lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


# @tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(loss)
    train_loss(loss)
    train_accuracy(labels, predictions)


# @tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

my_file = 'results/' + args.nets + '.csv'
my_fields = ['Train Accuracy','Train Loss','Test Accuracy', 'Test Loss']
my_rows = []
test_loss_increase = 0
prev_test_loss = -1
    
for epoch in range(args.epochs):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    my_row = [float(train_accuracy.result()*100), float(train_loss.result()), float(test_accuracy.result()*100), float(test_loss.result())]
    my_rows.append(my_row)
    template = 'Epoch: [{}/{}], Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          args.epochs,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
    # wandb.log({
    #     "TrainLoss": train_loss.result(),
    #     "TestLoss": test_loss.result(),
    #     "TrainAcc": train_accuracy.result()*100,
    #     "TestAcc": test_accuracy.result()*100
    # })
    # initialize previous test loss
    if prev_test_loss < 0:
        prev_test_loss = test_loss.result()
    # apply early stopping, check:
    if test_loss.result() > prev_test_loss:
        # consecutive increase in test loss, record patience
        test_loss_increase += 1
    else:
        # test loss decreases, make count zero
        test_loss_increase = 0
    if test_loss_increase >= 3:
        # patience >= 3, break the for loop
        # stop training = early stopping
        # reset states
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        break
    # record this as prev test loss for next epoch
    prev_test_loss = test_loss.result()
    # reset states
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

# write to file
with open(my_file, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(my_fields) 
        
    # writing the data rows 
    csvwriter.writerows(my_rows)
    
    # finish and close file
    csvfile.close()


# torch.save(model.trainable_variables, "trained_models/trained_"+args.nets+".pth")
# tf.saved_model.save(model, "trained_models/trained_"+args.nets+".pth")


