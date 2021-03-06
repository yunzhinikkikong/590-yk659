
#MODIFIED FROM CHOLLETT P120 
from keras import layers 
from keras import models
from keras.utils import to_categorical
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from keras import optimizers


#### Organize all hyper-param in one place 
##### Uncomment to choose dataset

dataset="MNIST"
#dataset="MNIST_FASHION"
#dataset="CIFAR-10"

model_type = "CNN"
#model_type = "DFF_ANN"
data_augmentation = True

# all hyper param

#HIDDEN LAYER PARAM
N_HIDDEN    =   3          #NUMBER OF HIDDLE LAYERS
N_NODES  	=	64          #NODES PER HIDDEN LAYER
ACT_TYPE    =   'relu'  
LR=0.001
OUTPUT_ACTIVATION = 'softmax'
OPTIMIZER	=	'rmsprop'
LOSS = 'categorical_crossentropy'
METRICS = ['acc']

# 10 probability scores
output_size = 10
# for CNN
filter_size = (3,3)
pooling_size = (2,2)
last_dense_node = 64
# data augmentation thresold
dropout = 0.5
epochs=100
# batch size will be include after laoding dataset

####################################################
#### Load MNIST, MNIST Fashion, CIFAR-10 datasets
####################################################



if(dataset=="MNIST"): 
    ### MNIST
    from keras.datasets import mnist
    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    ### Reformat the data
    if(model_type=="DFF_ANN"):
        train_images = train_images.reshape((60000, 28*28))
        test_images = test_images.reshape((10000, 28*28))
    if(model_type=="CNN"):
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))     

if(dataset=="MNIST_FASHION"): 
    ### MNIST Fashion
    from keras.datasets import fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    ### Reformat the data
    if(model_type=="CNN"):
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
    if(model_type=="DFF_ANN"):
        train_images = train_images.reshape((60000, 28*28))
        test_images = test_images.reshape((10000, 28*28))
    
if(dataset=="CIFAR-10"):   
    ### CIFAR-10
    from keras.datasets import cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    ### Reformat the data
    if(model_type=="CNN"):
        train_images = train_images.reshape((50000, 32,32,3))
        test_images = test_images.reshape((10000, 32, 32, 3))
    if(model_type=="DFF_ANN"):
        train_images = train_images.reshape((50000, 32*32))
        test_images = test_images.reshape((10000, 32*32))
    
batch_size=int(0.05*train_images.shape[0])




#NORMALIZE
train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255  






#### Add function to visualize a random (or specified) image in the dataset

def plot_image(number):

    image=train_images[number]
    from skimage.transform import rescale, resize, downscale_local_mean
    image = resize(image, (10, 10), anti_aliasing=True)
    #print((255*image).astype(int))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

# pick any image
plot_image(3)



#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)


####################################################
#### Do 80-20 split of the ???training??? data into (train/validation)
######################################################
f_train=0.8; f_val=0.2

if(f_train+f_val != 1.0):
	raise ValueError("f_train+f_val MUST EQUAL 1");

#PARTITION DATA
rand_indices = np.random.permutation(train_images.shape[0])
CUT1=int(f_train*train_images.shape[0]); 
train_idx, val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
print('------PARTITION INFO---------')
print("train_idx shape:",train_idx.shape)
print("val_idx shape:"  ,val_idx.shape)
print(train_images.shape[0])


train_image=train_images[train_idx]
val_image=train_images[val_idx]


train_label=train_labels[train_idx]
val_label=train_labels[val_idx]


#-------------------------------------
#BUILD MODEL 
#-------------------------------------


#BUILD LAYER ARRAYS 
ACTIVATIONS=[]; LAYERS=[]   
for i in range(0,N_HIDDEN):
    LAYERS.append(N_NODES)
    ACTIVATIONS.append(ACT_TYPE)

print("LAYERS:",LAYERS)
print("ACTIVATIONS:", ACTIVATIONS)

if(model_type=="DFF_ANN"):
    model = models.Sequential()
    #HIDDEN LAYERS
    model.add(layers.Dense(LAYERS[0], activation=ACTIVATIONS[0], input_shape=(train_images.shape[1],)))
    for i in range(1,len(LAYERS)):
        model.add(layers.Dense(LAYERS[i], activation=ACTIVATIONS[i]))
    #OUTPUT LAYER
    #SOFTMAX  --> 10 probability scores 
    model.add(layers.Dense(output_size, activation=OUTPUT_ACTIVATION))
    

    


if(model_type=="CNN"):

    model = models.Sequential()
    model.add(layers.Conv2D(LAYERS[0], filter_size, activation=ACTIVATIONS[0], input_shape=(train_images.shape[1], train_images.shape[2], train_images.shape[3])))
    model.add(layers.MaxPooling2D(pooling_size))
    for i in range(1,len(LAYERS)):
        model.add(layers.Conv2D(LAYERS[i], filter_size, activation=ACTIVATIONS[i])) 
        model.add(layers.MaxPooling2D(pooling_size))
    model.add(layers.Flatten())
    if(data_augmentation==True):   
        # a Dropout layer
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(last_dense_node, activation=ACT_TYPE))
    #OUTPUT LAYER
    #SOFTMAX  --> 10 probability scores 
    model.add(layers.Dense(output_size, activation=OUTPUT_ACTIVATION))
    

#-------------------------------------y
#COMPILE AND TRAIN MODEL
#-------------------------------------


#COMPILE
   
if(OPTIMIZER=='rmsprop' and LR!=0):
        opt = optimizers.RMSprop(learning_rate=LR)
else:
        opt = OPTIMIZER 
model.compile(optimizer=opt, 
              loss=LOSS, 
              metrics=METRICS
                 )
history = model.fit(
        train_image,
        train_label,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_image,val_label)
        )


#-------------------------------------
# Include a method to save model and hyper parameters
#-------------------------------------

# Save the model 
model.save('mnsit.h5')

class hyper_param:

    #INITIALIZE
	def __init__(self):
            self.dataset=dataset
            self.model_type = model_type
            self.data_augmentation = data_augmentation
            self.N_HIDDEN    =   N_HIDDEN         
            self.N_NODES  	=	N_NODES        
            self.ACT_TYPE    =   ACT_TYPE  
            self.LR=LR
            self.OUTPUT_ACTIVATION = OUTPUT_ACTIVATION 
            self.OPTIMIZER	=	OPTIMIZER
            self.LOSS = LOSS
            self.METRICS =  METRICS
            self.output_size = output_size 
            self.filter_size = filter_size
            self.pooling_size = pooling_size
            self.last_dense_node = last_dense_node
            self.dropout = dropout
            self.epochs= epochs
            self.batch_size = batch_size


##################################################
#### Include a training/validation history plot
###################################################
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#-------------------------------------
#Print (train/test/val) metrics
#-------------------------------------
train_loss, train_acc = model.evaluate(train_image, train_label, batch_size=batch_size)
val_loss, val_acc = model.evaluate(val_image, val_label, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)



#-------------------------------------
#Include a method to read a model from a file
#-------------------------------------
from keras.models import load_model
model = load_model('mnsit.h5')
#-------------------------------------
#Have a function(s) that visualizes what the CNN is doing inside
#-------------------------------------


### Extracts the outputs of the top five layers and 
### Creates a model that will return these outputs, given the model input

from keras.preprocessing import image
img=train_images[1]
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

from keras import models
layer_outputs = [layer.output for layer in model.layers[:5]]

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)


### Visualizing every channel in every intermediate activation

layer_names = []
for layer in model.layers[:5]:
    layer_names.append(layer.name)
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    # Number of features in the feature map
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    # Tiles the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        # Post-processes the feature to make it visually palatable
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                              :, :,
                                              col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # Displays the grid
            display_grid[col * size : (col + 1) * size,
                                       row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
    scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()










		

























