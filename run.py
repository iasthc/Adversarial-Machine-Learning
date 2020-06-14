import sys
import numpy as np
from keras.applications import resnet
from keras.preprocessing import image
from keras.models import load_model
from keras.activations import relu, softmax
import keras.backend as K
import matplotlib.pyplot as plt

model = load_model('model.h5')

img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(224,224))

# Create a batch and preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = resnet.preprocess_input(x)

# Get the initial predictions
preds = model.predict(x)

# Inverse of the preprocessing and plot the image
def plot_img(x, i):
    """
    x is a BGR image with shape (? ,224, 224, 3) 
    """
    t = np.zeros_like(x[0])
    t[:,:,0] = x[0][:,:,2]
    t[:,:,1] = x[0][:,:,1]
    t[:,:,2] = x[0][:,:,0]  
    plt.imshow(np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(sys.argv[1] + '.adv.' + str(i) + '.png', bbox_inches = 'tight', pad_inches = 0)

# Get current session (assuming tf backend)
sess = K.get_session()
# Initialize adversarial example with input image
x_adv = x
# Added noise
x_noise = np.zeros_like(x)

# Set variables
epochs = 400
epsilon = 0.01
target_class = int(sys.argv[2])
prev_probs = []

for i in range(epochs): 
    # One hot encode the target class
    target = K.one_hot(target_class, 2)
    
    # Get the loss and gradient of the loss wrt the inputs
    loss = -1*K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    x_noise = x_noise + delta

    # Perturb the image
    x_adv = x_adv + epsilon*delta

    # Get the new image and predictions
    x_adv = sess.run(x_adv, feed_dict={model.input:x})
    preds = model.predict(x_adv)

    # Store the probability of the target class
    prev_probs.append(preds[0][target_class])

    if i%50==0:
        print(i, preds[0][target_class])
        plot_img(x_adv, i)