from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import sys
import numpy as np

files = sys.argv[1:]

net = load_model('model.h5')

cls_list = ['cat', 'dog']

for f in files:
    img = image.load_img(f, target_size=(224, 224))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1][:5]

    print(f,' is ',cls_list[top_inds[0]])
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
    print()