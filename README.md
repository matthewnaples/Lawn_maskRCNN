# <u>README</u>

# <u>Mask R-CNN for Grass Detection and Segmentation from Satellite Images</u>

This is an application of the Mask R-CNN model based on the existing implementation by Matterport Inc. released by MIT under their license. The model itself is based on the open-source Python libraries Keras and Tensorflow.

![](/Users/joelnorthrup/Desktop/Deep Learning/100 epoch model predict.png)

# Getting Started

First we need to clone Matterport's implementation of the Mask R-CNN from Github, as well as the Weights from the model trained with the COCO dataset.

```python
!git clone https://github.com/matterport/Mask_RCNN.git
!cd Mask_RCNN

!git clone https://github.com/waleedka/coco
!pip install -U setuptools
!pip install -U wheel
!make install -C coco/PythonAPI

!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
```

We also import and download required packages.

```python
%tensorflow_version 1.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

!pip install Cython

ls /content/coco/PythonAPI

cd ./Mask_RCNN

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
from PIL import Image
from os import listdir
from xml.etree import ElementTree
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import cv2
import time
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
from imgaug.augmentables.batches import Batch

# Root directory of the project
ROOT_DIR =  os.getcwd()  #os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn.utils import Dataset
from mrcnn import visualize
from mrcnn.model import MaskRCNN
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
```

# Model Configurations and Methods

After we have imported and downloaded the required packages and repositories, we must configure our model. We do that by first defining a myMaskRCNNConfig class.

```python
class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
     # Grass + Background
    NUM_CLASSES = 1+1
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 131
    
    # Learning rate
    LEARNING_RATE=0.0009
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10
    
config = myMaskRCNNConfig()
```

We display our model configurations. 

```
config.display()
```

Next, we define our GrassDataset class. In this class are all of our functions to read in the data and annotations from our xml files. Our read_xml() function reads in our xml files. Our load_dataset() function loads corresponding images and annotations and randomly splits them up into our train and test datasets. Our load_image() and load_mask() functions loads our images and masks, respectively. Our extract_boxes() function extracts bounding boxes from our annotations.

We created the augment_data() function to increase our dataset through augmentations. When called, the function replicates images in from dataset that are randomly augmented. The annotations for these augmented images are also correspondingly augmented.  We used this function to increase our dataset from 69 annotated images to 5000 annotated images.

```python
class GrassDataset(utils.Dataset):
    def __init__(self, class_map=None):
      self._image_ids = []
      self.image_info = []
      # Background is always the first class
      self.class_info = [{"source": "", "id": 0, "name": "BG"}]
      self.source_class_ids = {}
      #matt's addition to constructor self.augmentations should hold [(img,PolygonsOnImage), (),...]
      self.augmentations=[]

  #read xml file to create coordinate system of mask and class id
    def read_xml(self,filename):
      tree = ElementTree.parse(filename)
      root = tree.getroot()
      #get image dimensions
      width = int(root.find('.//size/width').text)
      height = int(root.find('.//size/height').text)
      depth = int(root.find('.//size/depth').text)
      #create lists
      polygons = []
      names = []
      #iterates through each object attribute within the XML file
      for i,x in enumerate(root.findall('object')):
        #adds the name for each object
        names.append(x.find('name').text)
        #iterates through each polygon attribute within the object attribute
        for y in x.iter(tag = 'polygon'):
          #count to switch between x and y coordinate system
          count = 0
          #create x and y coordinate lists
          x_coord = []
          y_coord = []
          #iterates through each x and y coordinate
          for z in y.getchildren():
            count += 1
            #convert coordinate to integer value
            coord = int(z.text)
            #checks if it is an x or y coordinate
            if(count % 2 ==0):
              y_coord.append(coord)
            else:
              x_coord.append(coord)
        #append x and y coordinate list to polygons list
        xyCoord = list(zip(x_coord,y_coord))
        polygons.append(Polygon(xyCoord,names[i]))
      return width,height,depth,polygons

    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        
        # Add classes. We have one class to add.
        self.add_class("dataset", 1, "grass")
        #self.add_class("dataset", 2, "house")
        
        # define data locations for images and annotations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations/'
        
        # Iterate through all files in the folder to 
        #add class, images and annotaions
        for filename in listdir(images_dir):
            #print(filename)
            # extract image id
            image_id = filename[:-4]
            #print(image_id)
            
            # skip bad images
            if filename == '.DS_Store':
              continue
            # skip all images after 52 if we are building the train set
            if is_train and int(image_id) >= 52:
                continue
            # skip all images before 52 if we are building the test/val set
            if not is_train and int(image_id) < 52 :
                continue
            
            # setting image file
            img_path = images_dir + filename
            
            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'
            w,h,d,polygons = self.read_xml(ann_path)
            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, height = h, width = w, depth = d,polygons = polygons)
# extract bounding boxes from an annotation file

    def load_image(self, image_id):
          """Load the specified image and return a [H,W,3] Numpy array.
          """
          # Load image
          if self.image_info[image_id]['path'] == "augmented":
            non_augmented_size = len(self.image_info)- len(self.augmentations)
            # changing skimage.io.imread() --> imageio.imread()
            #print("here are the augmentations")
            #print(self.augmentations)
            #img = Image.fromarray(train_set.augmentations[0][0])
            image = self.augmentations[ image_id - non_augmented_size ][0]
          else:
            image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
          if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
          if image.shape[-1] == 4:
            image = image[..., :3]
          return image
      

    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

# load the masks for an image
#Generate instance masks for an image.
#       Returns:
#        masks: A bool array of shape [height, width, instance count] with
#            one mask per instance.
#       class_ids: a 1D array of class IDs of the instance masks.
#     
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
      
        #read from image info
        w = info['width']
        h = info['height']
        polygons = info['polygons']
        #polygons --> [Polygon(), Polygon(), ...]
        #polygons[i].yy_int
        # create one array for all masks, each on a different channel
        mask = zeros([h, w, len(polygons)], dtype='uint8')

        # create masks
        class_ids = list()
        for i in range(len(polygons)):
          #rr,cc = skimage.draw.polygon(poly[i][1],poly[i][0])
          rr,cc = skimage.draw.polygon(polygons[i].yy_int,polygons[i].xx_int)

          rr[rr > mask.shape[0]-1] = mask.shape[0]-1
          cc[cc > mask.shape[1]-1] = mask.shape[1]-1
          mask[rr, cc, i] = 1
          class_ids.append(self.class_names.index(polygons[i].label))
        return mask, asarray(class_ids, dtype='int32')

# load an image reference
#     """Return the path of the image."""
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']

    def augment_data(self,numImagesToAugment, numAugmentationsPerImage,sequential_model):
      imgs = []
      polygons_on_images = []
      if(numImagesToAugment > len(self.image_info)-len(self.augmentations)):
        raise Exception("Please choose a number <= %d" % len(self.image_info)-len(self.augmentations))
      for i in range(numImagesToAugment):
        info = self.image_info[i]
        
        curr_img = imageio.imread(info['path'])
        imgs.append(curr_img)
        polygons_on_images.append( PolygonsOnImage(info['polygons'],curr_img.shape) )
      image_and_polygons_augmentations=[]
      for i in range(numImagesToAugment):
        for _ in range(numAugmentationsPerImage):
          image_aug, psoi_aug = sequential_model(image=imgs[i], polygons=polygons_on_images[i])
          
          image_and_polygons_augmentations.append( (image_aug,psoi_aug) )

      #update the instance's augmentation array
      self.augmentations.extend(image_and_polygons_augmentations)

      #we've got the necessary info, now use it to add_image()
      for (img,poi) in image_and_polygons_augmentations:
        #parameters to add...
        image_id = "augmented"
        img_path = "augmented"
        ann_path = "augmented"
        height = img.shape[0]
        width = img.shape[1]
        depth = img.shape[2]
        polygons= poi.polygons
        #add the image
        self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path,
                      height = height, width = width, depth = depth,polygons = polygons)
      
```

# Loading our Datasets and Data Augmentation

All of the satellite images that comprise our dataset were annotated with the RectLabel software tool. Here is one of our annotated images.

![img](https://lh6.googleusercontent.com/WynXJBmnkt8oTwIdMajmVhP1rdUUnNhUmiM6T-uUzJUhuHowArRBjFORfqcL5TNaSrTIyTRUfZrSmm1L7gwK43SyS3_Qu9o3jsiaQk0pF73GS86ZpY_VxY-UviAg0d09XkYGOrvo)

We load this image and its annotations, as well as the other 68 images and their respective annotations into our model. Our load_dataset() function splits this data into our train and test datasets. Note: our data is stored in our Google Drive.

```python
# prepare train set
train_set = GrassDataset()
train_set.load_dataset('/content/drive/My Drive/Grass', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = GrassDataset()
test_set.load_dataset('/content/drive/My Drive/Grass', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
```

We next create an augmentation function with Imageio library and pass that function as one of the parameters into our augment_data() function. The other parameters are creating 72 new augmented images and annotations for each of the images in the test dataset and train dataset. We then print the sizes of the new datasets.

```python
aug = iaa.Sequential([
    iaa.Affine(rotate=(-20, 20), translate_percent=(-0.2, 0.2), scale=(0.8, 1.2),
               mode=["constant", "edge"], cval=0),
    iaa.Fliplr(0.5),
    iaa.PerspectiveTransform((0.01, 0.1)),
    iaa.Sometimes(0.5, iaa.ChannelShuffle(0.35)),
    #iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 3.0))),
    #iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-20, 20))),
    iaa.LinearContrast((0.8, 1.2), per_channel=0.5)])

train_set.augment_data(51,72,aug)
test_set.augment_data(18,72,aug)
print('length of train after aug: %d' % len(train_set.image_info))
print('length of test after aug: %d' % len(test_set.image_info))
```

Next, we call on the prepare() function in our test and train datasets which prepares the data and augmentations so that it can run through our model.

```python
train_set.prepare()
test_set.prepare()
```

To verify that our data is loaded correctly, we display a few random images and their respective augmentations.

```python
image_ids = np.random.choice(train_set.image_ids, 4)
for image_id in image_ids:
    image = train_set.load_image(image_id)
    mask, class_ids = train_set.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, train_set.class_names)
```

![](/Users/joelnorthrup/Desktop/Deep Learning/download (8).png)

![](/Users/joelnorthrup/Desktop/Deep Learning/download (9).png)

![download (10)](/Users/joelnorthrup/Desktop/Deep Learning/download (10).png)

![download (11)](/Users/joelnorthrup/Desktop/Deep Learning/download (11).png)

For a better idea of what our annotated data looks like, we display one random image from our dataset with the annotations colored.

```python
# Load random image and mask.
image_id = random.choice(train_set.image_ids)
image = train_set.load_image(image_id)
mask, class_ids = train_set.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)
# Display image and additional stats
print("image_id ", image_id, train_set.image_reference(image_id))
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, train_set.class_names)
```

![](/Users/joelnorthrup/Desktop/Deep Learning/download (12).png)

# Training our Model

We are now ready to begin training our model. As you remember, we are using the training weights from Matterport's implementation with the COCO dataset. Other implementations of the model were trained on the imagenet dataset. 

```python
# Which weights to start with?
init_with = "coco"  # coco or imagenet

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits","mrcnn_bbox_fc" ,"mrcnn_bbox","mrcnn_mask"])
```

We are now ready to begin training our model. After running our model many times, we found that an optimal learning rate is .0009. Learning rates higher than .0009 lead to the exploding gradient problem. We are leveraging the pre-trained model with the COCO dataset and only running our data through the "heads" layers of our model. While our model would benefit if we set the epochs higher, we were limited by our disk space while running on our Google Colab Pro environment. 

```python
print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')
model.keras_model.metrics_tensors = []
model.train(train_set, test_set, learning_rate=.0009, epochs=100, layers='heads')
history = model.keras_model.history.history
```

# Using our Model to Predict

Now that we have run our model (it took us approximately 2.5 hours), we will save our model weights, and load them so that we can predict with our model.

```python
import time
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

model_path = '/content/drive/My Drive/Grass/grass_mask_rcnn' + '.' + str(time.time()) + '.h5'

model.keras_model.save_weights(model_path)

#Loading the model in the inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')

# loading the trained weights to the custom dataset
model.load_weights(model_path, by_name=True)
```

Once we have loaded our saved weights, we are now able to use our model for segment detection. We detect on images that we did not include in our train and test datasets.

```python
img = load_img("/content/drive/My Drive/lawn-master/images/030.jpg")
img = img_to_array(img)

test = model.detect([img],verbose=1)
# Display results
t = test[0]
visualize.display_instances(img, t['rois'], t['masks'], t['class_ids'], 
                            test_set.class_names, t['scores'], 
                            title="Predictions")
```

![](/Users/joelnorthrup/Desktop/Deep Learning/download (6).png)

We can also display the ground truth masks vs the predicted masks. We use the visualize.display_differences() function to do this.

```python
image_id = random.choice(train_set.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(train_set, config, image_id, use_mini_mask=False)
info = train_set.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       train_set.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
r = results[0]

visualize.display_differences(image, gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'],r['scores'],r['masks'],
                              train_set.class_names)

modellib.log("gt_class_id", gt_class_id)
modellib.log("gt_bbox", gt_bbox)
modellib.log("gt_mask", gt_mask)
```

![](/Users/joelnorthrup/Desktop/download.png)

# Model Results

After displaying our model's mask predictions, we now plot the AP (Average Precision) metric for our model. We choose a random image from our dataset and plot the AP metric.

```python
image_id = random.choice(train_set.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(train_set, config, image_id, use_mini_mask=False)
results = model.detect([image], verbose=1)
r = results[0]
AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])

visualize.plot_precision_recall(AP, precisions, recalls)
```

![](/Users/joelnorthrup/Desktop/download (1).png)

We then find the mAP(mean Average Precision), and IoU (Intersection over Union) metrics from our model and print the results. 

```python
AP_sum = 0
IoU = 0
IoU_objects = 0
Overlaps = []
#compute sum of AP for 18 original testing images
for i in range(18):
  image_id = i
  image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(test_set, config, image_id, use_mini_mask=False)
  results = model.detect([image], verbose=1)
  r = results[0]
  AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
  AP_sum+=AP
  overlap = utils.compute_overlaps_masks(gt_mask,r['masks'])
  IoU+=np.sum(overlap)
  IoU_objects+=np.count_nonzero(overlap)
  Overlaps.append(overlap)
  
print("Average Precision:",AP_sum/18)
print("Average IoU:",IoU/IoU_objects)
```

<u>Average Precision: 0.860185188482757</u> 

<u>Average IoU: 0.6242606770502378</u>
