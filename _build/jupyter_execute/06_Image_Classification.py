#!/usr/bin/env python
# coding: utf-8

# # <div class='bar_title'></div>
# 
# *Practical Data Science*
# 
# # Image Classification with Deep Learning
# 
# Matthias Griebel<br>
# Chair of Information Systems and Business Analytics
# 
# Winter Semester 21/22

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Processing-Image-Data" data-toc-modified-id="Processing-Image-Data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Processing Image Data</a></span><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Requirements</a></span></li><li><span><a href="#Dataset:-Imagewoof" data-toc-modified-id="Dataset:-Imagewoof-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Dataset: Imagewoof</a></span></li><li><span><a href="#Looking-at-the-data" data-toc-modified-id="Looking-at-the-data-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Looking at the data</a></span></li><li><span><a href="#Creating-DataLoaders" data-toc-modified-id="Creating-DataLoaders-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Creating DataLoaders</a></span><ul class="toc-item"><li><span><a href="#Factory-Methods:-ImageDataLoaders" data-toc-modified-id="Factory-Methods:-ImageDataLoaders-1.4.1"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>Factory Methods: ImageDataLoaders</a></span></li></ul></li><li><span><a href="#The-data-block-API" data-toc-modified-id="The-data-block-API-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>The data block API</a></span><ul class="toc-item"><li><span><a href="#Presizing" data-toc-modified-id="Presizing-1.5.1"><span class="toc-item-num">1.5.1&nbsp;&nbsp;</span>Presizing</a></span></li><li><span><a href="#Dataloaders" data-toc-modified-id="Dataloaders-1.5.2"><span class="toc-item-num">1.5.2&nbsp;&nbsp;</span>Dataloaders</a></span></li></ul></li></ul></li><li><span><a href="#Training" data-toc-modified-id="Training-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Training</a></span><ul class="toc-item"><li><span><a href="#The-cnn_learner" data-toc-modified-id="The-cnn_learner-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>The cnn_learner</a></span></li><li><span><a href="#Find-the-learning-rate" data-toc-modified-id="Find-the-learning-rate-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Find the learning rate</a></span></li><li><span><a href="#Fit-the-model" data-toc-modified-id="Fit-the-model-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Fit the model</a></span></li><li><span><a href="#Unfreeze-and-train-again" data-toc-modified-id="Unfreeze-and-train-again-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Unfreeze and train again</a></span></li></ul></li><li><span><a href="#Results" data-toc-modified-id="Results-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Results</a></span><ul class="toc-item"><li><span><a href="#Top-Losses" data-toc-modified-id="Top-Losses-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Top Losses</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li></ul></div>

# __Credits for this lecture__
# 
# <img src="https://images-na.ssl-images-amazon.com/images/I/516YvsJCS9L._SX379_BO1,204,203,200_.jpg" width="500" align="right"/>
# 
# **Jeremy Howard and Sylvian Gugger: "Deep Learning for Coders with Fastai and PyTorch: AI Applications without a PhD." (2020).**
# 
# Available as [Jupyter Notebook](https://github.com/fastai/fastbook) 
# 
# Materials taken from
# - https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb
# - https://github.com/hiromis/notes/blob/master/Lesson1.md
# - https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

# ## Processing Image Data

# ### Requirements

# __Hardware: Graphics Processing Unit (GPU)__
# 
# GPU is fit for training the deep learning systems in a long run for very large datasets. CPU can train a deep learning model quite slowly. GPU accelerates the training of the model. Hence, GPU is a better choice to train the Deep Learning Model efficiently and effectively ([Medium](https://medium.com/@shachishah.ce/do-we-really-need-gpu-for-deep-learning-47042c02efe2)).
# 
# Make sure your GPU environment is set up and you can run Jupyter Notebook.
# 
# __GPU on [Google Colab](http://colab.research.google.com)__
# 
# * Select 'Runtime' -> 'Change runtime time' -> 'Python 3' (and 'GPU') before running the notebook.

# __Libraries__
# 
# We are going to work with the fastai V2 library which sits on top of Pytorch.
# The fastai library as a layered API as summarized by this graph:
# 
# <img src="https://docs.fast.ai/images/layered.png" width="500"/>

# We need to install/upgrade fastai

# In[1]:


#!pip install git+https://github.com/fastai/fastai.git # we need the current git version


# and import the library

# In[2]:


from fastai.vision.all import *


# ### Dataset: Imagewoof
# 
# <img src="https://miro.medium.com/max/960/1*zZJpK1EXPU-gVyt46kNypQ.jpeg" width="500" align="right"/>
# 
# 
# We are going to use the [Imagewoof](https://github.com/fastai/imagenette) data set, a subset of 10 classes from Imagenet that aren't so easy to classify, since they're all dog breeds. 
# 
# The breeds are: Australian terrier, Border terrier, Samoyed, Beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden retriever, Old English sheepdog. 

# __Download and extract__
# 
# The first thing we have to do is download and extract the data that we want. `untar_data` will download that to some convenient path and untar it for us and it will then return the value of path.

# In[3]:


# URLs.IMAGEWOOF_160 = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160'
path = untar_data(URLs.IMAGEWOOF_160); path


# Next time you run this, since you've already downloaded it, it won't download it again. Since you've already untared it, it won't untar it again. So everything is designed to be pretty automatic and easy.

# ### Looking at the data
# 
# The first thing we do when we approach a problem is to take a look at the data. We always need to understand very well what the problem is and what the data looks like before we can figure out how to solve it. Taking a look at the data means understanding how the data directories are structured, what the labels are and what some sample images look like.

# __Python 3 pathlib__
# 
# For convenience, fast.ai adds functionality into existing Python stuff. One of these things is add a `ls()` method to path.

# In[4]:


path.ls()


# In[5]:


(path/'train').ls()


# Path objects from the [pathlib](https://docs.python.org/3/library/pathlib.html) module are much better to use than strings. It doesn't matter if you're on Windows, Linux, or Mac. It is always going to work exactly the same way.

# __get_image_files__
# 
# `get_image_files` will just grab an array of all of the image files based on extension in a path.

# In[6]:


fnames = get_image_files(path/'train')
fnames


# Here, 'n02115641' refers to the class _dingo_ in [imagenet](https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57).

# ### Creating DataLoaders
# 
# The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the names of the (sub-)folders. We will need to extract them to be able to classify the images into the correct categories. 
# 
# We will now explore different ways load these such datasets.

# #### Factory Methods: ImageDataLoaders

# The [`ImageDataLoaders`](https://docs.fast.ai/vision.data.html#ImageDataLoaders) Class is a basic wrapper around several DataLoaders with factory methods for computer vision problems.
# 
# Here, we can use `ImageDataLoaders.from_folder`:

# In[7]:


dls = ImageDataLoaders.from_folder(path, train='train', valid='val', item_tfms=Resize(224))


# __data.show_batch__
# 
# Let's take a look at a few pictures. `dls.show_batch` can be used to show me some of the contents  So you can see roughly what's happened is that they all seem to have being zoomed and cropped in a reasonably nice way.

# In[8]:


dls.show_batch()


# ### The data block API
# 
# The [data block API](https://docs.fast.ai/data.block.html#DataBlock) lets you customize the creation of the Dataloaders by isolating the underlying parts of that process in separate blocks, mainly:
# 
# 1. The types of your input and labels
# 2. `get_items` (how to get your input)
# 3. `splitter` (How to split the data into a training and validation sets?)
# 4. `get_y` (How to label the inputs?)
# 
# ... and suitable transforms. 

# In[9]:


woof = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=GrandparentSplitter(train_name='train', valid_name='val'),
                 get_y=parent_label,
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75)
                 )


# One important piece of this `DataBlock` call that we haven't seen before is in these two lines:
# 
# ```python
# item_tfms=Resize(460),
# batch_tfms=aug_transforms(size=224, min_scale=0.75)
# ```
# 
# These lines implement a fastai data augmentation strategy which we call *presizing*. Presizing is a particular way to do image augmentation that is designed to minimize data destruction while maintaining good performance.

# #### Presizing

# <img alt="Presizing on the training set" width="600" caption="Presizing on the training set" id="presizing" src="https://raw.githubusercontent.com/fastai/fastbook/master/images/att_00060.png" align="right">
# 
# Presizing adopts two strategies
# 
# 1. *Crop full width or height*: This is in `item_tfms`, so it's applied to each individual image before it is copied to the GPU. It's used to ensure all images are the same size. On the training set, the crop area is chosen randomly. On the validation set, the center square of the image is always chosen.
# 2. *Random crop and augment*: This is in `batch_tfms`, so it's applied to a batch all at once on the GPU, which means it's fast. On the validation set, only the resize to the final size needed for the model is done here. On the training set, the random crop and any other augmentations are done first.
# 
# To implement this process in fastai you use `Resize` as an item transform with a large size, and `RandomResizedCrop` as a batch transform with a smaller size. `RandomResizedCrop` will be added for you if you include the `min_scale` parameter in your `aug_transforms` function, as was done in the `DataBlock` call in the previous section. Alternatively, you can use `pad` or `squish` instead of `crop` (the default) for the initial `Resize`.

# #### Dataloaders

# From the Datablock we can automatically get a our DataLoaders:

# In[10]:


dls = woof.dataloaders(path)


# .. and have a look at the summary:

# In[11]:


woof.summary(path)


# In[12]:


dls.show_batch()


# To make the classes easier to read and interpret, we can modify our `get_y` function. First, we define a dictionary for the labels:

# In[13]:


dogs_dict = { 'n02086240': 'Shih-Tzu',
              'n02087394': 'Rhodesian_ridgeback',
              'n02088364': 'beagle',
              'n02089973': 'English_foxhound',
              'n02093754': 'Border_terrier',
              'n02096294': 'Australian_terrier',
              'n02099601': 'golden_retriever',
              'n02105641': 'Old_English_sheepdog',
              'n02111889': 'Samoyed',
              'n02115641': 'dingo',
              }


# And define our own `get_y`:

# In[14]:


get_y = lambda x: dogs_dict[x.parent.name]


# Now, we can create the `DataBlock` again

# In[15]:


woof = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=GrandparentSplitter(train_name='train', valid_name='val'),
                 get_y=get_y,
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75)
                 )


# In[16]:


dls = woof.dataloaders(path)
dls.show_batch()


# ## Training

# Now we will start training our model. We will use a convolutional neural network backbone and a fully connected head with a single hidden layer as a classifier. 

# ### The cnn_learner
# 
# This method creates a Learner object from the data object and model inferred from it with the backbone given in `arch`.

# In[17]:


learn = cnn_learner(dls=dls, arch=models.resnet34, metrics=accuracy)


# - __dls__: Dataloaders
# - __arch__: architecture. There are lots of different ways of constructing a convolutional neural network. For now, the most important thing for you to know is that there's a particular kind of model called ResNet which works extremely well nearly all the time. For a while, at least, you really only need to be doing choosing between two things which is what size ResNet do you want. There are ResNet34 and ResNet50. 
# - __metrics__: accuracy
# - __loss_func__: automatically inferred from `dls`. What kind of loss function would typically choose for this task?

# Let's print a summary of the model.

# In[18]:


learn.summary()


# __Resnet Architecture__
# 
# <img src="https://miro.medium.com/max/1314/1*S3TlG0XpQZSIpoDIUCQ0RQ.jpeg" style="width:70%" />
# 
# 

# ### Find the learning rate

# Please read the fast.ai [lr_finder docs](https://docs.fast.ai/callback.schedule#Learner.lr_find). Also, Sylvain Gugger from the fast.ai team wrote a nice [blog post](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) on how to find  good learning rate.

# In[19]:


learn.lr_find()


# ### Fit the model
# 
# Fit the model based on selected learning rate

# In[20]:


learn.fit_one_cycle(2, lr_max=0.001)


# In[21]:


learn.save('stage-1')


# So far we have fitted 2 epochs and it ran pretty quickly. Why is that so? Because we used a little trick (called transfer learning).
# 
# What did we do?
# We added a few extra layers at the end of architecture, and we only trained those. We left most of the early layers as they were. This is called freezing layers i.e weights of the layers.
# 
# - When we call fit or `fit_one_cycle()` on a create_cnn, it will just fine-tune these extra layers at the end, and run very fast.
# - To get a better model, we have to call `unfreeze()` to train the whole model.

# ### Unfreeze and train again
# 
# Since our model is working as we expect it to, we will unfreeze our model and train some more.

# In[22]:


learn.unfreeze()


# In[23]:


learn.lr_find()


# __Learning rates after unfreezing__
# 
# The basic rule of thumb is after you unfreeze (i.e. train the whole thing), pass a max learning rate parameter, pass it a slice, make the second part of that slice about 10 times smaller than your first stage.

# In[24]:


learn.fit_one_cycle(10, lr_max=slice(1e-5,1e-4))


# If the model overfits we can reload stage 1 and train the model again for fewer epochs or another learning rate.

# In[25]:


learn = learn.load('stage-1')
learn.unfreeze()


# In[26]:


learn.fit_one_cycle(3, lr_max=slice(1e-5,1e-4))


# ## Results

# It’s important to see what comes out of our model. We have seen one way of what goes in, now let’s see what our model has predicted.
# 
# The `ClassificationInterpretation` class has methods for creating confusion matrix as well as plotting misclassified images.

# In[27]:


interp = ClassificationInterpretation.from_learner(learn)


# ### Top Losses
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not.

# In[28]:


interp.plot_top_losses(9, largest=True, figsize=(18,18))


# ### Confusion Matrix
# 
# Furthermore, when we plot the confusion matrix. Interestingly, the model often confuses English Foxhounds with Beagles. This confirmes that our model works very well until a certain level as these two dog breeds look very similar.

# In[29]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# __Most Confused__
# 
# Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences.

# In[30]:


interp.most_confused(min_val=2)

