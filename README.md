# Image Classification with Swin Transformer

![image](https://www.itnonline.com/sites/default/files/styles/slider/public/brightcove/videos/images/posters/image_6286643812001.jpg?itok=0SwZFVDI)

## Getting Started

First, install the requirements necessary to run the python files.

```
$ pip install -r requirements.txt
```
As a side note, I used git lfs to upload my large data, but it seems like the data is corrupted. If that is the case in your usage, please download from this [shareable link](https://drive.google.com/uc?id=10-Ba3-WEYg1V8DeSpjUip6BmFdPzt9C7&export=download), and replace this .tfrecords in ```/data/processed/training10_0``` folder. My data processing pipeline will read images from ```./data/processed/training10_0/training10_0.tfrecords```.


Then, you can generate the processed images and labels with ```make_dataset.py```. This python file, however, is served as a function library for non-DL and DL model training. So, to run a training using Random Forest, you can do:

```
$ python3 -m scripts.train_random_forest
```

To run a Swin Transformer training, you can do:

```
$ python3 -m scripts.model
```

Finally, I have created a web app using streamlit. The app uses one of my best trained Swin Transformer model. In the web app, you can upload any of the images in my test_images dataset, upload them into web app, and you can run predictions by yourself and see their predicted class and confidence scores. You can freely try other images as you wish, but ideally you can make the image as the same mammography slice as I did. You can run the demo app with:

```
$ streamlit run app.py
```

As a side note, for safety issue (GCP does not allow me to upload security key to github), I hide out my API key at line #11. To run the model, you want to create a google storage bucket to save your AI model, use AI Platform to deploy this model, then create your own json key file. You can replace your json file name to my key entry.

I was also able to make this as a GCP-hosted cloud web app. To run the app, you want to put the model on GCP, as well as created an AI platform to support the model. Here is a nice [tutorial](https://www.youtube.com/watch?v=fw6NMQrYc6w) for GCP AI model deployment. After all these steps, you can deploy the model on GCP App Engine with:

```
$ make gcloud-deploy
```

Here's a demo of how the app works/looks like.

https://user-images.githubusercontent.com/72582001/165136001-ba043d53-97a7-4681-804f-1df98d2a3c9f.mov

## End Product
The end product is this [web app](https://aipi-540.uc.r.appspot.com/). You can click on this website to try it yourself by uploading my provided test images, or you can create same mammography image slices and see the results. 

## Data Sourcing & Processing

Pipeline comes from the Mammography dataset, and I store it into a tfrecords. From tfrecords, I do pre-processing steps and resizing to fit into the model. I also need to make images stacked into their 3 color channel. To extract images and labels from tfrecords, I need to decode image into int8, 
check every image is 598 by 598, and also need to check how many images are of each label.


## Non DL Modeling Details
For the Random Forest Classifier, I trained on both binary classification (cancer vs. noncancer), as well as noncancer, benign and malignant. Both have promising results. Here are the parameters used in RF Classifier:
N_estimators: 100
Criterion: gini
mini_samples_leaf : 2
Max_features: auto

## DL Modeling Details

In Swin Transformer, it adds shifted window approach to compute self-attention locally. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size.

During training, I first generate preprocessed image and labels from my dataset. I then assemble Swin Transformer model to: build a transformer encoder structure, wrap shifted window class, and create patch embeddings. I need to set up default parameters including: shifted window size, patch window size, embedding dimensions, MLP layer size, learning rate, batch size, and label smoothing. I start a training with 40 epochs, and save the best model with highest validation set accuracy. Later, I finetune hyperparameters, play with different params and see if there is a better result.



## Model Evaluation

For binary classification, it matches all SOTA results with a 0.87 accuracy. More training and fine tuning can push the score higher

For multi label classification, Swin Transformer provides a new model to predict different and harder classes on same dataset, and also achieves a promising result with a 0.69 accuracy.


**Pros:**

* Swin Transformer provides a new perspective to classify multi label mammography images: new model, new classification labels, promising results
* GCP can host Streamlit web app really well, providing a user friendly interface for as long as you wish

**Cons:**

* GCP AI logits calculation seems to have different result than what I got from Python/Colab, that might cause issues for individual experience
* Performance looks good but needs more improvement
* Datatype overflow for images in streamlit visualization, need to save 2 formats (0-1 and 0-255) of same image

In ```model.py```, once you do a training, you can just do

```
$ model.evaluate() 
```
to make an evaluation for the Swin Transformer model.

## Key User Interface Decisions

* Visibility: My app lets you upload an image and closely look at its image details
* Feedback: this web app is able to return a label and a confidence score for your uploaded image
* Application is simplified: choose model, upload image, see result
* Mapping: Clear relationships between the input (mammo image) and its result
* Consistency: no interruptions are expected; as long as I don’t stop GCP service, you can see the result
* Affordance: attributes of the web app communicate purpose


## Further improvements

* Fine tune the DL model better to push the accuracy score higher
* Make the UI page more interactive with more models to select, and more workflow options

## Citations
```
@article{DBLP:journals/corr/abs-2103-14030,
  author    = {Ze Liu and
               Yutong Lin and
               Yue Cao and
               Han Hu and
               Yixuan Wei and
               Zheng Zhang and
               Stephen Lin and
               Baining Guo},
  title     = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  journal   = {CoRR},
  volume    = {abs/2103.14030},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.14030},
  eprinttype = {arXiv},
  eprint    = {2103.14030},
  timestamp = {Thu, 08 Apr 2021 07:53:26 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-14030.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```

