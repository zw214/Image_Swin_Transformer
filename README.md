# Image_Swin_Transformer

## Motivation

## Getting Started

First, install the requirements necessary to run the python files.

```
$ pip install -r requirements.txt
```
As a side note, I used git lfs to upload my large data, but it seems like the data is corrupted. If that is the case in your usage, please download from this [shareable link](https://drive.google.com/uc?id=10-Ba3-WEYg1V8DeSpjUip6BmFdPzt9C7&export=download), and replace this .tfrecords in ```/data/processed/training10_0``` folder. My data processing pipeline will read images from this folder.


Then, you can generate the processed images and labels with ```make_dataset.py```. This python file, however, is served as a function library for non-DL and DL model training. So, to run a training using Random Forest, you can do:

```
$ python3 -m scripts/train_random_forest.py
```

To run a Swin Transformer training, you can do:

```
$ python3 -m scripts/model.py
```

Finally, I have created a web app using streamlit. The app uses one of my best trained Swin Transformer model. In the web app, you can upload any of the images in my test_images dataset, upload them into web app, and you can run predictions by yourself and see their predicted class and confidence scores. You can freely try other images as you wish, but ideally you can make the image as the same mammography slice as I did. You can run the demo app with:

```
$ streamlit run app.py
```
I was also able to make this as a GCP-hosted cloud web app. To run the app, you want to put the model on GCP, as well as created an AI platform to support the model. Once everything is done in your GCloud Platform, you can deploy the model on GCP with:

```
$ make gcloud-deploy
```

## Data Sourcing & Processing



## Modeling Details

## Model Evaluation

**Pros:**

* 


**Cons:**

* 

In ```model.py```, once you do a training, you can just do

```
$ model.evaluate() 
```
to make an evaluation for the Swin Transformer model.

## Further improvements

* 

## Citations
```
@article{

```