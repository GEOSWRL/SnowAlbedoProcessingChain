# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:31:07 2020

@author: aman3
"""

"""
step 1: train neural net to detect reflectance targets
step 2: detect reflectance targets across all images
step 3: manually get average DN values for reflectance targets
    step 3 (future): automatically detect and calculate average pixel value of reflectance targets
step 4: plot incoming radiation from upward pyranometer vs DN of reflectance target for each image w/ reflectance target
step 5: orthogonal linear regression to calculate hypothetical DN of incoming solar radiation from upward pyranometer
step 6: multiply each pixel in a given image by DN(image)/DN(incoming radiation)

"""
#from detecto import core, utils, visualize
from imageai.Prediction.Custom import ModelTraining

training_image_directory = "C:/Users/aman3/Documents/GradSchool/testing/training_dataset"
test_image = "C:/Users/aman3/Documents/GradSchool/testing/image_detect/DJI_0763"


def training(image_directory):
    model_trainer = ModelTraining()
    model_trainer.setModelTypeAsResNet()
    model_trainer.setDataDirectory(training_image_directory)
    model_trainer.trainModel(num_objects=1, num_experiments=200, enhance_data=True, batch_size=5, show_network_summary=True)
    
    
    
"""
def training(image_directory):
    dataset = core.Dataset(image_directory)
    model = core.Model(['reflectance_target', 'GCP'])
    model.fit(dataset)
    return model
    
def identify(image, model):
    image = utils.read_image('images/image0.jpg')
    predictions = model.predict(image)

    # predictions format: (labels, boxes, scores)
    labels, boxes, scores = predictions
    print(labels) 
    print(boxes)
    print(scores)
 """
    
training(training_image_directory)