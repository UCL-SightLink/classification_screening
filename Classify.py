import ClassUtils
import LoadUtils

import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import random

vgg16_state_path = "7vgg16_binary_classifier.pth"
# 7th Generation of VGG16 classifier - A prototype trained on only 1000 images
data_path = "zebra_annotations/classification_data"

classify = None
transform = None

def load_vgg_classifier(state_dict_path):
    # Ignore depreciation warnings --> It works fine for our needs
    model = models.vgg16(pretrained=True)

    # Modifies fully connected layer to output binary class predictions
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def load_resnet_classifier(state_dict_path):
    # Ignore depreciation warnings --> It works fine for our needs
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 1)

    state_dict = torch.load(state_dict_path)
    resnet.load_state_dict(state_dict)
    
    resnet.eval()
    return resnet

classify = load_vgg_classifier(vgg16_state_path)
transform = ClassUtils.vgg_transform

# Expects a numpy array image
def infer(image, infer_model=classify, infer_transform=transform):
    # If infer model and transform have not been initialised
    if infer_model is None or infer_transform is None:
        raise TypeError("Error: The inference classes have not been initialised properly.")
    if not torch.is_tensor(image):
        image = infer_transform(image)
    
    # Expects batches - this adds another dimensions 
    if len(image.shape) <= 3:
        image = image.unsqueeze(0)

    logit_pred = infer_model(image)

    prob = 1 / (1 + np.exp(-logit_pred.detach().numpy()[0]))
    # prob = max(0, min(np.exp(logit_pred.detach().numpy())[0], 1))
    return prob

# Expects a numpy image
def infer_and_display(image, threshold, actual_label, onlyWrong=False):
    probability = infer(image)
    prediction = probability > threshold
    is_correct = (actual_label == 1) == prediction

    if onlyWrong and is_correct:
        return prediction
    
    plt.imshow(torch.permute(image, (1, 2, 0)).detach().numpy())
    plt.title(f"Prediction: {prediction} with confidence {probability}%, Actual: {actual_label == 1}")
    plt.axis("off")
    plt.show()

    return probability


def example_init(examples=20):
    dataset = ClassUtils.CrosswalkDataset(data_path)
    
    random_points = [random.randint(0, len(dataset)-1) for i in range(examples)]

    for point in random_points:
        image, label = dataset[point]
        label = label[0]  # Do not need to train so doesn't have to be a tensor

        print(f"Prediction of {infer_and_display(image, 0.35, label)}% of a crosswalk (Crosswalk: {label==1})")
    
example_init()
