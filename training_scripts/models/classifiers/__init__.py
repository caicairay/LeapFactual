from .simple_classifier import *
from .wraped_vgg import *
cls_models = {
              'MLP': SimpleClassifier,
              'VGG': WrapedVGG,
              }