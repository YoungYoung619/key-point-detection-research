"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

from model.backbone.hourglass import hourglass

class redudant_point_network():
    def __init__(self, input, is_training, n_class):
        hourglass_feat = hourglass(input, is_training) ## shape [bs, h/4, w/4, 256]

        ## top left corner pooling

        ## top right cornet pooling

        ## bottom left corner pooling

        ## bottom right corner pooling

        ## max pooling for center
        pass