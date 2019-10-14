"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

img_size = (511, 511)
hm_size = (128, 128)
n_class = 80

min_radius_for_feedback_cal = 2
radius_scalar = 10

focal_loss_alpha = 2
focal_loss_belta = 4

## for vis
category_index = {0: {"name": "Background"},
                  1: {"name": "airplane"},
                  2: {"name": "bicycle"},
                  3: {"name": "bird"},
                  4: {"name": "boat"},
                  5: {"name": "bottle"},
                  6: {"name": "bus"},
                  7: {"name": "car"},
                  8: {"name": "cat"},
                  9: {"name": "chair"},
                  10: {"name": "cow"},
                  11: {"name": "diningtable"},
                  12: {"name": "dog"},
                  13: {"name": "horse"},
                  14: {"name": "motorbike"},
                  15: {"name": "person"},
                  16: {"name": "pottedplant"},
                  17: {"name": "sheep"},
                  18: {"name": "sofa"},
                  19: {"name": "train"},
                  20: {"name": "tvmonitor"}}

