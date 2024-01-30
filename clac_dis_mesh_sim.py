import numpy as np
import pandas as pd

def get_DV(disease_dict,delta=0.5):
    DV = 0
    for layer in disease_dict.values():
        DV = DV + pow(delta,layer)
    return DV

def get_intersection(disease_dict1,disease_dict2,delta=0.5):
    intersection_value = 0
    for key in disease_dict1.keys():
        if key in disease_dict2:
            intersection_value = intersection_value + pow(delta,disease_dict1[key]) + pow(delta,disease_dict2[key])
    return intersection_value

def cal_SimilarityByMeSHDAG(disease_dict1,disease_dict2):
    DV1 = get_DV(disease_dict1)
    DV2 = get_DV(disease_dict2)

    intersection_value = get_intersection(disease_dict1,disease_dict2)
    
    return intersection_value/(DV1+DV2)