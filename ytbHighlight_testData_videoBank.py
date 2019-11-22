import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import pickle
import copy
import json

def make_dataset(ytbData, subset, type_t):
    testDataIndx = []
    vid_prev = ''
    tmp = []
    for i, data in enumerate(ytbData):
        vid = data['video_id']
        domain = data['video'].split('/')[-2]
        if domain != subset:
            continue
        if data['type'] != type_t:
            continue
        if vid != vid_prev:
            vid_prev = vid
            if tmp != []:
                testDataIndx.append(tmp)
                tmp = []
                tmp.append(i)
        else:
            tmp.append(i)
    testDataIndx.append(tmp)
    return testDataIndx          
                
    
def get_annotate(ytbData, subset, type_t):
    root_dir = './DomainSpecificHighlight/'
    annotations = {}
    for i, data in enumerate(ytbData):
        vid = data['video_id']
        domain = data['video'].split('/')[-2]
        if domain != subset:
            continue
        if data['type'] != type_t:
            continue
        if vid in annotations:
            continue
        items = data['video'].split('/')
        annotation_path = root_dir + '/' + items[-2] + '/' + items[-1] + '/match_label.json'
        mturk_annotation = root_dir + '/' + items[-2] + '/' + items[-1] + '/mturk_label.json'
        with open (annotation_path, 'r') as fp:
            annotate =  json.load(fp)
        if os.path.exists(mturk_annotation):
            with open (mturk_annotation, 'r') as fp:
                mturk_annotate =  json.load(fp)
            for i, mturk_label in enumerate(mturk_annotate[1]):
                if float(mturk_label) > 0.5:
                    annotate[1][i] = 1
        
        annotations[vid] = annotate
    return annotations
        
class ytbHighlight_testData(data.Dataset):
    """
    Args:
        ytbData (string): ytbData.pkl.
        subset (string): domain
        type_t (sting): train valid  or test
    """

    def __init__(self,
                 ytbData_path,
                 subset,
                 type_t
                 ):
        with open(ytbData_path, 'rb') as fp:
            self.ytbData=pickle.load(fp)
        self.testDataIndx = make_dataset(
            self.ytbData, subset, type_t)
        self.annotate = get_annotate(self.ytbData, subset, type_t)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (input1, input2) where input1 has higher highlight score than input2.
        """
        indxs = self.testDataIndx[index]
        fea = torch.stack([self.ytbData[indx]['fea'] for indx in indxs], dim = 0) 
        frames = [ self.ytbData[indx]['frame_indices'] for indx in indxs ]
        vid = [ self.ytbData[indx]['video_id'] for indx in indxs ]
        return fea, vid, frames

    def __len__(self):
        return len(self.testDataIndx)
    
    
