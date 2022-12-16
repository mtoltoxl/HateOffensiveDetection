from datasets import load_dataset
from transformers import  AutoTokenizer, pipeline
from collections import defaultdict
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import pdb
import random
from transformers import pipeline



class TextAugmentation:

    def __init__(self):
        self.model_name = 'roberta-base'
        self.unmasker = pipeline('fill-mask', model = self.model_name)# 'distilbert-base-uncased')

    def data_augmentation(self, tweet: str, rationale: list):
        orig_input_list = tweet.split()
        len_input = len(orig_input_list)

        orig_word_list = []
        #Random index where we want to replace the word
        replace_idx = [i for i, x in enumerate(rationale) if x == 0]
        new_text_list = orig_input_list.copy()

        # change RANDOMly selected 5 rationales to <mask>

        if len(replace_idx) > 5:
            replace_idx = random.sample(replace_idx, 5)

        for i in replace_idx:
            orig_word_list.append(orig_input_list[i])
            if self.model_name == 'distilbert-base-uncased':
                new_text_list[i] = '[MASK]'
            elif self.model_name == 'roberta-base':
                new_text_list[i] = '<mask>'
            else:
                print(f"wrong model: {self.model_name}")
                return

        # concat to string
        new_mask_sent = ' '.join(new_text_list)
        augmented_text_list = self.unmasker(new_mask_sent)

        sorted_idx = sorted(replace_idx)

        for i in range(len(orig_word_list)):
            orig_word = orig_word_list[i]
            idx = sorted_idx[i]

            if len(orig_word_list) == 1:
                '''
                res = augmented_text_list[random.sample(range(0,len(augmented_text_list)), 1)[0]]
                new_text_list[idx] = res['token_str']
                '''
                for res in augmented_text_list:
                    if res['token_str'] != orig_word:
                        new_text_list[idx] = res['token_str']
                        break
                
            else:
                aug_text_list = augmented_text_list[i]
                '''
                res = aug_text_list[random.sample(range(0, len(aug_text_list)), 1)[0]]
                new_text_list[idx] = res['token_str']
                '''
                for res in aug_text_list:
                    if res['token_str'] != orig_word:
                        new_text_list[idx] = res['token_str']
                        break
                
                

        return ' '.join(new_text_list)


def load_custom_dataset(benchmark, task):

    if task == 'hatexplain':
        train_dataset = load_dataset(task, split=f'train')
        test_dataset = load_dataset(task, split=f'test')     
    else:
        train_dataset = load_dataset(benchmark, task, split=f'train')
        test_dataset = load_dataset(benchmark, task, split=f'test')
    return train_dataset, test_dataset


def preprocess(train_dataset, train_flag):
    aug_cls = TextAugmentation()
    train = defaultdict(list)
    train['label'] = []
    train['text'] = []
    l_list = []
    t_list = []
    if train_flag:
        file = open("./augmented_text.txt", "w")
    count = 0
    for entry in train_dataset:
        annotate = entry['annotators']
        label = max(set(annotate['label']))
        tweet = ' '.join(entry['post_tokens'])
        l_list.append(label)
        t_list.append(tweet)
        
        if label == 0 and train_flag:
            rationales = entry['rationales']
            len_rationales = len(entry['rationales'])
            if len_rationales == 0:
                # if no rationales, no aug
                continue
            else:
                rationale = [0] * len(entry['post_tokens'])
                for rat in rationales:
                    rationale = np.logical_or(rationale, rat) * 1
                rationale = rationale.tolist()
            if sum(rationale) > 0 and sum(rationale) != len(entry['post_tokens']):
                new = aug_cls.data_augmentation(tweet, rationale)

                file.write(f"{new}\n")
                l_list.append(label)
                t_list.append(new)
                count += 1
    
    train['label'] = l_list
    train['text'] = t_list
    if train_flag:
        file.close()
    
    df = pd.DataFrame(train)
    print(f"total augmented: {count}")

    print(len(df[df['label'] == 0]))
    
    print(len(df[df['label'] == 1]))
    
    print(len(df[df['label'] == 2]))
    
    return df



def preprocess_past(train_dataset, train=False):
    aug_cls = TextAugmentation()
    train = defaultdict(list)
    train['label'] = []
    train['text'] = []
    l_list = []
    t_list = []
    for entry in train_dataset:
        annotate = entry['annotators']
        label = max(set(annotate['label']))
        tweet = ' '.join(entry['post_tokens'])
        l_list.append(label)
        t_list.append(tweet)
        
    train['label'] = l_list
    train['text'] = t_list
    
    df = pd.DataFrame(train)
    return df

def preprocess_binary(train_dataset):
    train = defaultdict(list)
    train['label'] = []
    train['text'] = []
    train['rationale'] = []
    l_list = []
    t_list = []
    r_list = []
    for entry in train_dataset:
        annotate = entry['annotators']
        label = max(set(annotate['label']))
        if label == 1:
            continue
        if label == 2:
            label = 1
        tweet = ' '.join(entry['post_tokens'])
        rationale = entry['rationales']
        l_list.append(label)
        t_list.append(tweet)
        r_list.append(rationale)
    train['label'] = l_list
    train['text'] = t_list
    train['rationale'] = r_list
    
    df = pd.DataFrame(train)
    return df

def preprocess_w_rationale(train_dataset):
    train = defaultdict(list)
    train['label'] = []
    train['text'] = []
    train['rationale'] = []
    l_list = []
    t_list = []
    r_list = []
    count = 0
    count1 = 0
    for entry in train_dataset:
        annotate = entry['annotators']
        label = max(set(annotate['label']))
        len_tokens = len(entry['post_tokens'])
        if label == 1:
            rationale = [1/len_tokens]*len_tokens
        if label == 0 or label == 2:
            len_rationales = len(entry['rationales'])
            if len_rationales == 0:
                continue
            else:
                sum_rationale = torch.tensor([0.0]*len_tokens)
                for i in entry['rationales']:
                    if len_tokens != len(i):
                        continue
                    sum_rationale += torch.tensor([float(j) for j in i])
                sum_rationale = 1/len_rationales * sum_rationale

        tweet = ' '.join(entry['post_tokens'])
        if label == 0 or label == 2:
            rationale = softmax(sum_rationale).tolist()
            if len(rationale)>128:
                rationale = rationale[:128]
        l_list.append(label)
        t_list.append(tweet)
        r_list.append(rationale)
    train['label'] = l_list
    train['text'] = t_list
    train['rationale'] = r_list
    
    df = pd.DataFrame(train)
    return df

def softmax(x):
    x = x.numpy()
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=0)

def cross_entropy(input1, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    # logsoftmax = nn.LogSoftmax(dim=0)
    # return torch.sum(-target * logsoftmax(input1))
    
    # if size_average:
    #     return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    # else:
    #     return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

    return torch.sum(-target * input1)
    

def masked_cross_entropy(input1,target,mask):
    cr_ent=0
    for h in range(0,mask.shape[0]):
        cr_ent+=cross_entropy(input1[h]*mask[h],target[h]*mask[h])

    return cr_ent/mask.shape[0]
