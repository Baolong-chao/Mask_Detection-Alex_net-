import torch
import numpy as np
from neural_network import AlexNet
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import os

'''
a= torch.ones(2,2,2)
b=a
print(b)
print('-'*20)
for i in range(b.shape[0]):
    print(i)
    for j in range(b.shape[1]):
        print(j)
        for z in range(b.shape[2]):
            print("!!!!")
            if b[i,j,z] >0:
                b[i,j,z]=0
            print('-'*10)
print('-'*20)
print(b)
'''

if __name__ == "__main__" :
    model__state_dict = torch.load('model.pkl')
    model_eval = AlexNet()
    print(type(model__state_dict))
    model_eval.load_state_dict(model__state_dict)
    print(type(model_eval.parameters()))
    print('-' * 50)
    for p in model_eval.parameters():
        print(p)
    for p in model_eval.parameters():
        p.data = torch.sign(p.data)

    print('-'*50)
    for p in model_eval.parameters():
        print(p)