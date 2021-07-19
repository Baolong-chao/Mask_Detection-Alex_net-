from neural_network import AlexNet
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import os

def load_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((127, 127), Image.BILINEAR)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1)) / 255
    img = np.expand_dims(img, 0)
    img = torch.tensor(img)
    return img

def convert(para):

    if para > 0:
        if para < 1/1024:
            q = 0
        elif para < 1/384:
            q = 1/512
        elif para < 1/192:
            q = 1/256
        elif para < 1/96:
            q = 1/128
        elif para < 1/48:
            q = 1/64
        elif para < 1/24:
            q = 1/32
        elif para < 1/12:
            q = 1/16
        elif para < 1/6:
            q = 1/8
        elif para < 1/3:
            q = 1/4
        elif para < 1:
            q = 1/2
        elif para >= 1:
            q = 1

    if para < 0:
        if para >-1/1024:
            q = 0
        elif para > -1/384:
            q = -1/512
        elif para > -1/192:
            q = -1/256
        elif para > -1/96:
            q = -1/128
        elif para > -1/48:
            q = -1/64
        elif para > -1/24:
            q = -1/32
        elif para > -1/12:
            q = -1/16
        elif para > -1/6:
            q = -1/8
        elif para > -1/3:
            q = -1/4
        elif para > -1:
            q = -1/2
        elif para <= -1:
            q = -1

    return q


if __name__ == "__main__" :
    model__state_dict = torch.load('try_model.pkl', map_location='cpu')
    model_eval = AlexNet()
    '''
    temp = model__state_dict
    flag = 0
    zero_flag = 0
    for key in temp.keys():
        for i in range(temp[key].shape[0]):
            if flag%2 == 0:  #判断是weight还是bias
                for j in range(temp[key].shape[1]):
                    if flag < 9: #判断卷积还是全连接阶段
                        for x in range(temp[key].shape[2]):
                            for y in range(temp[key].shape[3]):
                                temp[key][i,j,x,y]= convert(temp[key][i,j,x,y])    #卷积层weight处理
                                if temp[key][i,j,x,y] == 0:
                                    zero_flag += 1
                    else:
                        temp[key][i,j] = convert(temp[key][i,j])                #全连接层weight处理
                        if temp[key][i, j] == 0:
                            zero_flag += 1
            else:
                temp[key][i] = convert(temp[key][i])                          #bias处理
                if temp[key][i] == 0:
                    zero_flag += 1
        flag += 1
        print("%s  has done\n"%key)
    model__state_dict = temp
    print("the Number of zero is %d\n"%zero_flag)
    torch.save(model__state_dict, "try_model.pkl")  #存入try_model文件
    '''

    print(type(model__state_dict))
    print('-' * 50)
    model_eval.load_state_dict(model__state_dict)
    print(type(model_eval.parameters()))
    print('-' * 50)

    for p in model_eval.parameters():
        print(p.data)

    model_eval.eval()




for file in os.listdir("predict/1"):
    infer_path = os.path.join("predict/1/" + file)
    infer_img = Image.open(infer_path)
    plt.imshow(infer_img)
    plt.show()
    infer_img = load_image(infer_path)
    predicts = model_eval(infer_img)
    _, predict = torch.max(predicts, 1)
    print(predicts)
    predicts=torch.softmax(predicts,dim=1)
    print(predicts)
    print("the number of 1 is:", predict, "\n")

for file in os.listdir("predict/0"):
    infer_path = os.path.join("predict/0/"+file)
    infer_img = Image.open(infer_path)
    plt.imshow(infer_img)
    plt.show()
    infer_img = load_image(infer_path)
    predicts = model_eval(infer_img)
    _, predict = torch.max(predicts,1 )
    print(predicts)
    predicts=torch.softmax(predicts,dim=1)
    print(predicts)
    print("the number of 0 is:", predict, "\n")