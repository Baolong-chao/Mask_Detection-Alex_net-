import torch

file = 'try_model.pkl'
f = open(file,'rb')
data = torch.load(f,map_location='cpu')#可使用cpu或gpu

print(type(data))
print(data.keys())
print(type(data['features.0.weight']))


fp = open('save.txt','w')
i=0
for key in data.keys():
    fp.write('-'*50+str(key)+'-'*50+'\n')
    print(data[str(key)].shape)
    for num in range(data[str(key)].shape[0]):
        fp.write('-'*20+str(num)+'kernel'+'-'*20+'\n')
        if i%2 == 0:
            for numm in range(data[str(key)].shape[1]):
                fp.write(str(data[str(key)][num][numm].numpy()))
                fp.write('\n')
        else:
            fp.write(str(data[str(key)][num].numpy()))
            fp.write('\n')
    i += 1