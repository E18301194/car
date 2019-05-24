  
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as tfs
import numpy as np
import torch
from torchvision import models
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch import optim
from torchvision import models



use_gpu = True
def train_tf(x):
    im=tfs.Compose([
        tfs.Resize(480),
        tfs.RandomHorizontalFlip(),
        tfs.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    x=im(x)
    return x

def test_tf(x):
    im2=tfs.Compose([
        tfs.Resize(480),
        tfs.ToTensor(),
        tfs.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    x=im2(x)
    return x

def get_acc(output,	label):
    total=output.shape[0]
 
    _,pred_label=output.max(1)

    num_correct=(pred_label==label).sum().data[0] 
    x=num_correct.float()
    return x / total





# try:
#     if iter(train_data).next()[0].shape[0] == batch_size and \
#     iter(test_data).next()[0].shape[0] == batch_size:
#         print('Success!')
#     else:
#         print('Not success, maybe the batch size is wrong!')
# except:
#     print('not success, image transform is wrong!')
#
# try:
#     model = get_model()
#     score = model(Variable(iter(train_data).next()[0], volatile=True))
#     if score.shape[0] == batch_size and score.shape[1] == 10:
#         print('successed!')
#     else:
#         print('failed!')
# except:
#     print('model is wrong!')








def train(model, train_data, valid_data,max_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,weight_decay=0.9)
    model.eval()

    if use_gpu:
        model= model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0,1,2]) 
     
    freq_print = int(len(train_data) / 100)

    metric_log = dict()
    metric_log['train_loss'] = list()
    metric_log['train_acc'] = list()
    if valid_data is not None:
        metric_log['valid_loss'] = list()
        metric_log['valid_acc'] = list()

    for e in range(max_epoch):
        model.train()
        running_loss = 0
        running_acc = 0

        for i, data in enumerate(train_data, 1):
            img, label = data
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            img = Variable(img)
            label = Variable(label)

            optimizer.zero_grad()
          
            score=model(img)

          
            loss=criterion(score,label)

           
            loss.backward()
            optimizer.step()




            running_loss += loss.data[0]
            running_acc += get_acc(score,label)

            if i % freq_print == 0:
                print('[{}]/[{}], train loss: {:.3f}, train acc: {:.3f}'.format(
                    i, len(train_data), running_loss / i, running_acc / i))

        metric_log['train_loss'].append(running_loss / len(train_data))
        metric_log['train_acc'].append(running_acc / len(train_data))

        if valid_data is not None:
            model.eval()
            running_loss = 0
            running_acc = 0
            for data in valid_data:
                img, label = data
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)

           
                score = model(img)

               
                loss = criterion(score, label)




                running_loss += loss.data[0]
                running_acc += get_acc(score,label)

            metric_log['valid_loss'].append(running_loss / len(valid_data))
            metric_log['valid_acc'].append(running_acc / len(valid_data))
            print_str = 'epoch: {}, train loss: {:.3f}, train acc: {:.3f}, \
            valid loss: {:.3f}, valid accuracy: {:.3f}'.format(
                e + 1, metric_log['train_loss'][-1], metric_log['train_acc'][-1],
                metric_log['valid_loss'][-1], metric_log['valid_acc'][-1])

        else:
            print_str = 'epoch: {}, train loss: {:.3f}, train acc: {:.3f}'.format(
                e + 1,
                metric_log['train_loss'][-1],
                metric_log['train_acc'][-1])
        print(print_str)
        print()
   
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
import os
from PIL import Image
class TestSet(object):
    def __init__(self, test_dir, test_transform):
        self.test_dir = test_dir
        self.test_transform = test_transform
        self.test_img_list = os.listdir(test_dir)

    def __getitem__(self, item):
        test_img_name = self.test_img_list[item]
        test_img = Image.open(os.path.join(self.test_dir, test_img_name))
        test_img = self.test_transform(test_img)
        return test_img, test_img_name

    def __len__(self):
        return len(self.test_img_list)
        
def predict_result(model, test_data, use_gpu):
    img_id = list()
    prob_result = list()
    model.eval()
    for data in test_data:
        img, img_name = data
        print(data)
        if use_gpu:
            img = img.cuda()
        img = Variable(img, volatile=True)
        score = F.softmax(model(img), dim=1)
        img_id.extend(img_name)
       
        prob_result.extend([i.numpy() for i in score.cpu().data])
        
    prob_result = np.array(prob_result)
    
    img_id = np.array(img_id)[:, None]
   
    all_data = np.concatenate((img_id, prob_result), axis=1)
    submission = pd.DataFrame(all_data)
    
    return submission

def main():

   batch_size = 32
   train_set = ImageFolder('./dataset/train',train_tf)
 #       train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)

 #       valid_set = ImageFolder('./dataset/valid',test_tf)
 #       valid_data = DataLoader(valid_set, 2 * batch_size, shuffle=False, num_workers=2)
 #       train_valid_set = ImageFolder('./dataset/train_valid/', train_tf)
 #       train_valid_data = DataLoader(train_valid_set, batch_size, shuffle=True, num_workers=2)

 #       net=models.resnet34()
 #       net.fc = nn.Linear(64512,10)
 #       train(net,train_data,valid_data,1)
 #       train(net, train_valid_data, None,1)
 #       torch.save(net,'net.pth')
   use_gpu = True
   model=torch.load('/DATA/maran/anaconda3/Project/car/net.pth')
   model=torch.nn.DataParallel(model,device_ids=[0,1])
   test_set = TestSet('/DATA/maran/anaconda3/Project/car/dataset/train/c0',test_tf)
   test_data = DataLoader(test_set,batch_size, num_workers=2)

   idx_to_class = dict()
   
   
   for i in train_set.class_to_idx:
      idx_to_class[train_set.class_to_idx[i]] = i
 

   submission = predict_result(model, test_data, use_gpu)
  
#   submission.columns = [['img'] + [i for i in idx_to_class.values()]]
#   submission.to_csv('./submission.csv', index=False)



       
      
      
     
if __name__ == '__main__':
        main()