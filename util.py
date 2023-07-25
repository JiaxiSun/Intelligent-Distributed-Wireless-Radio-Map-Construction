import os
import torch
import random
import numpy as np
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

class Recoder(object): # save the object value
    def __init__(self):
        self.last=0
        self.values=[]
        self.nums=[]
    def update(self,val,n=1):
        self.last=val
        self.values.append(val)
        self.nums.append(n)
    def avg(self):
        sum=np.sum(np.asarray(self.values)*np.asarray(self.nums))
        count=np.sum(np.asarray(self.nums))
        return sum/count

def seed_everything(seed: int):# fix the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def toNP(tensor):
    return tensor.detach().cpu().numpy()

def makeDIRs(folder):
    if not os.path.exists(f'models/{folder}/'):
        os.makedirs(f'models/{folder}/')
    if not os.path.exists(f'results/{folder}/'):
        os.makedirs(f'results/{folder}/')

def checkPoint(epoch, epochs, model, pathloss, ul_commCost, dl_commCost, saveModelInterval, saveLossInterval):
  
    if epoch % saveModelInterval == 0 or epoch == epochs:
        torch.save(model.state_dict(), f'models/model' + str(epoch) + '.pth',_use_new_zipfile_serialization=False)
    if epoch % saveLossInterval == 0 or epoch == epochs:
        np.save(f'results/pathloss.npy', np.asarray(pathloss))
        np.save(f'results/ul_commCost.npy', np.asarray(ul_commCost))
        np.save(f'results/dl_commCost.npy', np.asarray(dl_commCost))

def backward(optimizer,loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()