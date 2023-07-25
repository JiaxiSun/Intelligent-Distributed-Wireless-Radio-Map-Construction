import os
import numpy as np
from torch.utils.data import Dataset

os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

class DoraSet(Dataset):
    def __init__(self,dataset_path,set='train',clientId=1):
        self.set=set
        folder=dataset_path
        if set=='test':
            self.data = np.reshape(np.load(folder + 'test.npy'),(-1,6))
        else:
            self.data=np.reshape(np.load(folder+f'user_{clientId:02d}.npy'),(-1,6))

    def __getitem__(self, idx):
        return self.data[idx,:2],self.data[idx,2:]

    def __len__(self):
        return len(self.data)

class DoraSetComb(Dataset):
    def __init__(self,datasets):
        self.dataLen=[]
        self.datasets=datasets
        for i in datasets:
            self.dataLen.append(len(i))

    def __getitem__(self, idx):
        for i in range(len(self.dataLen)):
            if idx<np.sum(self.dataLen[:i+1]):
                if i==0:
                    idx2=idx
                else:
                    idx2=idx-np.sum(self.dataLen[:i])
                break
        return self.datasets[i][idx2]

    def __len__(self):
        return np.sum(self.dataLen)

if __name__=='__main__':
    dataset=DoraSet("data/train/",set="train",clientId=1)
    pos,pathloss=dataset[0]
    print(f'pos:',pos)
    print(f'pathloss:',pathloss)
