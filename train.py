import copy
import sys
from model import DoraNet
from util import *
from dataset import DoraSet, DoraSetComb
import os

seed_everything(42)

class EqualUserSampler(object):
    def __init__(self, n) -> None:
        self.i = 0
        self.n = n
        self.get_order()
    
    def get_order(self):
        self.users = np.arange(0, 90).reshape(9, -1)
        for i in range(9):
            np.random.shuffle(self.users[i])
        
        self.users = self.users.T 
        for i in range(10):
            np.random.shuffle(self.users[i])
        
        # 10 x 9
        self.users = self.users.reshape(-1)
    
    def get_useridx(self):
        selection = list()
        for _ in range(self.n):
            selection.append(self.users[self.i])
            self.i += 1
            if self.i % len(self.users) == 0:
                self.get_order()

            self.i %= len(self.users)
        return selection

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
sys.path.append("../..")

epochs = 500  # total epochs
local_epochs = 10 # local epochs of each user at an iteration
saveLossInterval = 1  # intervals to save loss
saveModelInterval = 2  # intervals to save model
batchSize = 512  # batchsize for training and evaluation
num_users = 90   # total users
num_activate_users = 5
lr = 3e-3  # learning rate
cudaIdx = "cuda:0"  # GPU card index
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
num_workers = 0  # workers for dataloader
evaluation = False  # evaluation only if True
criterion = torch.nn.MSELoss().to(device)


class Link(object):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.size = np.zeros((1,), dtype=np.float64)

    def pass_link(self, pay_load):
        for k, v in pay_load.items():
            self.size = self.size + np.sum(v.numel())
        return pay_load


class FedAvgServer: # used as a center
    def __init__(self, global_parameters, down_link):
        self.global_parameters = global_parameters
        self.down_link = down_link

    def download(self, user_idx):
        local_parameters = []
        for i in range(len(user_idx)):
            local_parameters.append(self.down_link.pass_link(copy.deepcopy(self.global_parameters)))
        return local_parameters

    def upload(self, local_parameters):
        for i, (k, v) in enumerate(self.global_parameters.items()):
            tmp_v = torch.zeros_like(v)
            for j in range(len(local_parameters)):
                tmp_v += local_parameters[j][k]
            tmp_v = tmp_v / len(local_parameters)  # FedAvg
            self.global_parameters[k] = tmp_v


class Client: # as a user
    def __init__(self, data_loader, user_idx):
        self.data_loader = data_loader
        self.user_idx = user_idx

    def train(self, model, learningRate, idx, global_model): # training locally
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
        for local_epoch in range(1, local_epochs + 1):
            for i, (pos, pathloss) in enumerate(self.data_loader):
                pos = pos.float().to(device)
                pathloss = pathloss.float().to(device)
                optimizer.zero_grad()
                p_pathloss = model(pos)
                loss = torch.mean(torch.abs(p_pathloss[pathloss != 0] - pathloss[pathloss != 0]))
                loss.backward()
                optimizer.step()
                # print(f"Client: {idx}({self.user_idx:2d}) Local Epoch: [{local_epoch}][{i}/{len(self.data_loader)}]---- loss {loss.item():.4f}")


def activateClient(train_dataloaders, user_idx, server):
    local_parameters = server.download(user_idx)
    clients = []
    for i in range(len(user_idx)):
        clients.append(Client(train_dataloaders[user_idx[i]], user_idx[i]))
    return clients, local_parameters


def train(train_dataloaders, user_idx, server, global_model, up_link, learningRate):
    clients, local_parameters = activateClient(train_dataloaders, user_idx, server)
    for i in range(len(user_idx)):
        model = DoraNet().to(device)
        model.load_state_dict(local_parameters[i])
        model.train()
        clients[i].train(model, learningRate, i, global_model)
        local_parameters[i] = up_link.pass_link(model.to('cpu').state_dict())
    server.upload(local_parameters)
    global_model.load_state_dict(server.global_parameters)


def valid(data_loader, model, epoch):
    with torch.no_grad():
        model.eval()
        losses = Recoder()
        scores = Recoder()
        for i, (pos, pathloss) in enumerate(data_loader):
            pos = pos.float().to(device)
            pathloss = pathloss.float().to(device)
            p_pathloss = model(pos)
            loss = torch.mean(torch.abs(p_pathloss[pathloss != 0] - pathloss[pathloss != 0])) ## unit in dB
            tmp1 = torch.sum(torch.abs(10 ** (0.1 * p_pathloss[pathloss != 0]) - 10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
            tmp2 = torch.sum(torch.abs(10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
            score = tmp1 / tmp2
            if score>1:
                score=torch.tensor([1])
            losses.update(loss.item(), len(pos))
            scores.update(score.item(), len(pos))
        print(f"Global Epoch: {epoch}----loss:{losses.avg():.4f}----pathloss_score:{-10 * np.log10(scores.avg()):.4f}", flush=True)
    return -10 * np.log10(scores.avg())


def train_main(train_dataset_path):
    train_dataloaders = []
    train_datasets = []
    valid_datasets = []
    if not os.path.exists(f'models/'):
        os.makedirs(f'models/')
    if not os.path.exists(f'results/'):
        os.makedirs(f'results/')
    for i in range(1, num_users + 1):
        all_dataset = DoraSet(train_dataset_path, set='train', clientId=i)
        train_size = int(0.985 * len(all_dataset))
        valid_size = len(all_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, [train_size, valid_size])
        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
        train_loader = torch.utils.data.DataLoader(all_dataset, batchSize, shuffle=True, num_workers=num_workers)
        train_dataloaders.append(train_loader)

    valid_data_comb = DoraSetComb(valid_datasets)
    valid_loader = torch.utils.data.DataLoader(valid_data_comb, 2048, shuffle=False, num_workers=num_workers)
    model = DoraNet()
    global_parameters = model.state_dict()
    up_link = Link("uplink")
    down_link = Link("downlink")
    server = FedAvgServer(global_parameters, down_link)

    pathloss_scores = []
    ul_commCost_scores = []
    dl_commCost_scores = []
    sampler = EqualUserSampler(5)
    for epoch in range(1, epochs + 1):  ## start training
        user_idx = sampler.get_useridx()
        # user_idx = np.random.choice(a=num_users, size=num_activate_users, replace=False, p=None).tolist()
        if lr < 50:
            train(train_dataloaders, user_idx, server, model, up_link, lr)
        elif lr < 150:
            train(train_dataloaders, user_idx, server, model, up_link, lr * 0.1)
        else:
            train(train_dataloaders, user_idx, server, model, up_link, lr * 0.02)
        
        test_model = copy.deepcopy(model).to(device)
        pathloss_score = valid(valid_loader, test_model, epoch)
        pathloss_scores.append(pathloss_score)
        ul_commCost_scores.append(up_link.size)
        dl_commCost_scores.append(down_link.size)
        checkPoint(epoch, epochs, model, pathloss_scores, ul_commCost_scores, dl_commCost_scores, saveModelInterval, saveLossInterval)


if __name__ == '__main__':
    
    train_main("data/train/")
