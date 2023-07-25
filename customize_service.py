# !/usr/bin/python
# -*- coding: UTF-8 -*-

import json
from model_service.pytorch_model_service import PTServingBaseService
import torch
import os
from model import DoraNet


class RadioMapService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('--------------------init--------------------')
        self.model_name = model_name
        self.model_path = model_path
        print(f"model_name:{model_name}")
        print(f"model_path:{model_path}")

        dir_path = os.path.dirname(os.path.realpath(model_path))
        self.train_dataset_path = dir_path.replace("/model", "/model/data/train/")
        print(f"dir_path={dir_path}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.file_names = []
        model = DoraNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        self.model = model

    def _preprocess(self, data):
        print('--------------------preprocess--------------------')
        preprocessed_data = {}
        for file_name, file_content in data['all_data'].items():
            print(f"file_name={file_name}, file_content={file_content}")
            self.file_name = file_name
            data_record = []
            lines = file_content.read().decode()
            lines = lines.split('\n')
            for line in lines:  # read all instance in the .txt
                if len(line) > 1:
                    data_record.append(json.loads(line))
            preprocessed_data[file_name] = data_record
        return preprocessed_data

    def _inference(self, data):
        print('--------------------inference----------------------')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_tmp = data[self.file_name]
        data_tmp = data_tmp[0]
        pos = torch.Tensor(data_tmp['pos']).view(-1, 2)
        result = self.model(pos).to(device)

        print(f'result:{type(result)}--{result}')
        pathloss = result.detach().squeeze(0).cpu().numpy().tolist()

        ## Here to change Codes to get the following pathloss uplink_loss downlink_loss ##

        uplink_loss = 5000
        downlink_loss = 5000
        results_fin = {'pathloss': pathloss, 'uplink_loss': uplink_loss, 'downlink_loss': downlink_loss}

        ##------------------------------------END-----------------------------------------##

        print(f'result_fin={results_fin}')
        return results_fin

    def _postprocess(self, data):
        print('--------------------postprocess--------------------')
        
        return data
