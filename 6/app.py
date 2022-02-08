from sys import exc_info
from flask import Flask, render_template, redirect, request, jsonify
import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import logging


app = Flask(__name__)

file_handler = logging.FileHandler('logs/error.log', encoding='utf-8', mode='a')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s:%(levelname)s-%(message)s')
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)

@app.route('/')
def index():
    return redirect('/create')

@app.route('/create')
def create():
    return render_template('create.html')

@app.route('/register_network', methods=['POST'])
def register_network():
    data = request.json
    name = data['name']
    layers = data['layers']
    path = os.path.join(app.root_path, 'static/networks', name)
    if not os.path.exists(path):
        os.mkdir(path)
    fname = os.path.join(path, 'architecture.json')
    with open(fname, 'w') as f:
        json.dump(layers, f, indent=4)
    return jsonify({'result': '新しいネットワークを登録しました。'})

@app.route('/optimize')
def optimize():
    networks = check_networks()
    datasets = check_datasets()
    return render_template('optimize.html',
                            networks=networks,
                            datasets=datasets)

@app.route('/evaluate')
def eveluate():
    return render_template('evaluate.html')

@app.route('/run_optimize', methods=['POST'])
def run_optimize():
    network = request.json['network']
    optimizer = request.json['optimizer']
    loss_function = request.json['loss_function']
    dataset = request.json['dataset']
    epochs = request.json['epochs']
    batch_size = request.json['batch_size']
    lr = request.json['learning_rate']
    data_path = os.path.join(app.root_path, 'static/data', dataset, 'data.csv')
    X = np.loadtxt(data_path, delimiter=',')
    target_path = os.path.join(app.root_path, 'static/data', dataset, 'target.csv')
    t = np.loadtxt(target_path, delimiter=',')
    
    # epochs,batch_size,lrが空欄だった場合デフォルト値を入れておく
    if epochs == '':
        epochs = 100
    if batch_size == '':
        batch_size = X.shape[0]
    if lr == '':
        lr = 1e-3

    net_path = os.path.join(app.root_path, 'static/networks', network, 'architecture.json')
    with open(net_path, 'r') as f:
        architecture = json.load(f)
    try:
        net = Net(architecture)
        # もし前回保存したパラメータがあれば読み込む
        param_path = os.path.join(app.root_path, 'static/networks', network, 'model.prm')
        if os.path.exists(param_path):
            params = torch.load(param_path)
            net.load_state_dict(params)

        if optimizer == 'SGD':
            opt = optim.SGD(net.parameters(), lr=float(lr))
        elif optimizer == 'Adam':
            opt = optim.Adam(net.parameters(), lr=float(lr))

        if loss_function == 'MSE':
            loss_fn = nn.MSELoss()
        elif loss_function == 'CrossEntropyLoss':
            loss_fn = nn.CrossEntropyLoss()

        X = torch.tensor(X, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        dataset = TensorDataset(X, t)
        data_loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

        result = train(net, data_loader, opt, loss_fn, int(epochs))
        # 学習済みのパラメータを保存する
        torch.save(net.state_dict(), param_path)
    except Exception as ex:
        app.logger.error(ex, exc_info=True)
        return jsonify({'result': 'モデル最適化時にエラーが発生しました。'})

    return jsonify({'result': result})

def check_datasets():
    path = os.path.join(app.root_path, 'static/data')
    path_list = os.listdir(path)
    if '.DS_Store' in path_list:
        path_list.remove('.DS_Store')
    return path_list

def check_networks():
    path = os.path.join(app.root_path, 'static/networks')
    path_list = os.listdir(path)
    if '.DS_Store' in path_list:
        path_list.remove('.DS_Store')
    return path_list

class Net(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.model = create_network(params)

    def forward(self, x):
        out = self.model(x)
        return out

# パラメータは{ "0": {'type': 'Linear', 'in': 123, 'out': 123}, ... }の形で与える
def create_network(params):
    net = nn.Sequential()
    for i,v in enumerate(params):
        if params[str(i)]['type'] == 'Linear':
            net.add_module(str(i), nn.Linear(int(params[str(i)]['in']), int(params[str(i)]['out'])))
        elif params[str(i)]['type'] == 'ReLU':
            net.add_module(str(i), nn.ReLU())
        elif params[str(i)]['type'] == 'Sigmoid':
            net.add_module(str(i), nn.Sigmoid())
    return net

def train(network, data_loader, optimizer, loss_fn, epochs):
    losses = []
    accuracies = []
    for epoch in range(epochs):
        loss_per_epoch = 0.0
        acc_per_epoch = 0.0
        for x,t in data_loader:
            optimizer.zero_grad()
            y = network(x)
            loss = loss_fn(y, t)
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item()
            y_index = torch.max(y, 1)[1]
            t_index = torch.max(t, 1)[1]
            acc_per_epoch += torch.sum(y_index == t_index) / len(data_loader.dataset)
        losses.append(loss_per_epoch)
        accuracies.append(acc_per_epoch)
    path = 'static/results/' + datetime.now().strftime('%Y%m%d%H%M%S') + '.png'
    abs_path = os.path.join(app.root_path, path)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(losses)
    ax1.set_xlabel('epoch', loc='right')
    ax1.set_ylabel('loss', loc='top')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(accuracies)
    ax2.set_xlabel('epoch', loc='right')
    ax2.set_ylabel('accuracy', loc='top')
    plt.tight_layout()
    plt.savefig(abs_path)
    return path

@app.route('/clear', methods=['POST'])
def clear():
    network = request.json['network']
    # もし保存された学習済みパラメータがあれば削除
    param_path = os.path.join(app.root_path, 'static/networks', network, 'model.prm')
    if os.path.exists(param_path):
        os.remove(param_path)
    return jsonify({'result': 'モデルのパラメーターを初期化しました。'})

@app.route('/delete', methods=['POST'])
def delete():
    network = request.json['network']
    # 選ばれた該当するネットワークフォルダを削除
    path = os.path.join(app.root_path, 'static/networks', network)
    shutil.rmtree(path)
    return jsonify({'result': 'ネットワーク「{}」を削除しました。'.format(network)})

if __name__ == '__main__':
    app.run()
