import os

from ray import client
from secretflow import reveal

from metrics import MetronAtK

# 设置 RAY_memory_monitor_refresh_ms 环境变量
os.environ['RAY_memory_monitor_refresh_ms'] = '0'
from NewFLModel import NewFLModel

import secretflow as sf
from secretflow.data.horizontal import read_csv
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.data.split import train_test_split
import numpy as np
from secretflow.ml.nn.core.torch import (
    metric_wrapper,
    optim_wrapper,
    BaseModule,
    TorchModel,
)
from secretflow.ml.nn import FLModel
from torchmetrics import Accuracy, Precision
from secretflow.security.aggregation import SecureAggregator
from secretflow.utils.simulation.datasets import load_mnist
from torch import nn, optim
import torch
from torch.nn import functional as F
from dataloder import dataLoder
import pandas as pd
from typing import List

item_num = 10
def list_datasets():
    data_dir = './data'
    datasets = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    if not datasets:
        print("无可用数据集.")
    else:
        print("可用数据集:")
        for idx, dataset in enumerate(datasets, start=1):
            print(f"{idx}. {dataset}")
    return datasets

def list_models():
    model_dir = './model'
    models = [f.name for f in os.scandir(model_dir) if f.is_dir()]
    if not models:
        print("无可用模型文件.")
    else:
        print("可用模型文件:")
        for idx, model in enumerate(models, start=1):
            print(f"{idx}. {model}")
    return models

def save_user_data(user_id, item_ids, ratings, output_dir='.'):
    # 创建一个DataFrame
    df = pd.DataFrame({
        'item_id': item_ids,
        'rating': ratings
    })

    # 定义文件名
    file_name = f"client-{user_id}.csv"
    file_path = f"{output_dir}/{file_name}"
    # 保存为CSV文件
    df.to_csv(file_path, index=False)

# 定义模型1
class RecModel(BaseModule):
    def __init__(self):
        super(RecModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=item_num, embedding_dim=32)  # 假设嵌入维度
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.long()  # 将输入转换为long类型
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # 展平嵌入层输出
        x = self.fc(x)
        return torch.sigmoid(x)

# 定义模型2
class FedRAP(BaseModule):
    def __init__(self):
        super(FedRAP, self).__init__()
        self.item_personality = nn.Embedding(num_embeddings=item_num, embedding_dim=32)
        self.item_commonality = nn.Embedding(num_embeddings=item_num, embedding_dim=32)
        self.affine_output = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = x.long()  # 将输入转换为long类型
        item_personality = self.item_personality(x)
        item_commonality = self.item_commonality(x)
        x = item_personality + item_commonality
        x = x.view(x.size(0), -1)  # 展平嵌入层输出
        logits = self.affine_output(x)
        rating = torch.sigmoid(logits)
        return rating

def trainFromdata(dataname, epochs=2,batch_size = 32, modelchoose = 0):
    datas, _, _ ,i_num= dataLoder(dataname)
    global item_num
    item_num = i_num
    num = len(datas[0])
    clientnamelist: List[str] = []
    clients = []
    for i in range(num):
        save_user_data(i, datas[1][i], datas[2][i])
        clientnamelist.append('client-'+str(i))
    # In case you have a running secretflow runtime already.
    sf.shutdown()

    sf.init(parties=clientnamelist+['server'], address='local')
    for i in range(num):
        clients.append(sf.PYU(clientnamelist[i]))
    server =  sf.PYU('server')
    path_dict = {}
    for i in range(num):
        path_dict[clients[i]]='client-'+str(i)+'.csv'

    aggregator = PlainAggregator(server)
    comparator = PlainComparator(server)

    hdf = read_csv(filepath=path_dict, aggregator=aggregator, comparator=comparator)
    label = hdf["rating"]
    data = hdf.drop(columns="rating")
    train_data, test_data = train_test_split(
        data, train_size=0.99, shuffle=True, random_state=1234
    )
    train_label, test_label = train_test_split(
        label, train_size=0.99, shuffle=True, random_state=1234
    )
    loss_fn = nn.BCELoss
    optim_fn = optim_wrapper(optim.SGD, lr=1e-2)
    metrics = [
        metric_wrapper(Accuracy, task="binary"),
        metric_wrapper(Precision, task="binary")
    ]

    # 定义模型
    if modelchoose == 0:
        model_def = TorchModel(
            model_fn=RecModel,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(Accuracy, task="binary")]
        )
    elif modelchoose == 1:
        model_def = TorchModel(
            model_fn=FedRAP,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(Accuracy, task="binary")]
        )
    else:
        model_def = TorchModel(
            model_fn=RecModel,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(Accuracy, task="binary")]
        )

    aggregator = SecureAggregator(server, participants=clients)

    # spcify params
    fl_model = NewFLModel(
        server=server,
        device_list=clients,
        model=model_def,
        aggregator=aggregator,
        strategy='fed_avg_w',  # fl strategy
        backend="torch",  # backend support ['tensorflow', 'torch']
        # aggregate=[1, 0, 0]
    )
    if modelchoose == 0:
        fl_model.fit(
            train_data,
            train_label,
            validation_data=(test_data, test_label),
            epochs=epochs,
            batch_size=batch_size,
            aggregate_freq=1,
            aggregate=[1, 0, 0]
        )
    elif modelchoose == 1:
        fl_model.fit_without_personalize(
            train_data,
            train_label,
            validation_data=(test_data, test_label),
            epochs=epochs,
            batch_size=batch_size,
            aggregate_freq=1,
            aggregate=[0, 1, 0, 0]
        )
    else:
        fl_model.fit(
            train_data,
            train_label,
            validation_data=(test_data, test_label),
            epochs=epochs,
            batch_size=batch_size,
            aggregate_freq=1,
            aggregate=[1, 0, 0]
        )
    return fl_model,num,clients
def savemodel(model,clients,modelname,selected_dataset,num, modelchoose = 0):
    # 文件夹路径
    if modelchoose == 1:
        directory_path = f"model/{modelname}-{'2'}-{selected_dataset}-{num}"
    else:
        directory_path = f"model/{modelname}-{'1'}-{selected_dataset}-{num}"

    # 文件夹
    os.makedirs(directory_path, exist_ok=True)

    path_dict = {}
    for i in range(num):
        path_dict[clients[i]]=os.path.join(directory_path, 'client-'+str(i))
    model.save_model(path_dict)

def loadmodel(modelname):
    num = int(modelname.split('-')[-1])
    clientnamelist: List[str] = []
    clients = []
    for i in range(num):
        clientnamelist.append('client-'+str(i))
    # In case you have a running secretflow runtime already.
    sf.shutdown()

    sf.init(parties=clientnamelist+['server'], address='local')
    for i in range(num):
        clients.append(sf.PYU(clientnamelist[i]))
    server =  sf.PYU('server')

    aggregator = PlainAggregator(server)
    comparator = PlainComparator(server)

    loss_fn = nn.BCELoss
    optim_fn = optim_wrapper(optim.SGD, lr=1e-2)
    metrics = [
        metric_wrapper(Accuracy, task="binary"),
        metric_wrapper(Precision, task="binary")
    ]
    parts = modelname.split('-')
    dataname = '-'.join(parts[2:-1])
    if dataname == "ml-1m":
        i_num = 3706
    elif dataname == "ml-100k":
        i_num = 1682
    elif dataname == "lastfm-2k":
        i_num = 12454
    elif dataname == "amazon":
        i_num = 11830
    elif dataname == "ml-100k-mini1":
        i_num = 107
    elif dataname == "ml-100k-mini2":
        i_num = 107
    else:
        i_num = 107
    global item_num
    item_num = i_num
    modelchoosename = str(modelname.split('-')[1])
    # 定义模型
    if modelchoosename == "2":
        # print("FedRAP")
        model_def = TorchModel(
            model_fn=FedRAP,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(Accuracy, task="binary")]
        )
    else:
        model_def = TorchModel(
            model_fn=RecModel,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(Accuracy, task="binary")]
        )
    aggregator = SecureAggregator(server, participants=clients)

    # spcify params
    fl_model = NewFLModel(
        server=server,
        device_list=clients,
        model=model_def,
        aggregator=aggregator,
        strategy='fed_avg_w',  # fl strategy
        backend="torch",  # backend support ['tensorflow', 'torch']
        # aggregate=[1, 0, 0]
    )
    # 文件夹路径
    directory_path = f"model/{modelname}"
    # 文件夹
    os.makedirs(directory_path, exist_ok=True)

    path_dict = {}
    for i in range(num):
        path_dict[clients[i]]=os.path.join(directory_path, 'client-'+str(i))

    fl_model.load_model(path_dict)
    return fl_model,clients,server

def predictFromnegative(model,clients,server,dataname, batch_size = 32):
    negativedata(dataname)
    num = len(clients)
    path_dict = {}
    for i in range(num):
        path_dict[clients[i]]='negative-client-'+str(i)+'.csv'

    aggregator = PlainAggregator(server)
    comparator = PlainComparator(server)

    hdf = read_csv(filepath=path_dict, aggregator=aggregator, comparator=comparator)
    data = hdf.drop(columns="true_itemId")
    result = model.predict(data,batch_size=batch_size)
    top_list = []
    for i in range(num):
        tensor_data_list =reveal(result[clients[i]])
        data_list = [t.item() for t in tensor_data_list]
        df = pd.read_csv(f"negative-client-{i}.csv")
        df['score'] = data_list
        # 选择 'score' 最高的前十个 'item_id'
        top_10 = df.nlargest(10, 'score')[['missing_itemId','true_itemId', 'score']]

        #保存整个 DataFrame 到新的 CSV 文件
        output_csv = f"negative-client-{i}.csv"
        df.to_csv(output_csv, index=False)
        #提取前十个 'item_id' 保存为列表
        top_10_item_ids = top_10['true_itemId'].tolist()
        top_list.append(top_10_item_ids)
    # print(top_list)
    return top_list

def predictFromtotal(model,clients,server,dataname, batch_size = 32):
    totaldata(dataname)
    num = len(clients)
    path_dict = {}
    for i in range(num):
        path_dict[clients[i]]=f"total-client-{i}.csv"

    aggregator = PlainAggregator(server)
    comparator = PlainComparator(server)

    hdf = read_csv(filepath=path_dict, aggregator=aggregator, comparator=comparator)
    data = hdf.drop(columns="true_itemId")
    result = model.predict(data,batch_size=batch_size)
    top_list = []
    for i in range(num):
        tensor_data_list =reveal(result[clients[i]])
        data_list = [t.item() for t in tensor_data_list]
        df = pd.read_csv(f"total-client-{i}.csv")
        df['score'] = data_list
        # 选择 'score' 最高的前十个 'item_id'
        top_10 = df.nlargest(10, 'score')[['item_id','true_itemId', 'score']]

        #保存整个 DataFrame 到新的 CSV 文件
        output_csv = f"total-client-{i}.csv"
        df.to_csv(output_csv, index=False)
        #提取前十个 'item_id' 保存为列表
        top_10_item_ids = top_10['true_itemId'].tolist()
        top_list.append(top_10_item_ids)
    # print(top_list)
    return top_list

def negativedata(dataname):
    dataset_dir = "./data/" + dataname + "/" + "ratings.dat"
    if dataname == "ml-1m":
        rating = pd.read_csv(dataset_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataname == "ml-100k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataname == "lastfm-2k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif dataname == "amazon":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    elif dataname == "ml-100k-mini1":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataname == "ml-100k-mini2":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    else:
        pass
    # Reindex
    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    item_id = rating[['mid']].drop_duplicates().reindex()
    item_id['itemId'] = np.arange(len(item_id))
    # 创建原始ID到新ID的映射
    original_to_new = item_id.set_index('mid')['itemId']
    # 创建新ID到原始ID的映射
    new_to_original = item_id.set_index('itemId')['mid']

    rating = pd.merge(rating, item_id, on=['mid'], how='left')
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]

    # 按userId分组，查找每个userId缺少的itemId
    for user_id, group in rating.groupby('userId'):
        existing_item_ids = group['itemId'].unique()
        missing_item_ids = np.setdiff1d(np.arange(len(item_id)), existing_item_ids)

        # 创建一个DataFrame来保存缺失的itemId
        missing_df = pd.DataFrame({
            'missing_itemId': missing_item_ids,
            'true_itemId': new_to_original.loc[missing_item_ids].tolist()
        })
        missing_df.to_csv(f"negative-client-{user_id}.csv", index=False)

def totaldata(dataname):
    dataset_dir = "./data/" + dataname + "/" + "ratings.dat"
    if dataname == "ml-1m":
        rating = pd.read_csv(dataset_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataname == "ml-100k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataname == "lastfm-2k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif dataname == "amazon":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    elif dataname == "ml-100k-mini1":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataname == "ml-100k-mini2":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    else:
        pass
    # Reindex
    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    item_id = rating[['mid']].drop_duplicates().reindex()
    item_id['itemId'] = np.arange(len(item_id))
    # 创建原始ID到新ID的映射
    original_to_new = item_id.set_index('mid')['itemId']
    # 创建新ID到原始ID的映射
    new_to_original = item_id.set_index('itemId')['mid']

    totallist = np.arange(len(item_id))
    rating = pd.merge(rating, item_id, on=['mid'], how='left')
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]

    # 按userId分组，查找每个userId缺少的itemId
    for user_id, group in rating.groupby('userId'):
        # 创建一个DataFrame来保存缺失的itemId
        df = pd.DataFrame({
                'item_id': np.arange(len(item_id)),
                'true_itemId': new_to_original.loc[totallist].tolist()
            })
        df.to_csv(f"total-client-{user_id}.csv", index=False)

def evalute(model,clients,server,dataname,batch_size = 32):
    textdata(dataname)
    num = len(clients)
    path_dict = {}
    for i in range(num):
        path_dict[clients[i]] = 'text-client-' + str(i) + '.csv'

    aggregator = PlainAggregator(server)
    comparator = PlainComparator(server)

    hdf = read_csv(filepath=path_dict, aggregator=aggregator, comparator=comparator)
    data = hdf["item_id"]
    result = model.predict(data, batch_size=batch_size)
    test_users = []
    test_items = []
    test_scores = []
    negative_users = []
    negative_items = []
    negative_scores = []
    for i in range(num):
        tensor_data_list = reveal(result[clients[i]])
        data_list = [t.item() for t in tensor_data_list]
        df = pd.read_csv(f"text-client-{i}.csv")
        df['score'] = data_list
        # 保存整个 DataFrame 到新的 CSV 文件
        output_csv = f"text-client-{i}.csv"
        df.to_csv(output_csv, index=False)
        test_users.append(i)
        test_items.append(df['item_id'].iloc[0])
        test_scores.append(data_list[0])
        for j in range(1,len(data_list)):
            negative_users.append(j)
            negative_items.append(df['item_id'].iloc[j])
            negative_scores.append(data_list[j])
    metron = MetronAtK(top_k=10)
    metron.subjects = [test_users,
                        test_items,
                        test_scores,
                        negative_users,
                        negative_items,
                        negative_scores]
    hit_ratio = metron.cal_hit_ratio()
    ndcg = metron.cal_ndcg()
    return hit_ratio, ndcg

def textdata(dataname):
    _, _, datas,_ = dataLoder(dataname)
    test_users, test_items, negative_users, negative_items = [tensor.tolist() for tensor in datas]
    # 创建一个新的字典来存储用户ID和对应的项目ID
    user_to_items = {}

    # 遍历test_users和test_items，将每个用户ID对应的项目ID加入字典
    for user, item in zip(test_users, test_items):
        if user not in user_to_items:
            user_to_items[user] = []
        user_to_items[user].append(item)

    # 遍历negative_users和negative_items，将每个用户ID对应的项目ID加入字典
    for user, item in zip(negative_users, negative_items):
        if user not in user_to_items:
            user_to_items[user] = []
        user_to_items[user].append(item)

    # 保存每个用户的项目ID到单独的CSV文件中
    for user, items in user_to_items.items():
        # 创建一个DataFrame
        df = pd.DataFrame(items, columns=['item_id'])

        # 定义文件名
        filename = f'text-client-{user}.csv'
        # 保存到CSV文件
        df.to_csv(filename, index=False)

def main():
    print("功能列表:")
    print("1. 模型训练")
    print("2. 商品推荐")
    print("3. 模型评估")

    choice = input("选择执行的功能: ")

    if choice == '1':
        datasets = list_datasets()
        if datasets:
            dataset_choice = input("输入要执行训练的数据集序号: ")
            print("1.个性化联邦推荐模型，2.加性个性化联邦推荐模型")
            modelchoose = input("选择待训练模型序号:")
            try:
                selected_dataset = datasets[int(dataset_choice) - 1]
                epochs = input("设置epochs: ")
                batch_size = input("设置batch_size: ")
                model, num, clients= trainFromdata(selected_dataset,int(epochs),int(batch_size),int(modelchoose)-1)
                print("0.保存模型，1.直接结束")
                issave = input("选择接下来的操作:")
                if issave == '0':
                    inputfilename = input("请为模型命名:")
                    savemodel(model, clients, inputfilename, selected_dataset, num, int(modelchoose)-1)
                    print("已保存，自动退出")
                if issave == '1':
                    return
            except (IndexError, ValueError):
                print("无效选择. 请重新输入.")

    elif choice == '2':
        models = list_models()
        if models:
            model_choice = input("输入要加载的模型序号: ")
            try:
                selected_model = models[int(model_choice) - 1]
                model,clients,server = loadmodel(selected_model)
                print("0.从所有商品进行推荐，1.从未评分商品进行推荐")
                choose = input("选择接下来的操作:")
                batch_size = input("设置batch_size: ")
                parts = selected_model.split('-')
                namedata = '-'.join(parts[2:-1])
                if choose == '0':
                    top10_list = predictFromtotal(model, clients, server, namedata, int(batch_size))
                else:
                    top10_list = predictFromnegative(model, clients, server, namedata, int(batch_size))
                for i in range(len(top10_list)):
                    print('client-',i,'推荐前十商品为',top10_list[i])
            except (IndexError, ValueError):
                print("无效选择. 请重新输入.")

    elif choice == '3':
        models = list_models()
        if models:
            model_choice = input("输入要加载的模型序号: ")
            batch_size = input("设置batch_size: ")
            # try:
            selected_model = models[int(model_choice) - 1]
            model, clients, server = loadmodel(selected_model)
            parts = selected_model.split('-')
            namedata = '-'.join(parts[2:-1])
            hit_ratio, ndcg = evalute(model, clients, server, namedata, int(batch_size))
            print('模型的命中率HR@10为:', hit_ratio,'归一化折损累计增益NDCG@10为:', ndcg)
            # except (IndexError, ValueError):
            #     print("无效选择. 请重新输入.")
    else:
        print("无效选择. 请输入 1, 2, 或者 3.")

if __name__ == '__main__':
    main()