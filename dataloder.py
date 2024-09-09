from data import SampleGenerator
import pandas as pd
import numpy as np

def dataLoder(dataname):
    """
    :param dataname: 文件名
    :return: 训练集all_train_data, 验证集validate_data, 测试集test_data
    """
    # Load Data
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

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=rating)
    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data
    all_train_data = sample_generator.store_all_train_data(3)
    return all_train_data, validate_data, test_data, len(item_id)