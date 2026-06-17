import pandas as pd
import numpy as np
import os
from os import path
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time

if __name__ == "__main__":

    data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
    DP_DIR = Path("processed_data", data_name) 
    export_dir = Path(os.getcwd())
    files_path = Path(export_dir, DP_DIR)
    min_num_of_items_per_user = 2
    min_num_of_users_per_item = 2

    # Load ML1M data
    if data_name == "ML1M":
        data = pd.read_csv(Path(files_path, "ratings.dat"), sep="::", engine="python",
                        names=["user_id_original", "item_id_original", "rating", "timestamp"])
        
    # Load Yahoo data
    elif data_name == "Yahoo":
        data = pd.read_csv(Path(files_path, "Yahoo_ratings.csv"), names=["user_id_original", "item_id_original", "rating"])

    # Load Pinterest data
    elif data_name == "Pinterest":
        data = pd.read_csv(Path(files_path, "pinterest_data.csv"), names=["user_id_original", "item_id_original", "rating"])

    # Convert the ratings to binary values (1 if rating exists, 0 otherwise). 
    # Keep only ratings over 70/100.

    if data_name=='Yahoo':
        data["rating"] = data["rating"].apply(lambda x: 0 if x == 255 else x) # for Yahoo only
        data["rating"] = data["rating"].apply(lambda x: 1 if x > 70 else 0)
    elif data_name=='ML1M' or data_name=="ML1M_demographic":
        data["rating"] = data["rating"].apply(lambda x: 1 if x > 3.5 else 0)

    data = data[data['rating']==1]

    num_rows_1 = 1
    num_rows_2 = 2

    while num_rows_1 != num_rows_2:
        # save only users with min_num_of_items_per_user items or more
        user_counts = data.groupby(['user_id_original'])['item_id_original'].nunique().reset_index(name='item_count')
        filtered_users = user_counts[user_counts['item_count'] >= min_num_of_items_per_user]['user_id_original']
        data = data[data['user_id_original'].isin(filtered_users)].reset_index(drop=True)
        num_rows_1 = data.shape[0]
        
        # save only items with min_num_of_users_per_item users or more
        item_counts = data.groupby(['item_id_original'])['user_id_original'].nunique().reset_index(name='user_count')
        filtered_items = item_counts[item_counts['user_count'] >= min_num_of_users_per_item]['item_id_original']
        data = data[data['item_id_original'].isin(filtered_items)].reset_index(drop=True)
        num_rows_2 = data.shape[0]


    # Encode target values
    item_encoder = LabelEncoder()
    user_encoder = LabelEncoder()
    user_encoder.fit(data.user_id_original)
    item_encoder.fit(data.item_id_original)

    data["user_id"] = user_encoder.transform(data.user_id_original)
    data["item_id"] = item_encoder.transform(data.item_id_original)

    # Get the number of users and items in the dataset
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]

    print('num_items = ', num_items, ' num_users = ', num_users)

    # transform the data to encoding representation
    user_group = data[["user_id","item_id"]].groupby(data.user_id)

    users_data = pd.DataFrame(
        data={
            "user_id": list(user_group.groups.keys()),
            "item_ids": list(user_group.item_id.apply(list)),
        }    
    )


    mlb = MultiLabelBinarizer()
    user_one_hot = pd.DataFrame(mlb.fit_transform(users_data["item_ids"]),columns=mlb.classes_, index=users_data["item_ids"].index)

    user_one_hot["user_id"]=users_data["user_id"]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(user_one_hot.iloc[:,:-1], user_one_hot.iloc[:,-1], test_size=0.2, random_state=42)

    X_train.reset_index(drop=True, inplace=True)

    X_test.index = np.arange(X_train.shape[0], num_users)

    X_test.to_csv(Path(files_path, f'test_data_{data_name}.csv'))

    X_train.to_csv(Path(files_path, f'train_data_{data_name}.csv'))

    num_features = X_train.shape[1]

    data_array = X_train.to_numpy() #np array of one hot, shape (|U_train|,|I|)

    jaccard_dict = {}
    for i in range(num_features):
        for j in range(i, num_features):
            intersection = (data_array[:,i]*data_array[:,j]).sum()
            union = np.count_nonzero(data_array[:,i]+data_array[:,j])
            if union == 0:
                jaccard_dict[(i,j)]=0
            else:
                jaccard_dict[(i,j)]=(intersection/union).astype('float32')

    file_path = Path(files_path, f'jaccard_based_sim_{data_name}.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump(jaccard_dict, f)

    cosine_items = cosine_similarity(X_train.T).astype('float32')
    cosine_items.shape

    cosine_items_dict = {}

    # Loop through the rows and columns of the ndarray and add each element to the dictionary
    for i in range(cosine_items.shape[0]):
        for j in range(i,cosine_items.shape[1]):
            cosine_items_dict[(i, j)] = cosine_items[i][j]

    file_path = Path(files_path, f'cosine_based_sim_{data_name}.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump(cosine_items_dict, f)

    pop_array = (X_train.sum(axis=0)/X_train.sum(axis=0).max()).astype('float32') 
    pop_dict = {}

    for i in range(num_items):
        pop_dict[i]=pop_array[i]

    file_path = Path(files_path, f'pop_dict_{data_name}.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump(pop_dict, f)

    data_array = pd.concat([X_train, X_test], axis=0).to_numpy() #np array of one hot, shape (|U|,|I|)

    w_count = user_one_hot.iloc[:,:-1].sum(axis=1) # numer of items in user's history, shape = |U|

    n_appearance = user_one_hot.iloc[:,:-1].sum(axis=0) # number of appearances of item in user histories, shape = |I|

    tf_idf_dict = defaultdict(dict)
    for u in range(num_users):
        for i in range(num_items):
            if data_array[u,i] == 1:
                tf = 1/w_count[u]
                idf = np.log10(num_users/n_appearance[i])
                tf_idf_dict[u][i] = tf*idf

    file_path = Path(files_path, f'tf_idf_dict_{data_name}.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump(tf_idf_dict, f)