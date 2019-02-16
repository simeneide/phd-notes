## LUDEWIG IMPORT DATA
from ludewig.evaluation import loader
import pandas as pd
import numpy as np
from spotlight.interactions import Interactions, SequenceInteractions
import torch
def preprocess_data(dataset, device):
    if dataset == "rsc":
        ##RCS DATA##
        dat, dat_seq , ind2val = preprocess_rsc15(density_value = 0.005, limit_train = None, limit_test = None)

    elif dataset == "generated":
        ## GENERATED DATA ##
        dat, dat_seq, ind2val = preprocess_generated()

    # GENERATE A SMALLER TRAINING SET FOR METRIC TESTING
    dat_seq = {name : torch.tensor(dat_seq[name].sequences).long().to(device) for name in ['train','test']}
    dat_seq['train_small'] = dat_seq['train'][torch.randint(len(dat_seq['train']), (5000,))]
    return dat, dat_seq, ind2val

def preprocess_rsc15(density_value = 1.0, limit_train = None, limit_test = None):
    """
    Return index normalized sequences for train and test.
    
    density_value: randomly filter out events (0.0-1.0, 1:keep all)

    limit_train = limit_train #limit in number of rows or None
    limit_test = limit_test #limit in number of rows or None
    """
    data_path = 'ludewig/data/rsc15/single/'
    file_prefix = 'rsc15-clicks'
    density_value = density_value if density_value else 1.0

    remove_imdups = False
    train, test = loader.load_data(data_path, file_prefix, 
                                   rows_train=limit_train, 
                                   rows_test=limit_test, 
                                   density=density_value)
    for dat in train, test:
        dat.columns = ['sessionId','itemId','time']

    ind2val, val2ind = {}, {}
    for col in ['sessionId','itemId']:
        vals = np.unique(np.concatenate((train[col].values, test[col].values)))
        ind2val[col] = {idx+1 : id for idx, id in enumerate(vals)}
        val2ind[col] = {val : key for key, val in ind2val[col].items()}
        for df in [train, test]:
            df[col+"_idx"] = df[col].map(lambda x: val2ind[col][x])

    #train = train.to_sequence()
    #test = test.to_sequence()
    dat = {'train' : train, 'test' :  test}
    
    # Transform into sequence interaction object
    
    dat_seq = {}
    for name, df in dat.items():
        dat_seq[name] = Interactions(user_ids=df.sessionId_idx.values,
                    item_ids=df.itemId_idx.values,
                    timestamps=df.time.values).to_sequence(max_sequence_length = 10)
        
    return dat, dat_seq , ind2val

def preprocess_generated(num_users = 100, num_items = 1000, num_interactions = 10000):
    from spotlight.datasets.synthetic import generate_sequential
    from spotlight.cross_validation import user_based_train_test_split

    dataset = generate_sequential(num_users=num_users,
                                  num_items=num_items,
                                  num_interactions=num_interactions,
                                  concentration_parameter=0.0001,
                                  order=3)

    dat = {key: dat for key, dat in zip(["train","test"], user_based_train_test_split(dataset))}
    dat_seq = {key : val.to_sequence() for key, val in dat.items()}

    ind2val = {}
    ind2val['itemId'] = {idx : item for item, idx in enumerate(range(dataset.item_ids.max()))}

    return dat, dat_seq, ind2val

