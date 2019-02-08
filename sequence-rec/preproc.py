## LUDEWIG IMPORT DATA
from ludewig.evaluation import loader
import pandas as pd
import numpy as np
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

    ind2val, val2ind = {}, {}
    col_transform = {'SessionId' : 'sessionId', 'ItemId' : 'itemId'}
    for col in ['SessionId','ItemId']:
        vals = np.unique(np.concatenate((train[col].values, test[col].values)))
        ind2val[col] = {idx+1 : id for idx, id in enumerate(vals)}
        val2ind[col] = {val : key for key, val in ind2val[col].items()}
        for df in [train, test]:
            df[col+"_idx"] = df[col].map(lambda x: val2ind[col][x])

    #train = train.to_sequence()
    #test = test.to_sequence()
    dat = {'train' : train, 'test' :  test}
    return dat , ind2val