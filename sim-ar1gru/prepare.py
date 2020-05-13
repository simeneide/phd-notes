#%%
import DATAHelper
import os

import sys
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')

import FINNHelper
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructType, StructField, FloatType
from pyspark.sql import Window
import datetime
import numpy as np
import pandas as pd
import pickle
import datetime
from torch.utils.data import Dataset, DataLoader
import torch
from DATAHelper import pad_sequences
import utils


def mkdir(path):
    if os.path.isdir(path) == False:
        os.makedirs(path)


## PREPARE DATA

#%%
'''Build ind2val from a sequences dataframe. Needs to include the action column!'''
# Build ind2val:


def load_postcodes_to_spark(sqlContext):
    '''
    ASSUMES file postkoder.csv in base dir!
    File is received from PEM and is extracted from finn data
    However, we will only need to postnr -> fylke columns
    '''

    postkode = pd.read_table('postkoder.csv', sep=',')[['Postnr', 'Fylke']]
    postkode.columns = ['post_code', 'region']

    def stringify_postcode(x):
        x = str(x)
        return '0' * (4 - len(x)) + x

    postkode['post_code'] = postkode.post_code.map(stringify_postcode)
    postcode_spark = sqlContext.createDataFrame(postkode)
    return postcode_spark


def add_catvar_if_above_threshold(df, catvar, th=100):
    df = df.withColumn('proposed_category',
                       F.concat_ws('-', 'category', catvar))

    accepted_categories = (df.groupby('proposed_category').count().filter(
        F.col('count') > th).withColumn('keep', F.lit(True)).drop('count'))
    df = (df.join(
        accepted_categories, on='proposed_category', how='left').withColumn(
            'category',
            F.when(
                F.col('keep') == 'true', F.col('proposed_category')).otherwise(
                    F.col('category'))).drop('keep').drop('proposed_category'))
    return df


def build_global_ind2val(sqlContext, sequences, data_dir, drop_groups=False, min_item_views=50):

    active_items = (
        sequences.select(F.explode('action').alias('id'))
        .select(F.explode('id').alias('id'))
        .groupby('id').count()
        .withColumnRenamed(
            'count', 'actions')
        .filter(F.col('actions') > min_item_views)
        .filter(F.col('id') != 'noClick'))#.persist()

    ## LOAD PRETRAINED W2V IDs so that we can filter on it
    w2v_ind2item, w2v_item2ind, w2v_itemvec = prepare_w2v_from_file_or_web(
        data_dir)

    w2v_items_spark = (sqlContext.createDataFrame(
        list(w2v_item2ind.keys()), StringType()).withColumnRenamed(
            'value', 'id').withColumn('w2v_exist', F.lit(True))).persist()

    #%%
    # Fetch all items published 90 days before start_date:
    logging.info("Get items from contentDB..")
    published_after = datetime.datetime.strftime(
        (datetime.datetime.strptime(start_date, '%Y-%m-%d') -
         datetime.timedelta(120)), '%Y-%m-%d')

    published_before = start_date
    postcode_spark = load_postcodes_to_spark(sqlContext)

    category_pars = [
        'vertical', 'main_category', 'sub_category', 'prod_category', 'make',
        'post_code', 'model'
    ]

    q = f"""
        select id, {', '.join(category_pars)}
        from ad_content
        where (first_published >= '{published_after}')
        AND state = 'ACTIVATED'
        """
        #

    content = (
        FINNHelper.contentdb(sqlContext, q)
        # Use regions instead of postcodes (but keep name):
        .join(postcode_spark, on='post_code',
              how='left').drop('post_code').withColumnRenamed(
                  'region', 'post_code')
        # CONCAT CATEGORY STRINGS
        .fillna('', subset=category_pars).dropDuplicates(['id']))

    ## BUILD CATEGORY STRUCTURE:
    df = content.withColumn('category', F.col(category_pars[0]))
    for catvar in category_pars[1:]:  # skip first (vertical)
        df = add_catvar_if_above_threshold(df=df, catvar=catvar, th=100)

    content = df.withColumn('contentDB', F.lit(True)).persist()

    items = (
        active_items.join(
            w2v_items_spark, on='id',
            how='inner')  # USE ONLY ITEMS THAT ARE PRESENT IN W2V!!
        .join(content, on='id',
              how='left').filter((F.col('contentDB') == True)
                                 | (F.col('actions') >= min_item_views*5)))
    logging.info(
        f'There are {active_items.count()} items with actions. Found {content.count()} items in contentDB. Found {w2v_items_spark.count()} in pretrained w2v.'
    )
    logging.info(f'After filters we are left with {items.count()} items.')
    items_loc = items.toPandas()

    ### CONCAT DUMMYITEMS WITH REAL ITEMS:
    # Create some dummyitems
    unk = '<UNK>'
    fillitems = pd.DataFrame({
        'id': ['PAD', 'noClick', unk],
        'idx': [0, 1, 2],
        'actions': [-1, -1, -1],
        'category': ['PAD', 'noClick', unk]
    })
    # Add index to all real items (starting at 3):
    items_loc['idx'] = range(3, items_loc.shape[0] + 3)
    all_items = pd.concat([fillitems,
                           items_loc]).reset_index(drop=True).fillna(unk)
    all_items['idx']

    if drop_groups:
        logging.info('Dropping group information from datatset...')
        all_items['category'] = unk

    ind2val = {}
    ind2val['itemId'] = {
        int(idx): str(item)
        for idx, item in zip(all_items.idx.values, all_items.id.values)
    }

    ## ATTRIBUTE VECTORS
    # Attribute vectors on items. Each index of the array has a value corresponding
    # to the item index as described in ind2val['itemId]
    itemattr = {}
    # actions
    actions = np.zeros((all_items.idx.shape))
    for idx, action in zip(all_items.idx.values, all_items.actions.values):
        actions[idx] = action
    itemattr['actions'] = actions

    # Categorical variables:
    for var in ['category']:
        ind2val[var] = {
            int(idx): str(item)
            for idx, item in zip(all_items.idx.values, all_items[var].values)
        }
        ind2val['category'] = {
            idx: name
            for idx, name in enumerate(all_items['category'].unique())
        }
        vec = np.zeros((all_items.idx.shape))
        val2ind = {val: idx for idx, val in ind2val[var].items()}
        for idx, item in zip(all_items.idx.values, all_items[var].values):
            vec[idx] = int(val2ind.get(item))
        itemattr[var] = vec

    # displayType
    display_types = sequences.select(F.explode("displayType")).distinct().toPandas().values.flatten()
    ind2val['displayType'] = {i+1 : val for i, val in enumerate(display_types)}
    ind2val['displayType'][0] = "<UNK>"

    ## SAVE
    # Ind2val
    with open(f'{data_dir}/ind2val.pickle', 'wb') as handle:
        pickle.dump(ind2val, handle)
    logging.info('saved ind2val.')

    with open(f'{data_dir}/itemattr.pickle', 'wb') as handle:
        pickle.dump(itemattr, handle)
    logging.info('saved itemattr.')
    return ind2val, itemattr


def prepare_sequences(sqlContext,
                      sequences,
                      start_date,
                      end_date,
                      ind2val,
                      data_path,
                      data_dir,
                      data_type,
                      maxlen_time,
                      maxlen_action,
                      limit=False):
    ''' 
    If ind2val is supplied, this will be used instead of creating its own
    If sequences is supplied, we will not read new parquet.
    '''

    logging.info('Start preparing sequences...')

    sequences = (
        sequences.filter(F.col('date') < end_date)  # remove future data
        .filter(F.col('date') >= start_date)  # remove old data
    )

    item2ind = {val: ind for ind, val in ind2val['itemId'].items()}

    # Indexize clicks for all dataset:
    def indexize_item_array(L):
        if L is None:
            return None
        if len(L) >= 0:
            return [int(item2ind.get(l, 2)) for l in L]
        else:
            return None

    indexize_item_array = F.udf(indexize_item_array, ArrayType(IntegerType()))

    # Indexize displayType for all dataset:
    displaytype2ind = {val: ind for ind, val in ind2val['displayType'].items()}
    def indexize_displaytype_array(L):
        if L is None:
            return None
        if len(L) >= 0:
            return [int(displaytype2ind.get(l, 0)) for l in L]
        else:
            return None

    indexize_displaytype_array = F.udf(indexize_displaytype_array, ArrayType(IntegerType()))

    #%% Indexize actions
    def indexize_item_array_of_array_loc(L):
        newL = []
        for l in L:
            newl = []
            for i in l:
                newi = int(item2ind.get(i, 2))
                newl.append(newi)
            newL.append(newl)
        return newL

    indexize_item_array_of_array = F.udf(indexize_item_array_of_array_loc,
                                         ArrayType(ArrayType(IntegerType())))
    # Test sequence:
    #L = [['149406082', '154407944', '145074295', '154317246'], ['149406082', '154407944', '145074295', '154317246']]
    #indexize_item_array_of_array_loc(L)
    
    ##% INDEXIZE userId
    userId2ind = {val: ind for ind, val in ind2val['userId'].items()}
    indexize_userId = F.udf(lambda x: userId2ind.get(x,0), IntegerType())

    #%% Combine today training data with historical and future click data:
    fulldat = (
        sequences
        .withColumn("userId", indexize_userId("userId"))
        .withColumn('click', indexize_item_array('click'))
        .withColumn('action', indexize_item_array_of_array('action'))
        .withColumn("displayType", indexize_displaytype_array("displayType"))
        )

    logging.info('Collect slates to sequences across days..')
    w = Window.partitionBy('userId').orderBy('date')

    df = fulldat
    event_vars = ['displayType', 'timestamp', 'action', 'click', 'click_idx']
    for col in event_vars:
        df = df.withColumn(col, F.collect_list(col).over(w))
    #%% COLLAPSE INTO ONE ROW PER USER
    sequence_slates = (df.groupby('userId').agg(
        *[F.max(col).alias(col) for col in event_vars]))
    ## flatten into simple lists
    flat_list = lambda l: [item for sublist in l for item in sublist]
    flattenfunc = lambda col, ouput_type: F.udf((flat_list), ouput_type)(col)

    flattened = (
        sequence_slates.withColumn('click', flattenfunc('click', ArrayType(IntegerType())))
        .withColumn('click_idx', flattenfunc('click_idx', ArrayType(IntegerType())))
        .withColumn('action',flattenfunc('action', ArrayType(ArrayType(IntegerType()))))
        .withColumn('displayType',flattenfunc('displayType',ArrayType(IntegerType())))
        .withColumn('timestamp',flattenfunc('timestamp',ArrayType(StringType())))
        )

    ## -- COLLECT AND SAVE
    logging.info('starting collect spark2pandas..')
    dat = DATAHelper.toPandas(flattened)

    logging.info(
        f'Dataset for [{start_date},{end_date}] has {dat.shape[0]} sequences and {len(ind2val["itemId"])} unique items.'
    )
    logging.info("processing data to tensors..:")
    data = construct_data_torch_tensors(dat, maxlen_time=maxlen_time, maxlen_action=maxlen_action)
    logging.info('Save data to files..')
    if not limit:
        save_dir = f'{data_dir}/{data_type}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logging.info('starting saving..')

        # All data
        torch.save(data, f'{save_dir}/data.pt')
        logging.info('saved data.')
    else:
        logging.info('Limit was set, skip saving..')
    logging.info(f'Done saving for [{start_date},{end_date}].')
    return True


# %%
def construct_data_torch_tensors(dat, maxlen_time, maxlen_action):
        logging.info(
            f'Building dataset of {dat.shape[0]} sequences. (timelength, candlength) = ({maxlen_time}, {maxlen_action})'
        )
        dat = dat.reset_index(drop=True)

        action = torch.zeros(
            (len(dat), maxlen_time,
             maxlen_action)).long()  # data_sequence, time_seq, candidates
        click =       torch.zeros(len(dat), maxlen_time).long()  # data_sequence, time_seq
        displayType = torch.zeros(len(dat), maxlen_time).long()  # data_sequence, time_seq

        click_idx = torch.zeros(
            len(dat), maxlen_time).long()  # data_sequence, time_seq
        lengths = torch.zeros((len(dat), maxlen_time)).long()

        userId = torch.tensor(dat.userId.values)

        for i in dat.index:
            # action
            row_action = dat.at[i, 'action'][:maxlen_time]
            obs_time_len = min(maxlen_time, len(row_action))

            lengths[i, :obs_time_len] = torch.tensor(
                [len(l) for l in row_action])

            row_action_pad = torch.from_numpy(
                pad_sequences(row_action[:obs_time_len],
                              maxlen=maxlen_action,
                              padding='post',
                              truncating='post'))
            action[i, :obs_time_len] = row_action_pad

            # Click
            click[i, :obs_time_len] = torch.tensor(
                dat.at[i, 'click'])[:obs_time_len]

            # Click index
            click_idx[i, :obs_time_len] = torch.tensor(
                dat.at[i, 'click_idx'])[:obs_time_len]

            displayType[i,:obs_time_len] = torch.tensor(dat.at[i, 'displayType'])[:obs_time_len]

        ## Set those clicks that were above the maximum candidate set to PAD:
        logging.info(
            f'There are {(click_idx >= maxlen_action).float().sum()} clicks that are above the maxlength action. Setting to click_idx=0 but with click= 0 ("PAD")..'
        )
        click_idx[(click_idx >= maxlen_action)] = 0
        click[(click_idx >= maxlen_action)] = 0

        data = {
            'userId' : userId,
            'lengths': lengths,
            'displayType' : displayType,
            'action': action,
            'click': click,
            'click_idx': click_idx
        }
        return data

#%% DATALOADERS
class SequentialDataset(Dataset):
    '''
     Note: displayType has been uncommented for future easy implementation.
    '''
    def __init__(self, data):

        self.data = data

    def __getitem__(self, idx):
        batch = {key: val[idx] for key, val in self.data.items()}
        return batch

    def __len__(self):
        return len(self.data['click'])


def prepare_dataset(data_dir, data_type):
    logging.info(f'Building dataset for {data_dir} {data_type}.')
    logging.info('Load ind2val..')
    with open(f'{data_dir}/ind2val.pickle', 'rb') as handle:
        ind2val = pickle.load(handle)

    logging.info('Load data..')
    data = torch.load(f'{data_dir}/{data_type}/data.pt')

    dataset = SequentialDataset(data)

    with open(f'{data_dir}/{data_type}/dataset.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=4)


#%% PREPARE DATA IN TRAINING
def load_dataloaders(data_dir,
                     data_type,
                     batch_size=1024,
                     split_trainvalid=0.95,
                     num_workers=0,
                     override_candidate_sampler=None,
                     t_testsplit = 5):

    logging.info('Load data..')
    data = torch.load(f'{data_dir}/{data_type}/data.pt')
    dataset = SequentialDataset(data)
    
    with open(f'{data_dir}/ind2val.pickle', 'rb') as handle:
        ind2val = pickle.load(handle)

    num_testusers = int(len(dataset) * (1-split_trainvalid))
    torch.manual_seed(0)
    num_users = len(dataset)
    perm_user = torch.randperm(num_users)
    valid_user_idx = torch.arange(num_testusers)
    train_user_idx = torch.arange(num_testusers, num_users)
    dataset.data['mask_train'] = torch.ones_like(dataset.data['click'])
    dataset.data['mask_train'][valid_user_idx, t_testsplit:] = 0

    subsets = {'train': dataset, 'valid': torch.utils.data.Subset(dataset, valid_user_idx)}
    dataloaders = {
        phase: DataLoader(ds, batch_size=batch_size, shuffle=True)
        for phase, ds in subsets.items()
    }
    for key, dl in dataloaders.items():
        logging.info(
            f"In {key}: num_users: {len(dl.dataset)}, num_batches: {len(dl)}"
        )


    with open(f'{data_dir}/itemattr.pickle', 'rb') as handle:
        itemattr = pickle.load(handle)

    return ind2val, itemattr, dataloaders


import PYTORCHHelper
def prepare_w2v_from_file_or_web(data_dir, end_date=None):
    model_path = "recommendations-models/FM/fastai-w2v-dummy"
    filepath = f'{data_dir}/w2v_vectors.pickle'

    if end_date is not None:
        Warning('SETTING EPOCH WHEN LOADING W2V FILES IS NOT IMPLEMENTED!!!')
        end_epoch = int(
            datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp() *
            1000)

    if os.path.isfile(filepath):
        logging.info('Load w2v items from local file..')
        with open(filepath, 'rb') as handle:
            ind2item, item2ind, itemvec = pickle.load(handle)
    else:
        logging.info('Load w2v from server..')
        item_model, item2ind = PYTORCHHelper.get_pt_lookup_model(model_path)
        ind2item = {key: val for val, key in item2ind.items()}
        itemvec = item_model.hidden_embedding.weight
        # Normalize:
        itemvec = torch.nn.functional.normalize(itemvec, p=2, dim=1)
        with open(filepath, 'wb') as handle:
            pickle.dump((ind2item, item2ind, itemvec), handle)

    return ind2item, item2ind, itemvec


def get_w2v_item(data_dir, end_date=None):
    '''
    Collects all pretrained itemvectors, 
    and organizes it into a new matrix according to the dict in ind2val['itemId'].
    If there exists no item vector, it will be filled with zeros.
    
    Returns:
    prior dict: {'mean' : torch.(),
                    'prior_exist' : torch.()}
    '''
    # LOAD GLOBAL IND2VAL
    with open(f'{data_dir}/ind2val.pickle', 'rb') as handle:
        ind2val = pickle.load(handle)
    if end_date is not None:
        end_epoch = int(
            datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp() *
            1000)

    w2v_ind2item, w2v_item2ind, w2v_itemvec = prepare_w2v_from_file_or_web(
        data_dir)

    new_lookup = {val: key for key, val in ind2val['itemId'].items()}

    new_itemvec = torch.zeros((len(new_lookup), w2v_itemvec.size()[1]))

    new_itemvec = PYTORCHHelper.remap_embedding(old_embedding=w2v_itemvec,
                                                new_embedding=new_itemvec,
                                                old_lookups=w2v_item2ind,
                                                new_lookups=new_lookup)

    with open(f'{data_dir}/w2v_itemvec.pickle', 'wb') as handle:
        pickle.dump(new_itemvec, handle)
    return new_itemvec

#%%

if __name__ == '__main__':
    sc, sqlContext = FINNHelper.create_spark_cluster(driver_memory='60G',
                                                     max_result_size='16G')
    torch.set_grad_enabled(False)
    param = utils.load_param()

    end_date = param.get('end_date')
    end_datetime = end_date#datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    start_datetime = end_datetime - datetime.timedelta(
        param.get('lookback'))
    start_date = str(start_datetime)

    end_test_datetime = end_datetime + datetime.timedelta(
        param.get('lookforward_test'))
    end_test_date = str(end_test_datetime)

    data_type = param.get('data_type')
    data_path = param.get('data_path')

    logging.info('-' * 20)
    logging.info('PREPARE DATASET..')
    logging.info('-' * 20)
    logging.info(f'Train period: \t [{start_date}, {end_date})')
    logging.info(f'Test period:  \t [{end_date}, {end_test_date})')
    logging.info(f'Data type:    \t {data_type}')

    # make dir if not exist
    mkdir(f'{param.get("data_dir")}/{data_type}')
    ## FIND IND2VAL FOR ITEMS AND THE USERS THAT SHOULD BE INCLUDED
    logging.info("Read sequences..")
    sequences = (
        sqlContext.read.parquet(data_path).withColumnRenamed(
            'inscreen', 'action').filter(
                F.col('date') < end_test_date)  # remove future data
        .filter(F.col('date') >= start_date)  # remove old data
    )

    ## Prepare item indicies
    logging.info('Prepare ind2item and item attributes..')
    ind2val, itemattr = build_global_ind2val(sqlContext,
                                             sequences=sequences,
                                             data_dir=param.get('data_dir'),
                                             drop_groups=param.get(
                                                 'drop_category', False),
                                                 min_item_views = param.get("min_item_views", 100))


    ## Prepare Users
    users = (
        sequences
        .withColumn('clicks_on_day', F.size('click'))
        .groupby('userId')
        .agg(
            F.count('*').alias('unique_days'),
            F.sum('clicks_on_day').alias('tot_clicks')
            )
        .filter(F.col('tot_clicks') >= param.get('min_user_clicks'))
        .select('userId')
        )

    # userID
    unique_users = users.toPandas().values.flatten()
    ind2val['userId'] = {i+1 : val for i, val in enumerate(unique_users)}
    ind2val['userId'][0] = "<UNK>"
    #logging.info(f'There are {len(ind2val["userId"])} users in the dataset.')


    logging.info('Starting on the sequences..')
    sequences = (sequences.join(users, on='userId', how='inner'))
    ## PREPARE DATASETS

    logging.info('-- Prepare train sequences..')
    prepare_sequences(sqlContext,
                      sequences=sequences,
                      start_date=start_date,
                      end_date=end_date,
                      ind2val=ind2val,
                      data_path=data_path,
                      data_dir=param.get('data_dir'),
                      data_type=data_type,
                      maxlen_time=param.get('maxlen_time'),
                      maxlen_action=param.get('maxlen_action')
    )

    #logging.info('-- Prepare pytorch dataset..')
    #prepare_dataset(data_dir=param.get('data_dir'),
    #                data_type=data_type
    #                )

    logging.info('Done prepare.py')