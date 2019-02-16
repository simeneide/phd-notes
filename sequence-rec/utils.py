import torch
import numpy as np

def get_device(device = None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def torch_unique_ordered(seq, K, device):
    return torch.tensor(numpy_unique_ordered(seq.cpu().numpy())[:K]).to(device)

def numpy_unique_ordered(seq):
    """Remove duplicate from a list while keeping order with Numpy.unique
    Required:
        seq: A list containing all items
    Returns:
        A list with only unique values. Only the first occurence is kept.
    """
    array_unique = np.unique(seq, return_index=True)
    dstack = np.dstack(array_unique)
    dstack.dtype = np.dtype([('v', dstack.dtype), ('i', dstack.dtype)])
    dstack.sort(order='i', axis=1)
    return dstack.flatten()['v']#.tolist()

def compute_hitrate(seq, num_recs = 20, tau = 6):
    posterior_predictive = seqrec.guide(seq)
    feature_seq = seq[:,:tau]
    test_seq = seq[:,tau:]
    
    lprob = posterior_predictive(feature_seq)
    topK = lprob.argsort(dim=2, descending=True)[:,-1,:num_recs]

    hitrate = torch.tensor(0.0)
    totrate = torch.tensor(0.0)
    for i in range(len(seq)):
        hitmatrix = topK[i].unsqueeze(1) == test_seq[i].unsqueeze(0)
        hitrate += hitmatrix.sum()
        totrate += (test_seq[i] != 0).sum()
        
    return (hitrate/totrate).item()

def compute_bayesian_hitrate(seqrec, seq, num_recs = 20, tau = 6, num_samples = None, device = None):
    if device is None:
        device = get_device()
    
    feature_seq = seq[:,:tau]
    test_seq = seq[:,tau:]
    
    if num_samples is None:
        num_samples = num_recs
    # SAMPLE ALL POSTERIORS
    topK_samples = []
    for i in range(num_samples):
        posterior_predictive = seqrec.guide(feature_seq)
        pp = posterior_predictive(feature_seq)[:,-1]
        topK = pp.argsort(dim=1, descending=True)[:,:num_recs] # swap with torch.topk
        topK_samples.append(topK.unsqueeze(2))

    all_recs = torch.cat(topK_samples,2) # dim : (batch, topK, samples)
    # sum up across users:
    hitrate = torch.tensor(0.0)
    totrate = torch.tensor(0.0)
    misrate = torch.tensor(0.0)
    for i in range(len(seq)):
        onerec = all_recs[i,]
        recs = torch_unique_ordered(onerec.flatten(), K = num_recs, device = device)
        hitmatrix = recs.unsqueeze(1) == test_seq[i].unsqueeze(0)
        misrate += num_recs - len(recs)
        hitrate += hitmatrix.sum()
        totrate += (test_seq[i] != 0).sum() # all that are not the padding index count
    if misrate > 0:
        print("Found positive misrate!")
        print(hitrate, totrate, misrate)
    return (hitrate/totrate).item() #, all_recs, test_seq