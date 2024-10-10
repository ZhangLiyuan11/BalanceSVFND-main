import torch
from tqdm import tqdm

from model.BSVFND import BSVFNDModel
from utils.dataloader import BSVFNDDataset
from torch.utils.data import DataLoader
from utils.metrics import *
from utils.tools import *

path_fakesv = 'fakesv_checkpoint'
path_fakett = 'fakett_checkpoint'

def load_chechpoint(path,dataset):
    checkpoint_path = 'check_points/' + dataset + '/' + path
    model = BSVFNDModel(fea_dim=128, dropout=0.1,dataset=dataset)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    return model.cuda()


def get_dataloader(dataset):
    if dataset == 'fakesv':
        token = pretrain_bert_wwm_token()
        from utils.dataloader import fakesv_collate_fn as collate_fn
    else:
        token = pretrain_bert_uncased_token()
        from utils.dataloader import fakett_collate_fn as collate_fn
    dataset_test = BSVFNDDataset('vid_time3_test.txt',token,dataset)
    test_dataloader = DataLoader(dataset_test, batch_size=16,
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False,
                                 # worker_init_fn=_init_fn,
                                 collate_fn=collate_fn)
    return test_dataloader


def test():
    dataset = 'fakett'
    if dataset == 'fakesv':
        model = load_chechpoint(path_fakesv,dataset)
    else:
        model = load_chechpoint(path_fakett,dataset)
    test_dataloader = get_dataloader(dataset)
    tpred = []
    tlabel = []
    for batch in tqdm(test_dataloader):
        batch_data = batch
        for k, v in batch_data.items():
            batch_data[k] = v.cuda()
        label = batch_data['label']
        with torch.set_grad_enabled(False):
            outputs = model(**batch_data)
            _,preds = torch.max(outputs, 1)
        tlabel.extend(label.detach().cpu().numpy().tolist())
        tpred.extend(preds.detach().cpu().numpy().tolist())
    results = metrics(tlabel, tpred)
    print(results)


if __name__ == '__main__':
    test()
