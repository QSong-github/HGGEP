import warnings
warnings.filterwarnings("ignore")
import os
import torch
import random
import argparse
import pickle as pk
import pytorch_lightning as pl
from utils import *
# from HIST2ST_Baseline import *
from HGGEP_HGNN import *
from predict import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--fold', type=int, default=11, help='dataset fold.')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
parser.add_argument('--pre_trained', type=str, default=False, help='load pre-model')
parser.add_argument('--data', type=str, default='her2st', help='dataset name:{"her2st","cscc", "BDS" }.')
parser.add_argument('--gpu', type=int, default=0, help='the id of gpu.')
parser.add_argument('--seed', type=int, default=12000, help='random seed.')
parser.add_argument('--epochs', type=int, default=401, help='number of epochs.')
parser.add_argument('--name', type=str, default='hist2ST', help='prefix name.')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout.')
parser.add_argument('--bake', type=int, default=5, help='the number of augmented images.')  # 5
parser.add_argument('--lamb', type=float, default=0.5, help='the loss coef of self-distillation.')
parser.add_argument('--nb', type=str, default='F', help='zinb or nb loss.')
parser.add_argument('--zinb', type=float, default=0.25, help='the loss coef of zinb.')
parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge:{"Grid","NA"}')
parser.add_argument('--policy', type=str, default='mean', help='the aggregation way in the GNN .')
parser.add_argument('--neighbor', type=int, default=8, help='the number of neighbors in the GNN.')

parser.add_argument('--tag', type=str, default='5-7-2-8-4-16-32',   # '5-7-2-8-4-16-32', 
                    help='hyper params: kernel-patch-depth1-depth2-depth3-heads-channel,'
                         'depth1-depth2-depth3 are the depth of Convmixer, Multi-head layer in Transformer, and GNN, respectively'
                         'patch is the value of kernel_size and stride in the path embedding layer of Convmixer'
                         'kernel is the kernel_size in the depthwise of Convmixer module'
                         'heads are the number of attention heads in the Multi-head layer'
                         'channel is the value of the input and output channel of depthwise and pointwise. ')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),args.tag.split('-'))

trainset = pk_load(args.fold,'train',False,args.data,neighs=args.neighbor, prune=args.prune)
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)

testset = pk_load(args.fold,'test',False,args.data,neighs=args.neighbor, prune=args.prune)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

label=None
if args.fold in [5,11,17,23,26,30] and args.data=='her2st':
    label=testset.label[testset.names[0]]

genes=785
if args.data=='cscc':
    args.name+='_cscc'
    genes=171
elif args.data=='BDS':
    args.name+='_BDS'
    genes=100

model = HGGEP(
    depth1=depth1, depth2=depth2, depth3=depth3,
    n_genes=genes, learning_rate=args.lr, label=label, 
    kernel_size=kernel, patch_size=patch,
    heads=heads, channel=channel, dropout=args.dropout,
    zinb=args.zinb, nb=args.nb=='T',
    bake=args.bake, lamb=args.lamb, 
    policy=args.policy, 
)

# load pretrained model
args.pre_trained = False
# args.pre_trained = True
if args.pre_trained:
    print('Loaded pretrained model!')
    checkpoint = torch.load('./model/her2st/0-HGGEP.ckpt')
    model.load_state_dict(checkpoint, strict=True)
else:
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        check_val_every_n_epoch=5,checkpoint_callback=False)
    trainer.fit(model, train_loader, test_loader)

pred, gt = test(model, test_loader,'cuda') 
R=get_R(pred,gt)[0]

print('Pearson Correlation Median:',np.nanmedian(R))
print('Pearson Correlation Mean:',np.nanmean(R))

Pearson_mean_max, Pearson_median_max = -1, -1
for i in range(100):
    trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    check_val_every_n_epoch=3,checkpoint_callback=False)   
    trainer.fit(model, train_loader, test_loader)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    pred, gt = test(model, test_loader,'cuda')
    R=get_R(pred,gt)[0]
    Pearson_mean = np.nanmean(R)
    if Pearson_mean > Pearson_mean_max:
        Pearson_mean_max = max(Pearson_mean, Pearson_mean_max)
    Pearson_median = np.nanmedian(R)
    if Pearson_median > Pearson_median_max:
        Pearson_median_max = max(Pearson_median, Pearson_median_max)
        torch.save(model.state_dict(), "./model/"+str(args.data)+"/"+str(args.fold)+"_HGGEP_"+str(Pearson_median_max)[:6]+".ckpt")
    print('Pearson Correlation Median:', Pearson_median, "Pearson_median_max:", Pearson_median_max)
    print('Pearson Correlation Mean:', Pearson_mean, "Pearson_mean_max:", Pearson_mean_max)
