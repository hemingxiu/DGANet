# 导入所需的库
import os
import sys
import os.path as osp
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
import time
# 用于处理数据的工具
import random
import pandas as pd 
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error # 评分 MAE 的计算函数
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split # 拆分训练集与验证集工具
from data_utils import *
from model import DGAPred
from clac_dis_mesh_sim import cal_SimilarityByMeSHDAG
import pickle
#from alive_progress import alive_bar # 显示循环的进度条工具
# import sweetviz as sv
import warnings

def load_label(screen_drug_list,use_DGen,use_AGen,args):
    pd_label=pd.read_csv(args.rawpath+"sider_pert_mesh_list.csv",header=0,delimiter='\t') # 原始训练数据
    drug_col="pert_id"
    adr_col="MESH_ID"
    drug_side=data_feature(pd_label,screen_list=screen_drug_list,screen_col=drug_col,del_screen_col=False)
    #筛选有CTD GENE特征的Drug
    if use_DGen:
        pd_DGen=pd.read_csv(args.rawpath+"ctd_chem_pert_gene_ixns_list.csv",header=0,delimiter='\t')
        drug_list_DGen=sorted(np.unique(pd_DGen.loc[:,drug_col].values).tolist())
        drug_side=pd.concat([drug_side.loc[drug_side[drug_col]==id] for id in drug_list_DGen],ignore_index=True)
    drug_list=sorted(np.unique(drug_side.loc[:,"pert_id"].values).tolist())
    #筛选有CTD GENE特征的ADR
    if use_AGen:
        pd_AGen=pd.read_csv(args.rawpath+"ctd_gene_adr_asso_list_4386.csv",header=0,delimiter='\t')
        adr_list_Gen=sorted(np.unique(pd_AGen.loc[:,adr_col].values).tolist())
        drug_side=pd.concat([drug_side.loc[drug_side[adr_col]==id] for id in adr_list_Gen],ignore_index=True)

    drug_side=Convert_triplelist2matrix(drug_side,["pert_id","MESH_ID","label"],fillna_val=0) 
    adr_list = list(drug_side.columns)
    return drug_list,adr_list,drug_side

def load_drug_feature(screen_drug_list,args):
    drug_features = []
    #CS Feature
    pd_cs=pd.read_csv(args.rawpath+"drug_pert_similes_list.csv",header=0,delimiter='\t')
    drug_col="pert_id"
    drug_smiles=data_feature(pd_cs,screen_list=screen_drug_list,screen_col=drug_col)
    drug_cs=get_SMILES_Similarity(drug_smiles)
    print('drug CS shape:',drug_cs.shape)

    #CTD DGEN Feature
    pd_DGen=pd.read_csv(args.rawpath+"ctd_chem_pert_gene_ixns_list.csv",header=0,delimiter='\t')
    drug_col="pert_id"
    drug_DGen=data_feature(pd_DGen,screen_list=screen_drug_list,screen_col=drug_col,del_screen_col=False)
    # drug_DGen=pd.concat([drug_DGen['pert_id'],drug_DGen['GeneSymbol']]).unique()    #去除Action列，统一为1：存在相互作用
    drug_DGen=drug_DGen.drop_duplicates(subset=['pert_id','GeneSymbol'],keep='first')    #去除Action列，统一为1：存在相互作用
    #drug_DGen=pd.DataFrame(drug_DGen).insert(loc=2,column="ixn",value=1)                 #1:药物-基因存在相互作用
    drug_DGen["ixn"]=1
    drug_DGen=Convert_triplelist2matrix(drug_DGen,["pert_id","GeneSymbol","ixn"],fillna_val=0)
    # pd.DataFrame(drug_DGen).to_csv("mat_drug_DGen.csv",header=True,index=True)
    drug_DGen_sim = jaccard_similarity(drug_DGen)
    # pd.DataFrame(drug_DGen_sim).to_csv("mat_drug_DGen.csv",header=True,index=True)
    print('drug DGen shape:',drug_DGen_sim.shape)

    #LINCS GE Feature
    pd_ge=pd.read_csv(args.rawpath+"LINCS_Gene_Experssion_signatures_CD.csv",header=0,delimiter=',')
    drug_col="pert_id"
    drug_ge=data_feature(pd_ge,screen_list=screen_drug_list,screen_col=drug_col)
    drug_ge_sim = cosine_similarity(drug_ge)
    print('drug GE shape:',drug_ge_sim.shape)

    drug_features.append(drug_cs)
    drug_features.append(drug_DGen_sim)
    drug_features.append(drug_ge_sim)
    print(f"{str(time.strftime('%Y-%m-%d, %H:%M:%S'))}=Drug Feature:CS+CGI+GE")
    return drug_features

def load_adr_feature(screen_adr_list,args):
    side_features = []
    adr_list=pd.DataFrame(screen_adr_list,columns=["MESH_ID"])  #格式：MESH:D002311
    adr_list_id=adr_list["MESH_ID"].str.replace("MESH:","")     #格式：D002311
    #MESH feature
    pd_label=pd.read_csv(args.rawpath+"se_mesh_dict_list.csv",header=0,delimiter='\t') # 原始训练数据
    adr_col="MESH_ID"
    side_mesh=data_feature(pd_label,screen_list=adr_list_id.values,screen_col=adr_col,del_screen_col=False)
    side_mesh_sim=get_MESH_Similarity(side_mesh)

    #CTD Gene-Disease Feature
    pd_GDisease=pd.read_csv(args.rawpath+"ctd_gene_adr_asso_list_4386.csv",header=0,delimiter='\t')    #原数据已去重
    adr_GDisease=data_feature(pd_GDisease,screen_list=adr_list["MESH_ID"],screen_col=adr_col,del_screen_col=False)
    adr_GDisease=Convert_triplelist2matrix(adr_GDisease,["MESH_ID","GeneSymbol","ixn"],fillna_val=0)
    print(pd.DataFrame(adr_GDisease).shape)
    # pd.DataFrame(drug_DGen).to_csv("mat_drug_DGen.csv",header=True,index=True)
    adr_GDisease_sim = jaccard_similarity(adr_GDisease)
    
    side_features.append(side_mesh_sim)
    side_features.append(adr_GDisease_sim)
    print(f"{str(time.strftime('%Y-%m-%d, %H:%M:%S'))}=Drug Feature:MESH+GEN")
    return side_features

def Extract_positive_negative_samples(DAL, addition_negative_number='all'):
    k = 0
    interaction_target = np.zeros((DAL.shape[0]*DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    number_negative = interaction_target.shape[0] - number_positive
    final_positive_sample = data_shuffle[number_negative::]
    negative_sample = data_shuffle[0:number_negative]
    a = np.arange(number_negative)#number_negative
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (number_negative)) #随机抽样N次/打乱顺序
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]##取和正样本一样多的负样本0~n
    addition_negative_sample = negative_sample[b[number_positive::], :]#除final_negative_sample之外的负样本
    return addition_negative_sample, final_positive_sample, final_negtive_sample


def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    '''
    稀疏多标签交叉熵损失的torch实现
    '''
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[...,:1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss



def train_test(drug_feature,side_feature,data_train, data_test, data_neg, fold, args):
    drug_test, side_test, f_test, drug_train, side_train, f_train = split_train_test(drug_feature,side_feature,data_train, data_test)
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
                                              torch.FloatTensor(f_train))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
                                             torch.FloatTensor(f_test))
    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=16, pin_memory=True)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        num_workers=16, pin_memory=True)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = DGAPred(drug_feature[0].shape[0]*4, side_feature[0].shape[0]*3, args.embed_dim, args.batch_size).to(device)
    # Regression_criterion = nn.MSELoss()
    Classification_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    AUC_mn = 0
    AUPR_mn = 0

    rms_mn = 100000
    mae_mn = 100000
    endure_count = 0

    start = time.time()
    train_epoches = []
    test_epoches = []
    for epoch in range(1, args.epochs + 1):
        """epoch开始"""
        # ====================   training    ====================
        iter_loss_sum, step = train(model, _train, optimizer, Classification_criterion, device) # 一个iterater
        train_epoch = iter_loss_sum/step
        train_epoches.append(train_epoch)
        # ====================     test       ====================

        t_i_auc, t_iPR_auc, t_rmse, t_mae, t_ground_i, t_ground_u, t_ground_truth, t_pred1, t_pred2, test_iter_loss, test_step = test(model,
                                                                                                           _train,
                                                                                                           _train,
                                                                                                           device,
                                                                                                           lossfunction1=Classification_criterion)
        test_epoch = test_iter_loss/test_step
        test_epoches.append(test_epoch)


        if AUC_mn < t_i_auc and AUPR_mn < t_iPR_auc:
            AUC_mn = t_i_auc
            AUPR_mn = t_iPR_auc
            rms_mn = t_rmse
            mae_mn = t_mae
            endure_count = 0

        else:
            endure_count += 1

        print("Epoch: %d <Train> RMSE: %.5f, MAE: %.5f, AUC: %.5f, AUPR: %.5f " % (
        epoch, t_rmse, t_mae, t_i_auc, t_iPR_auc))
        start = time.time()

        if endure_count > 10:
            break
    i_auc, iPR_auc, rmse, mae, ground_i, ground_u, ground_truth, pred1, pred2, test_avg_loss, step_ = test(model, _test, _test, device,lossfunction1=Classification_criterion)

    time_cost = time.time() - start
    print("Time: %.2f Epoch: %d <Test> RMSE: %.5f, MAE: %.5f, AUC: %.5f, AUPR: %.5f " % (
        time_cost, epoch, rmse, mae, i_auc, iPR_auc))
    print('The best AUC/AUPR: %.5f / %.5f' % (i_auc, iPR_auc))
    print('The best RMSE/MAE: %.5f / %.5f' % (rmse, mae))

    ''' save model'''
    with open(os.path.join(args.rawpath,f'model_fold{str(fold)}.pkl'),'wb') as f:
        pickle.dump(model.state_dict(), f)
    ''' save test data'''
    with open(os.path.join(args.rawpath,f'testdata_fold{str(fold)}.pkl'),'wb') as f:
        # test_data={"ground_truth":ground_u,"pred_value":pred1}
        test_data={"ground_truth":ground_truth,"pred_value":pred1}
        pickle.dump(test_data, f)
    # plt.switch_backend('Agg')
    # fig = plt.figure()
    # pic1 = fig.add_subplot(2,1,1)
    # pic2 = fig.add_subplot(2,1,2)
    # pic1.plot(train_epoches,"skyblue",label="train_loss")
    # pic2.plot(test_epoches,"pink",label="test_loss")
    # pic1.legend()
    # pic2.legend()
    # pic1.set_ylabel("loss")
    # pic2.set_xlabel("epoch")
    # pic2.set_ylabel("loss")
    # plt.savefig(os.path.join(args.rawpath,"loss_curve.jpg"))

    return i_auc, iPR_auc, rmse, mae

def train(model, train_loader, optimizer, lossfunction1, device):

    model.train()
    avg_loss = 0.0

    for step, data in enumerate(train_loader, 0):
        batch_drug, batch_side, batch_ratings = data
        batch_labels = batch_ratings.clone().float()
        for k in range(batch_ratings.data.size()[0]):
            if batch_ratings.data[k] > 0:
                batch_labels.data[k] = 1
        optimizer.zero_grad()

        logits = model(batch_drug, batch_side, device)
        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_labels.to(device),y_pred=logits,mask_zero=True)
        total_loss = loss1
        total_loss.backward(retain_graph = True)
        optimizer.step()
        avg_loss += total_loss.item()

    return avg_loss, step

def test(model, test_loader, neg_loader, device, lossfunction1):
    model.eval()
    pred1 = []
    pred2 = []
    ground_truth = []
    label_truth = []
    ground_u = []
    ground_i = []
    test_avg_loss = 0.0
    for step, (test_drug, test_side, test_ratings) in enumerate(test_loader):

        test_labels = test_ratings.clone().long()
        for k in range(test_ratings.data.size()[0]):
            if test_ratings.data[k] > 0:
                test_labels.data[k] = 1
        ground_i.append(list(test_drug.data.cpu().numpy()))
        ground_u.append(list(test_side.data.cpu().numpy()))
        test_u, test_i, test_ratings = test_drug.to(device), test_side.to(device), test_ratings.to(device)
        scores_one, scores_two = model(test_drug, test_side, device)
        """Loss"""
        one_label_index = np.nonzero(test_labels.data.numpy())
        loss1 = lossfunction1(scores_one, test_labels.to(device))
        test_loss = loss1
        test_avg_loss += test_loss.detach().item()
        """"""
        pred1.append(list(scores_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data.cpu().numpy()))
        ground_truth.append(list(test_ratings.data.cpu().numpy()))
        label_truth.append(list(test_labels.data.cpu().numpy()))

    pred1 = np.array(sum(pred1, []), dtype = np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)

    ground_truth = np.array(sum(ground_truth, []), dtype = np.float32)
    label_truth = np.array(sum(label_truth, []), dtype=np.float32)


    iprecision, irecall, ithresholds = metrics.precision_recall_curve(label_truth,
                                                                      pred1,
                                                                      pos_label=1,
                                                                      sample_weight=None)
    iPR_auc = metrics.auc(irecall, iprecision)

    try:
        i_auc = metrics.roc_auc_score(label_truth, pred1)
    except ValueError:
        i_auc = 0

    one_label_index = np.nonzero(label_truth)
    rmse = sqrt(mean_squared_error(pred2[one_label_index], ground_truth[one_label_index]))
    mae = mean_absolute_error(pred2[one_label_index], ground_truth[one_label_index])


    return i_auc, iPR_auc, rmse, mae, ground_i, ground_u, ground_truth, pred1, pred2, test_avg_loss, step


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 200,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0005,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.00001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--N', type = int, default = 30000,
                        metavar = 'N', help = 'L0 parameter')
    parser.add_argument('--droprate', type = float, default = 0.5,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--use_drugside', type = bool, default = True,
                        help = 'add drug-side similarity')
    parser.add_argument('--rawpath', type=str, default='./data/',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()
    # 数据准备
    druglist = pd.read_csv(args.rawpath+"lincs_druglist_ge_go_521.csv") # 筛选实验药物
    
    remain_drug_list,adr_list, drug_side = load_label(druglist["pert_id"],True,True, args)
    print("drug_side shape:",pd.DataFrame(drug_side).shape)
    # pd.DataFrame(drug_side).to_csv("drug_side.csv",header=True,index=True)
    # drug_side_report=sv.analyze(drug_side)
    # drug_side_report.show_html(filepath=f"drug_side_report{str(time.strftime('%Y%m%d%H%M'))}.html",open_browser=False)
    # adr_list=pd.DataFrame(adr_list,columns=["MESH_ID"])

    #不参与训练的负样本，len(final_positive_sample)=len(final_negative_sample)
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side.values, addition_negative_number='all')
    final_sample = np.vstack((final_positive_sample, final_negative_sample))#均衡抽样
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1], addition_negative_sample[i, 2]))
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    drug_feature=load_drug_feature(remain_drug_list, args)
    side_feature=load_adr_feature(adr_list, args)

    
    fold = 1
    kfold = StratifiedKFold(5, random_state=1, shuffle=True)
    total_auc, total_pr_auc, total_rmse, total_mae = [], [], [], []
    for k, (train_split, test_split) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))
        data = np.array(data)

        #增加Drug-Side相似性
        if args.use_drugside:
            drug_side_forsim = drug_side.values
            drug_side_forsim[data[test_split][:,0],data[test_split][:,1]] = 0   #将测试集数据设为0（假设为未知不良反应）
            drug_side_sim = jaccard_similarity(drug_side_forsim)
            side_drug_sim = jaccard_similarity(drug_side_forsim.T)
            if fold ==1 :
                drug_feature.append(drug_side_sim)
                side_feature.append(side_drug_sim)
            else:
                drug_feature[len(drug_feature)-1]=drug_side_sim
                side_feature[len(side_feature)-1]=side_drug_sim
        
        auc, PR_auc, rmse, mae = train_test(drug_feature,side_feature,data[train_split].tolist(), data[test_split].tolist(), data_neg, fold, args)

        total_rmse.append(rmse)
        total_mae.append(mae)
        total_auc.append(auc)
        total_pr_auc.append(PR_auc)
        print("==================================fold {} end".format(fold))#每个Fold之后输出指标值平均值
        fold += 1
        print('Total_AUC:')
        print(np.mean(total_auc))
        print('Total_AUPR:')
        print(np.mean(total_pr_auc))
        print('Total_RMSE:')
        print(np.mean(total_rmse))
        print('Total_MAE:')
        print(np.mean(total_mae))
        sys.stdout.flush()