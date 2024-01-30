
# 导入所需的库
import pandas as pd # 用于处理数据的工具
import numpy as np
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from clac_dis_mesh_sim import cal_SimilarityByMeSHDAG
from alive_progress import alive_bar # 显示循环的进度条工具
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard
from sklearn.metrics import jaccard_score

'''适合M*N矩阵，列名有意义'''
def data_label(data:pd.DataFrame, screen_list: list=None,screen_col: str=None) -> pd.DataFrame:
    data = data.copy() # 复制数据，避免后续影响原始数据。
    if len(screen_list)>0: # 如果提供了 pred_labels 参数，则执行该代码块。
        data=pd.concat([data.loc[data[screen_col]==id] for id in screen_list],ignore_index=True)
        data=data.sort_values(by=screen_col,ascending=[True])#药物排序
        data = data.drop(columns=[screen_col]) # 去掉用于筛选药物的列。
        data=data.sort_index(axis=1,ascending=True)#ADR(列名)排序（需要先去掉筛选药物的列，否则也参与排序）
    elif screen_col:
        data=data.sort_values(by=screen_col,ascending=[True])
        data = data.drop(columns=[screen_col]) # 去掉用于筛选药物的列。
        data=data.sort_index(axis=1,ascending=True)#ADR(列名)排序
    
    return data # 返回最后处理的数据。

'''适合列表/以列表展示的特征'''
def data_feature(data: pd.DataFrame, screen_list: list=None,screen_col: str=None, del_screen_col=True) -> pd.DataFrame:
    
    data = data.copy() # 复制数据，避免后续影响原始数据。 
    if len(screen_list)>0: # 如果提供了 pred_labels 参数，则执行该代码块。
        data=pd.concat([data.loc[data[screen_col]==id] for id in screen_list],ignore_index=True)
        data=data.sort_values(by=screen_col,ascending=[True])
        if del_screen_col:  data = data.drop(columns=[screen_col]) # 去掉用于筛选药物/ADR的列。
    elif screen_col:
        data=data.sort_values(by=screen_col,ascending=[True])
        if del_screen_col:  data = data.drop(columns=[screen_col]) # 去掉用于筛选药物/ADR的列。

    return data # 返回最后处理的数据。

def jaccard_similarity(X, Y=None):
    if Y is None:
        Sim=1-pdist(X,"jaccard")
        Sim = squareform(Sim)
        Sim = Sim + np.eye(X.shape[0])
        Sim[np.isnan(Sim)] = 0
    else:
        Sim=jaccard_score(X,Y)
    return Sim

def Convert_triplelist2matrix(data: pd.DataFrame,pivot_cols=[],fillna_val=0.0):
    data = data.copy() # 复制数据，避免后续影响原始数据。
    mat_data=data.pivot(pivot_cols[0],pivot_cols[1],pivot_cols[2])  
    mat_data=mat_data.sort_index(axis=0,ascending=True)
    mat_data=mat_data.sort_index(axis=1,ascending=True)
    mat_data=mat_data.fillna(fillna_val)
    return mat_data

def cal_drug_similarityBySmiles(SMILES1,SMILES2):
    mol1=Chem.MolFromSmiles(SMILES1)
    mol2=Chem.MolFromSmiles(SMILES2)
    #Morgan Fingerprints
    fingerprint1=AllChem.GetMorganFingerprint(mol1,2)
    fingerprint2=AllChem.GetMorganFingerprint(mol2,2)

    fps1=Chem.RDKFingerprint(mol1)
    fps2=Chem.RDKFingerprint(mol2)
    return DataStructs.FingerprintSimilarity(fps1,fps2) 

def get_SMILES_Similarity(data: pd.DataFrame):
    print("开始计算SMILES相似性，总数：",str(data.shape[0]))
    smiles_sim_mat = np.zeros([data.shape[0],data.shape[0]], dtype = float, order = 'C')
    with alive_bar(data.shape[0]) as bar:
        for i in  range(data.shape[0]):
            bar()
            Smile_i=data.loc[i]["SMILES"]
            for j in range(i):
                Smile_j = data.loc[j]["SMILES"]
                smiles_sim_mat[i][j]=cal_drug_similarityBySmiles(Smile_i,Smile_j)
    smiles_sim_mat = smiles_sim_mat + smiles_sim_mat.T + np.eye(data.shape[0]) 
    return smiles_sim_mat


def get_MESH_Similarity(data: pd.DataFrame):
    print("开始计算MESH相似性，总数：",str(data.shape[0]))
    Mesh_sim_mat = np.zeros([data.shape[0],data.shape[0]], dtype = float, order = 'C')
    with alive_bar(data.shape[0]) as bar:
        for i in  range(data.shape[0]):
            bar()
            Dis_i=eval(data.loc[i]["Dict_MESH"])
            for j in range(i):
                Dis_j = eval(data.loc[j]["Dict_MESH"])
                Mesh_sim_mat[i][j]=cal_SimilarityByMeSHDAG(Dis_i,Dis_j)
    Mesh_sim_mat = Mesh_sim_mat + Mesh_sim_mat.T + np.eye(data.shape[0]) 
    return Mesh_sim_mat


def split_train_test(drug_features, side_features, data_train, data_test):
    data_train = np.array(data_train)
    data_test = np.array(data_test)

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    drug_test = drug_features_matrix[data_test[:, 0]]
    side_test = side_features_matrix[data_test[:, 1]]
    f_test = data_test[:, 2]

    drug_train = drug_features_matrix[data_train[:, 0]]
    side_train = side_features_matrix[data_train[:, 1]]
    f_train = data_train[:, 2]

    return drug_test, side_test, f_test, drug_train, side_train, f_train
