import torch as th
from torch_geometric.data import DataLoader
from Process.process import loadBiData, loadTree
from Twitter.GCN_Twitter import Net
import os
from Process.dataset import GraphDataset,BiGraphDataset,UdGraphDataset
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
cwd=os.getcwd()


################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
        treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))
    return treeDic

################################# load data ###################################
def loadData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data', dataname+'graph')
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadUdData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data',dataname+'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    data_path = os.path.join(cwd,'data', dataname + 'graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list




# def evaluate_single_input(model, input_data, treeDic, datasetname):
#     # 模拟从输入数据构建图数据
#     _, testdata_list = loadBiData(datasetname, treeDic, input_data, input_data, TDdroprate=0.2, BUdroprate=0.2)
#     test_loader = DataLoader(testdata_list, batch_size=1, shuffle=False)
#
#     model.eval()
#     with th.no_grad():
#         for batch_data in test_loader:
#             batch_data.to(device)
#             out = model(batch_data)
#             pred = out.max(1)[1]
#             return pred.item()
#
# def test_model_with_input(model_path, datasetname, input_data):
#     # 加载模型
#     model = Net(5000, 64, 64).to(device)
#     model.load_state_dict(th.load(model_path, map_location=device))
#
#     # 加载树数据
#     treeDic = loadTree(datasetname)
#
#     # 获取模型预测
#     prediction = evaluate_single_input(model, input_data, treeDic, datasetname)
#     print(f"Prediction for the input data: {prediction}")

def evaluate_single_input(model, input_data, treeDic, datasetname):
    # 从数据集中加载图数据
    _, testdata_list = loadBiData(datasetname, treeDic, input_data, input_data, TDdroprate=0.2, BUdroprate=0.2)
    test_loader = DataLoader(testdata_list, batch_size=1, shuffle=False)

    model.eval()  # 设置为评估模式
    with th.no_grad():  # 不需要计算梯度
        for batch_data in test_loader:
            batch_data = batch_data.to(device)  # 将数据送到设备上
            out = model(batch_data)  # 模型输出
            # 输出图数据的相关信息，例如节点特征、边的信息等
            print(f"Batch data: {batch_data}")
            print(f"Node features: {batch_data.x}")  # 节点特征
            print(f"Edge index: {batch_data.edge_index}")  # 边的索引
            print(f"Edge attributes: {batch_data.edge_attr}")  # 边的特征（如果有）

            pred = out.max(1)[1]  # 获取最大预测类别
            return pred.item()

def test_model_with_input(model_path, datasetname, input_data):
    # 加载模型
    model = Net(5000, 64, 64).to(device)
    model.load_state_dict(th.load(model_path, map_location=device))

    # 加载树数据
    treeDic = loadTree(datasetname)

    # 获取模型预测
    prediction = evaluate_single_input(model, input_data, treeDic, datasetname)
    print(f"Prediction for the input data: {prediction}")


if __name__ == "__main__":
    model_path = "GCNTwitter.pt"
    datasetname = "Twitter15"
    input_data = ["549964632883232768"]  # Replace with actual data
    test_model_with_input(model_path, datasetname, input_data)






