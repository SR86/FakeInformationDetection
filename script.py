

import torch as th
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from Process.process import *
from Process.rand5fold import load5foldData
from tools.evaluate import evaluation4class
from model.Twitter.BiGCN_Twitter import Net
import matplotlib.pyplot as plt

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def plot_f1_scores(f1_scores, datasetname):
    classes = ['Real', 'Fake', 'Unverified', 'Other']

    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, f1_scores, color=colors)
    plt.ylim(0, 1.05)
    plt.title(f'F1 Scores per Class - {datasetname}')
    plt.ylabel('F1 Score')

    for bar, score in zip(bars, f1_scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{score:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'outputs/{datasetname}_F1_scores.png')  # 保存图像
    plt.show()

def evaluate_saved_model(model_path, datasetname):
    model = Net(5000, 64, 64).to(device)
    model.load_state_dict(th.load(model_path, map_location=device))
    model.eval()

    fold0_x_test, fold0_x_train, \
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test, fold4_x_train = load5foldData(datasetname)
    treeDic = loadTree(datasetname)

    x_test = fold0_x_test
    x_train = fold0_x_train

    _, testdata_list = loadBiData(datasetname, treeDic, x_train, x_test, TDdroprate=0.2, BUdroprate=0.2)
    test_loader = DataLoader(testdata_list, batch_size=128, shuffle=False)

    all_preds = []
    all_labels = []

    with th.no_grad():
        for batch_data in test_loader:
            batch_data.to(device)
            out = model(batch_data)
            pred = out.max(1)[1]
            all_preds.append(pred.cpu())
            all_labels.append(batch_data.y.cpu())

    preds = th.cat(all_preds)
    labels = th.cat(all_labels)

    Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(preds, labels)

    print("Test Accuracy: {:.4f}".format(Acc_all))
    print("Class 1 F1: {:.4f}, Class 2 F1: {:.4f}, Class 3 F1: {:.4f}, Class 4 F1: {:.4f}".format(F1, F2, F3, F4))

    # 绘制 F1 分数柱状图
    plot_f1_scores([F1, F2, F3, F4], datasetname)

if __name__ == "__main__":
    datasetname = "Twitter15"  # 或者 "Twitter16"
    model_path = "BiGCNTwitter15.pt"
    evaluate_saved_model(model_path, datasetname)





