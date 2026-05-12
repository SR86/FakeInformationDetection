# bertevaluate.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import string
from tqdm import tqdm

# 0. GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# 1. 文本预处理
def text_preprocessing(text):
    if not isinstance(text, str):
        text = ''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation + "–—−±×÷"), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'reuters', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text

# 2. 加载测试数据
def load_test_data(test_path, num_samples=None):
    test_data = pd.read_csv(test_path, encoding='utf-8')

    if num_samples is not None and num_samples < len(test_data):
        test_data = test_data.sample(n=num_samples, random_state=42).reset_index(drop=True)

    texts = test_data['text'].apply(text_preprocessing).tolist()
    labels = test_data['target'].tolist()
    return texts, labels

# 3. 加载模型和 tokenizer
def load_model_and_tokenizer(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer




# 4. 评估函数（支持更多指标 + 绘图）
def evaluate(model, tokenizer, texts, labels):
    model.eval()
    predictions = []
    probs = []
    batch_size = 16
    num_batches = (len(texts) + batch_size - 1) // batch_size  # 计算总批次数

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating", ncols=100):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())  # for ROC

    # 计算各种指标
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print(f"\nEvaluation Results:")
    print(f"Accuracy:  {acc * 100:.4f}%")
    print(f"Precision: {precision * 100:.4f}%")
    print(f"Recall:    {recall * 100:.4f}%")
    print(f"F1 Score:  {f1 * 100:.4f}%")

    print("\nClassification Report:")
    print(classification_report(labels, predictions))

    os.makedirs("outputs/plots", exist_ok=True)

    # 绘制并保存混淆矩阵
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, classes=["Real", "Fake"], save_path="outputs/plots/confusion_matrix.png")

    # 绘制并保存ROC曲线
    plot_roc_curve(labels, probs, save_path="outputs/plots/roc_curve.png")


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, save_path=None):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(labels, probs, save_path=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # 配置路径
    model_dir = "./model/bert_lense"   # 模型目录
    test_path = "./data/ISOT dataset/test.csv"       # 测试集csv
    # num_samples = 1000                  # 随机抽取1000条数据测试
    num_samples = 2186

    texts, labels = load_test_data(test_path, num_samples=num_samples)
    model, tokenizer = load_model_and_tokenizer(model_dir)
    evaluate(model, tokenizer, texts, labels)





# if __name__ == "__main__":
#     model_dir = "model/bert_lense"  # 模型目录
#     model, tokenizer = load_model_and_tokenizer(model_dir)
#
#     while True:
#         user_text = input("请输入要检测的文本（输入 q 退出）：\n")
#         if user_text.lower() == 'q':
#             break
#
#         cleaned_text = text_preprocessing(user_text)
#         inputs = tokenizer([cleaned_text], return_tensors='pt', truncation=True, padding=True, max_length=512)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#
#         model.eval()
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             pred = torch.argmax(logits, dim=1).item()
#             prob = torch.softmax(logits, dim=1)[0][1].item()
#
#         label_name = "Fake" if pred == 1 else "Real"
#         print(f"\n预测结果: {label_name}（置信度 {prob * 100:.2f}%）\n")
