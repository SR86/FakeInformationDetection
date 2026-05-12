import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datasets import Dataset
import os
import string
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import EarlyStoppingCallback

# 0. GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


def load_data(train_path, test_path, num_samples=None):
    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data = pd.read_csv(test_path, encoding='utf-8')

    # 如果 num_samples 不为 None，则随机选择 num_samples 个样本
    if num_samples:
        train_data = train_data.sample(n=num_samples, random_state=42)
        test_data = test_data.sample(n=num_samples, random_state=42)

    train_texts = train_data['text'].tolist()
    train_labels = train_data['target'].tolist()
    test_texts = test_data['text'].tolist()
    test_labels = test_data['target'].tolist()

    return train_texts, test_texts, train_labels, test_labels


def tokenize_data(texts, tokenizer, max_length=512):
    if isinstance(texts, list):
        texts = [str(text) if text is not None else "" for text in texts]
    else:
        texts = str(texts) if texts is not None else ""

    return tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length)


# 2. Text Preprocessing
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


# 3. Load Model and Tokenizer
def load_model_and_tokenizer(model_dir, model_class):
    model = model_class.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


# 4. Train BERT-based model
def train_bert(llm_name, train_texts, train_labels, test_texts, test_labels, epochs, fine_tune=False,
               output_dir='./model/bert_lense_new'):
    if fine_tune and os.path.exists(output_dir):
        model, tokenizer = load_model_and_tokenizer(output_dir, AutoModelForSequenceClassification)
        print("BERTLense is fine-tuned on BERTLense again")
    else:
        if llm_name is None:
            llm_name = './model/roberta-base'
        print("BERTLense is fine-tuned on", llm_name)
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModelForSequenceClassification.from_pretrained(llm_name, num_labels=2).to(device)

    train_encodings = tokenize_data(train_texts, tokenizer)
    test_encodings = tokenize_data(test_texts, tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': torch.tensor(train_labels)
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': torch.tensor(test_labels)
    })

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        logging_dir='../bert_logs',
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        report_to="tensorboard"  # Add tensorboard support
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # 设置早停
    )

    # Record loss and accuracy per epoch
    train_loss = []
    eval_accuracy = []
    eval_loss = []

    for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train model for one epoch
        train_results = trainer.train()
        train_loss.append(train_results.training_loss)

        # Evaluate model on validation set
        eval_results = trainer.evaluate()
        eval_accuracy.append(eval_results['eval_accuracy'])
        eval_loss.append(eval_results['eval_loss'])

        print(f"Training Loss: {train_results.training_loss:.4f}")
        print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"Validation Loss: {eval_results['eval_loss']:.4f}")

        # Early stopping check will halt training early if no improvement is seen
        if trainer.state.best_model_checkpoint is not None:
            break


    # Adjust epochs range according to actual training epochs
    epochs_range = range(1, len(train_loss) + 1)
    # Plot the training progress
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, eval_loss, label='Eval Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, eval_accuracy, label='Eval Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()

    # Save the plot to a file (e.g., as 'training_progress.png')
    plot_path = './output_new/training_progress.png'  # You can customize the path
    plt.savefig(plot_path)

    # Show the plot
    plt.show()

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer, model, tokenizer


# 6. Fake News Detection Model (BERT only)
def FakeLense(text, bert_model, bert_tokenizer):
    # Text preprocessing
    text = text_preprocessing(text)

    # BERT prediction
    bert_inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    bert_outputs = bert_model(input_ids=bert_inputs['input_ids'], attention_mask=bert_inputs['attention_mask'])
    bert_prediction = torch.argmax(bert_outputs.logits, dim=1).item()

    if bert_prediction == 1:
        return "Fake News Detected."
    else:
        return "Real News Detected."
