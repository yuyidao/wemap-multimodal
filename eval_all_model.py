import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
from transformers import BertTokenizer, BertModel, ViltProcessor, ViltModel, CLIPProcessor, CLIPModel
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from model_data import clip


# BERT-ResNet Model Definition
class MultiModalModel(nn.Module):
    def __init__(self, num_classes, bert_model_name='model_data/local_bert_base_chinese'):
        super(MultiModalModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_fc = nn.Linear(self.bert.config.hidden_size, 256)

        self.resnet = models.resnet50(pretrained=True)
        self.resnet_fc = nn.Linear(self.resnet.fc.in_features, 256)
        self.resnet.fc = nn.Identity()

        self.fc = nn.Linear(256 + 256, num_classes)

    def forward(self, input_ids, attention_mask, images):
        text_features = self.bert_fc(self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
        image_features = self.resnet_fc(self.resnet(images))
        combined_features = torch.cat((text_features, image_features), dim=1)
        return self.fc(combined_features)


# Dataset for MultiModalModel (BERT-ResNet)
class MultiModalDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None, max_len=32):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(self.data['label'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, img_name = (self.data.iloc[idx, 0],
                                 self.class_to_idx[self.data.iloc[idx, 1]],
                                 self.data.iloc[idx, 2])
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=self.max_len)

        image_path = self.get_image_path(img_name, idx)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), image, torch.tensor(label,
                                                                                                      dtype=torch.long)

    def get_image_path(self, img_name, idx):
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        for ext in possible_extensions:
            potential_path = os.path.join(self.img_dir, self.data.iloc[idx, 1], img_name + ext)
            if os.path.exists(potential_path):
                return potential_path
        raise FileNotFoundError(f"Image file not found: {img_name}")


# CLIP and ViLT Dataset definitions
class CLIPDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(self.data['label'].unique()))}
        self.invalid_indices = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, img_name = (self.data.iloc[idx, 0],
                                 torch.tensor(self.class_to_idx[self.data.iloc[idx, 1]], dtype=torch.long),
                                 self.data.iloc[idx, 2])
        image_path = self.get_image_path(img_name, idx)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return text, image, int(label)

    def get_image_path(self, img_name, idx):
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        for ext in possible_extensions:
            potential_path = os.path.join(self.img_dir, self.data.iloc[idx, 1], img_name + ext)
            if os.path.exists(potential_path):
                return potential_path
        raise FileNotFoundError(f"Image file not found: {img_name}")


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(self.data['label'].unique()))}

        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = torch.tensor(self.class_to_idx[self.data.iloc[idx, 1]], dtype=torch.long)
        img_name = self.data.iloc[idx, 2]

        # 图像处理
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_path = None
        for ext in possible_extensions:
            potential_path = os.path.join(self.img_dir, self.data.iloc[idx, 1], img_name + ext)
            abs_path = os.path.abspath(potential_path)
            if os.path.exists(abs_path):
                image_path = abs_path
                break

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = (image - image.min()) / (image.max() - image.min())

        # 编码图像和文本
        encoding = self.processor(images=image, text=text, return_tensors="pt", padding="max_length", max_length=40, truncation=True)

        pixel_values = encoding["pixel_values"].squeeze(0)
        input_ids = encoding["input_ids"].squeeze(0)

        # print(pixel_values.shape, input_ids.shape, label.shape)
        # exit()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": label,
        }


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.vilt_model = ViltModel.from_pretrained("model_data/vilt-b32-mlm")
        self.classifier = nn.Linear(self.vilt_model.config.hidden_size, num_classes)

    def forward(self, input_ids, pixel_values):
        outputs = self.vilt_model(input_ids=input_ids, pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # 获取[CLS]标记的输出
        logits = self.classifier(pooled_output)
        return logits


# Evaluation Function
def evaluate_clip_model(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: batch[key].to(device) for key in batch if key != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, average='weighted'),
        "Recall": recall_score(all_labels, all_preds, average='weighted'),
        "F1-score": f1_score(all_labels, all_preds, average='weighted')
    }
    return metrics

def evaluate_vilt_model(model, dataloader, device):
    model.eval()  # 切换到评估模式
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # 禁止梯度计算
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, pixel_values=pixel_values)
            preds = torch.argmax(logits, dim=1)

            # 收集标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    metrics = {
        "Accuracy": accuracy_score(all_labels, all_predictions),
        "Precision": precision_score(all_labels, all_predictions, average='weighted'),
        "Recall": recall_score(all_labels, all_predictions, average='weighted'),
        "F1-score": f1_score(all_labels, all_predictions, average='weighted')
    }
    return metrics

def evaluate_bert_resnet_model(model, data_loader, device):
    model.eval()  # 切换到评估模式
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, images, labels = batch
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask, images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

# Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file = 'data.csv'
    img_dir = '../data_stroge/row_data/'

    # CLIP模型
    clip_model, clip_processor = clip.load("ViT-B/32", device=device)
    clip_dataset = CLIPDataset(data_file, img_dir, tokenizer=clip.tokenize, transform=clip_processor)
    clip_dataloader = torch.utils.data.DataLoader(clip_dataset, batch_size=4, shuffle=True)
    # 加载训练好的CLIP模型权重
    clip_model.load_state_dict(torch.load('model_data/clip_model.pth')['model_state_dict'])
    clip_model = clip_model.to(device).eval()

    # ViLT模型
    vilt_processor = ViltProcessor.from_pretrained("model_data/vilt-b32-mlm")
    vilt_dataset = CustomDataset(data_file, img_dir, vilt_processor)
    vilt_dataloader = DataLoader(vilt_dataset, batch_size=4, shuffle=True)
    vilt_model = CustomModel(num_classes=len(vilt_dataset.class_to_idx)).to(device)
    # 加载训练好的ViLT模型权重
    vilt_model.load_state_dict(torch.load("model_data/vilt_model.pth"))

    # BERT-ResNet模型
    bert_resnet_model = MultiModalModel(num_classes=len(os.listdir(img_dir))).to(device)
    tokenizer = BertTokenizer.from_pretrained('./model_data/local_bert_base_chinese')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    bert_resnet_dataset = MultiModalDataset(data_file, img_dir, tokenizer, transform=transform)
    bert_resnet_dataloader = DataLoader(bert_resnet_dataset, batch_size=4, shuffle=False, num_workers=4)
    # 加载训练好的BERT-ResNet模型权重
    bert_resnet_model.load_state_dict(torch.load("model_data/bert_resnet.pth"), strict=False)

    results = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-score": []}

    accuracy, precision, recall, f1 = evaluate_clip_model(clip_model, clip_dataloader, device)
    results["Model"].append("CLIP")
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1-score"].append(f1)
    accuracy, precision, recall, f1 = evaluate_vilt_model(vilt_model, vilt_dataloader, device)
    results["Model"].append("ViLT")
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1-score"].append(f1)
    accuracy, precision, recall, f1 = evaluate_bert_resnet_model(bert_resnet_model, bert_resnet_dataloader, device)
    results["Model"].append("BERT-ResNet")
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1-score"].append(f1)

    # 保存评估结果
    results_df = pd.DataFrame(results)
    results_df.to_csv("result/model_evaluation_results.csv", index=False)
    print("评估结果已保存至 result/model_evaluation_results.csv")


if __name__ == "__main__":
    main()

