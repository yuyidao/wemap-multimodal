import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# 定义MultiModalModel及消融实验模型
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
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.bert_fc(bert_output.pooler_output)

        image_features = self.resnet(images)
        image_features = self.resnet_fc(image_features)

        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined_features)
        return output


class TextOnlyModel(nn.Module):
    def __init__(self, num_classes, bert_model_name='model_data/local_bert_base_chinese'):
        super(TextOnlyModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, _):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.bert_fc(bert_output.pooler_output)
        return text_features


class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageOnlyModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet_fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.fc = nn.Identity()

    def forward(self, _, __, images):
        image_features = self.resnet(images)
        output = self.resnet_fc(image_features)
        return output


class NoFusionLayerModel(nn.Module):
    def __init__(self, num_classes, bert_model_name='model_data/local_bert_base_chinese'):
        super(NoFusionLayerModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.resnet = models.resnet50(pretrained=True)

        # 保存 ResNet 最后一层的输入特征维度
        resnet_features = self.resnet.fc.in_features

        # 替换 ResNet 的最后一层为 nn.Identity
        self.resnet.fc = nn.Identity()

        # 定义最终的全连接层
        self.fc = nn.Linear(self.bert.config.hidden_size + resnet_features, num_classes)

    def forward(self, input_ids, attention_mask, images):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output

        image_features = self.resnet(images)
        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined_features)
        return output


# 定义数据集类
class MultiModalDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None, max_len=32):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(self.data['label'].unique()))}
        print(f"Class to Index Mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        img_name = str(self.data.iloc[idx, 2])  # 强制转换为字符串
        label = torch.tensor(self.class_to_idx[self.data.iloc[idx, 1]], dtype=torch.long)

        # Tokenizer编码
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=self.max_len)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # 检查img_name是否为空
        if not img_name or img_name.lower() == 'nan':
            raise ValueError(f"Image name is invalid for index {idx}: {img_name}")

        img_name = img_name.zfill(4)

        # 处理图片路径
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        img_path = None

        for ext in possible_extensions:
            potential_path = os.path.join(self.img_dir, self.data.iloc[idx, 1], img_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None:
            raise FileNotFoundError(f"Image file not found for index {idx}: {img_name}")

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return input_ids, attention_mask, image, label


class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, bert_model_name='model_data/local_bert_base_chinese'):
        super(EarlyFusionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 去掉 ResNet 的最后一层

        # 融合后特征尺寸
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output

        image_features = self.resnet(images)
        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined_features)
        return output


class LateFusionModel(nn.Module):
    def __init__(self, num_classes, bert_model_name='model_data/local_bert_base_chinese'):
        super(LateFusionModel, self).__init__()
        self.text_model = BertModel.from_pretrained(bert_model_name)
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, num_classes)

        self.image_model = models.resnet50(pretrained=True)
        self.image_fc = nn.Linear(self.image_model.fc.in_features, num_classes)
        self.image_model.fc = nn.Identity()

    def forward(self, input_ids, attention_mask, images):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_logits = self.text_fc(text_output.pooler_output)

        image_features = self.image_model(images)
        image_logits = self.image_fc(image_features)

        # 融合阶段
        combined_logits = (text_logits + image_logits) / 2
        return combined_logits


def train_model(model_name, device, model, data_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, images, labels = batch
            input_ids, attention_mask, images, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                images.to(device),
                labels.to(device),
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}")
    print(f"Finished training {model_name}.")


def evaluate_model(device, model, data_loader, model_name):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, images, labels = batch
            input_ids, attention_mask, images, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                images.to(device),
                labels.to(device),
            )

            logits = model(input_ids=input_ids, attention_mask=attention_mask, images=images)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    metrics = {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, average="weighted"),
        "Recall": recall_score(all_labels, all_preds, average="weighted"),
        "F1-score": f1_score(all_labels, all_preds, average="weighted"),
    }
    return metrics