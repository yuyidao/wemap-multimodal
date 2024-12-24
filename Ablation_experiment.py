import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import class_definition as cla
import pandas as pd
import time

def main():
    # 配置文件路径和参数
    csv_file = 'train_data.csv'  # CSV文件路径
    img_dir = '../data_stroge/row_data/'  # 图像文件夹路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(os.listdir(img_dir))  # 类别数量等于子文件夹的数量

    tokenizer = BertTokenizer.from_pretrained('model_data/local_bert_base_chinese')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = cla.MultiModalDataset(csv_file, img_dir, tokenizer, transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    train_all(num_classes, device, data_loader)
    eval_all(num_classes, device, data_loader)


def train_all(num_classes, device, data_loader):

    model_names = [
        'MultiModalModel',
        'TextOnlyModel',
        'ImageOnlyModel',
        'NoFusionLayerModel',
        'EarlyFusionModel',
        'LateFusionModel']

    training_times = {}

    for model_name in model_names:
        # 初始化对应模型
        if model_name == 'MultiModalModel':
            model = cla.MultiModalModel(num_classes=num_classes).to(device)
        elif model_name == 'TextOnlyModel':
            model = cla.TextOnlyModel(num_classes=num_classes).to(device)
        elif model_name == 'ImageOnlyModel':
            model = cla.ImageOnlyModel(num_classes=num_classes).to(device)
        elif model_name == 'NoFusionLayerModel':
            model = cla.NoFusionLayerModel(num_classes=num_classes).to(device)
        elif model_name == 'EarlyFusionModel':
            model = cla.EarlyFusionModel(num_classes=num_classes).to(device)
        elif model_name == 'LateFusionModel':
            model = cla.LateFusionModel(num_classes=num_classes).to(device)

        # 损失函数
        criterion = nn.CrossEntropyLoss()
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        start_time = time.time()

        # model train
        cla.train_model(model_name, device, model, data_loader, criterion, optimizer)

        end_time = time.time()

        # 计算训练时长
        training_duration = end_time - start_time
        training_times[model_name] = training_duration
        print(f"Training time for {model_name}: {training_duration:.2f} seconds")

    # 保存训练时间数据到CSV文件
    training_times_df = pd.DataFrame(list(training_times.items()), columns=['Model', 'Training Time (Seconds)'])
    training_times_df.to_csv('model_training_times.csv', index=False)
    print("Training times for all models have been saved to 'model_training_times.csv'.")


def eval_all(num_classes, device, data_loader):
    # 模型路径
    model_paths = {
        'MultiModalModel': 'MultiModalModel.pth',
        'TextOnlyModel': 'TextOnlyModel.pth',
        'ImageOnlyModel': 'ImageOnlyModel.pth',
        'NoFusionLayerModel': 'NoFusionLayerModel.pth',
        'EarlyFusionModel': 'EarlyFusionModel.pth',
        'LateFusionModel': 'LateFusionModel.pth',
    }

    results = []

    for model_name, model_path in model_paths.items():
        print(f"Evaluating {model_name}...")

        # 初始化对应模型
        if model_name == 'MultiModalModel':
            model = cla.MultiModalModel(num_classes=num_classes).to(device)
        elif model_name == 'TextOnlyModel':
            model = cla.TextOnlyModel(num_classes=num_classes).to(device)
        elif model_name == 'ImageOnlyModel':
            model = cla.ImageOnlyModel(num_classes=num_classes).to(device)
        elif model_name == 'NoFusionLayerModel':
            model = cla.NoFusionLayerModel(num_classes=num_classes).to(device)
        elif model_name == 'EarlyFusionModel':
            model = cla.EarlyFusionModel(num_classes=num_classes).to(device)
        elif model_name == 'LateFusionModel':
            model = cla.LateFusionModel(num_classes=num_classes).to(device)

        # 加载模型权重
        model.load_state_dict(torch.load(model_path))

        # 评估模型
        accuracy, precision, recall, f1 = cla.evaluate_model(model, data_loader, device)

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    # 转换结果为DataFrame
    results_df = pd.DataFrame(results)

    # 保存为CSV文件
    results_df.to_csv('Ablation_evaluation_results.csv', index=False)


if __name__=="__main__":
    main()