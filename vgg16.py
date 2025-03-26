import os
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
import time



# 类别名称
def load_class_names(root):
    meta_path = os.path.join(root, 'devkit', 'cars_annos.mat')
    meta = scipy.io.loadmat(meta_path)
    class_names = [name[0] for name in meta['class_names'][0]]
    return class_names

# 文件读取
# putout:一个列表，每个元素是 (image_path, class_idx, bbox) 的元组。
def load_annotations(root, annos_file):
    annos_path = os.path.join(root, annos_file)
    annos = scipy.io.loadmat(annos_path)
    annotation = annos['annotations']
    data_list = []
    for i in range(len(annotation['fname'][0])):
        fname = annotation['fname'][0][i].tostring().decode('utf-8')
        class_idx = int(annotation['class'][0][i])-1   # 转换为 0 基索引
        bbox_x1 = int(annotation['bbox_x1'][0][i])
        bbox_y1 = int(annotation['bbox_y1'][0][i])
        bbox_x2 = int(annotation['bbox_x2'][0][i])
        bbox_y2 = int(annotation['bbox_y2'][0][i])
        if class_idx>2:
            continue
        image_path = os.path.join(root, 'cars_train' if 'train' in annos_file else 'cars_test', fname).replace('\x00','')
        data_list.append((image_path, class_idx, (bbox_x1, bbox_y1, bbox_x2, bbox_y2)))
    return data_list


#数据处理
class StanfordCarsDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    #确认数据量
    def __len__(self):
        return len(self.data_list)
    #数据增强处理
    def __getitem__(self, idx):
        image_path, label ,bbox= self.data_list[idx]
        image = Image.open(image_path).convert('RGB')
        # 根据边界框坐标裁剪
        image = image.crop(bbox)
        #根据transform进行数据处理
        if self.transform:
            image = self.transform(image)
        return image, label

# 图片数据处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),      #统一尺寸大小
    transforms.RandomHorizontalFlip(),  #随机水平翻转，防止过拟合
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色扰动
    transforms.ToTensor(),              #转变为torch.Tensor，像素归一，CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#标准化图像数据
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  #测试集不用翻转
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

'''
# 数据平衡处理(由于统计了各类别的频数，可省略该步骤)
def get_balanced_sampler(dataset):
    labels = [label for _, label, _ in dataset.data_list]
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}#类别权重
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))
'''

# 主函数
def main(root):
    # 加载类别名称
    class_names = load_class_names(root)[0:3]
    # 加载训练和测试数据
    train_data = load_annotations(root, 'devkit/cars_train_annos.mat')
    test_data = load_annotations(root, 'devkit/cars_test_annos_withlabels.mat')

    # 创建数据集和数据加载器
    train_dataset = StanfordCarsDataset(train_data, transform=train_transform)
    test_dataset = StanfordCarsDataset(test_data, transform=test_transform)

    batch_size = 32
    #train_sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载预训练模型并修改
    model = models.vgg16(pretrained=True)
    num_classes = 3
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练和评估
    num_epochs = 4
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'\r {running_loss}',end='')#实时显示累计损失
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')#当前epoch的平均损失（总损失除以批次数）
        torch.save(model.state_dict(), 'model.pth')
        # 评估
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        start_time = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        end_time = time.time()

        accuracy = correct / total * 100
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        params = sum(p.numel() for p in model.parameters())
        fps = total / (end_time - start_time)

        print(f'Accuracy: {accuracy:.2f} %')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'Params: {params}')
        print(f'FPS: {fps:.2f}')

    # 可视化部分训练图像
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.ravel()):
        img_path, label,_ = train_data[i]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(class_names[label])
        ax.axis('off')
    plt.show()

if __name__ == "__main__":
    root = '.\\stanford_cars'
    main(root)