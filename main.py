# encoding:utf-8

# 用表情数据集训练神经网络

import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data import Dataset

from my_image_folder import MyImageFolder


def train(model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        count_batch = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每次训练含有train和validate两部分
        for phase in ['train', 'val']:
            if phase == 'train':
                # 调整
                scheduler.step()
                # 训练模式
                model.train(True)
            else:
                # 验证模式
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for data in data_loaders[phase]:
                count_batch += 1
                # 获得inputs和labels
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # 梯度归0
                optimizer.zero_grad()

                # 前向
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # 反向
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                # 每批10个输出
                if count_batch % 10 == 0:
                    batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc = running_corrects / (batch_size*count_batch)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'.format(
                        phase, epoch, count_batch, batch_loss, batch_acc, time.time()-begin_time))
                    begin_time = time.time()

            # 平均
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 保存模型
            if phase == 'train':
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                torch.save(model, 'checkpoint/densenet_checkpoint.pth')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    # 训练完成
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型的权重
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # 变换
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # 数据集
    image_datasets = {x: MyImageFolder(img_root_path='data/KDEF',
                                       label_path=('data/KDEF/' + x + '.txt'),
                                       data_transforms=data_transforms,
                                       dataset=x) for x in ['train', 'val']}
    batch_size = 8
    # 数据加载器
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 是否使用GPU
    use_gpu = torch.cuda.is_available()
    # 模型
    model = models.densenet121(pretrained=True)
    if use_gpu:
        model = model.cuda()
    # multi-GPU
    model = torch.nn.DataParallel(model, device_ids=[0])

    # 定义loss
    criterion = nn.CrossEntropyLoss()
    # 定义参数优化器,更新权重
    optimizer_ft = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)

    # 训练模型
    model = train(model=model,
                  criterion=criterion,
                  optimizer=optimizer_ft,
                  scheduler=exp_lr_scheduler,
                  num_epochs=25,
                  use_gpu=use_gpu)

    # save best model
    torch.save(model, "checkpoint/best_densenet.pth")


