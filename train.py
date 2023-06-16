import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from ECA_mobileNetV2 import MobileNetV2
from dataset import RafDataset

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 16
    epochs = 100

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ''))  # get data root path
    print(data_root)
    image_path = os.path.join('../','original')  
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = RafDataset(image_path, phase='train',label_path='label.txt', transform=data_transform['train'])
    test_dataset = RafDataset(image_path, phase='test',label_path='label.txt', transform=data_transform['val'])
    train_num = len(train_dataset)

    #{'Anger':0,'Disgust':1, 'Fear': 2 , 'Happiness': 3, 'Neutral' : 4 , 'Sadness': 5, 'Surprise': 6}

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    val_num = len(test_dataset)
    validate_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = MobileNetV2(num_classes=7)

    model_weight_path = os.path.join('./', "mobileNet-ECA-RAFDB-0.91199.pth")
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    # for param in net.features.parameters():
    #     param.requires_grad = False

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.0001)
    # optimizer = optim.SGD(params, lr=0.045, momentum=0.9, weight_decay=0.00001)
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.00001) 
    # optimizer = optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.00001)
    # optimizer = optim.SGD(params, lr=0.00001, momentum=0.9, weight_decay=0.00001)
    # optimizer = optim.SGD(params, lr=0.000001, momentum=0.9, weight_decay=0.00001)
    # optimizer = optim.SGD(params, lr=0.00000001, momentum=0.9, weight_decay=0.00001)
    best_acc = 0.0 
    # Maximum accuracy from the previous training session
    save_path = './mobileNet-ECA-RAFDB.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        # mark the train stage, net.train() enables Dropout and BN
        train_loss = 0.0
        train_acc  = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels,_ = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            # print(labels)
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            train_loss += loss.item()
            preds = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(preds, labels.to(device)).sum().item()
        
        train_loss /= train_num
        train_acc = train_acc/ train_num

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels, img_path= val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f val_accuracy: %.5f' %
              (epoch + 1, train_loss,train_acc, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('accuracy: %.5f' % best_acc)

    print('Finished Training')

if __name__ == '__main__':
    main()
