"""
author:Shuaifeng
data:10/8/2022
"""
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from tqdm import tqdm
import sys

from model import FullyConnected

def train_model(
    train_loader,
    val_loader,
    test_loader,
    comment,
    num_classes=16,
    num_epoch=200,
    if_writer=False,
    model_path = None,
    lr=1e-2
):
    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    # Instantiate the model
    net = FullyConnected(num_classes)
    if model_path:
        net.load_state_dict(torch.load(model_path))

    # path to save the model
    comment = comment + "_" + net.name
    PATH =  comment + ".pth"
    # print model summary
    if if_writer:
        print(summary(net, torch.zeros((32, 4))))
    # send model to GPU
    net.to(device)

    # set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40, 60], gamma=0.2
    )  # 10, 15

    if if_writer:
        writer = SummaryWriter(comment=comment)

    # train model
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        running_acc = 1.0
        with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
            for i, (pos, label, pwr) in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                pos = pos.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(pos)
                loss = criterion(outputs, label)
                prediction = torch.argmax(outputs, dim=-1)
                acc = (prediction == label).float().mean().item()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_acc = (acc + i * running_acc) / (i + 1)
                log = OrderedDict()
                log["loss"] = running_loss
                log["acc"] = running_acc
                tepoch.set_postfix(log)
            scheduler.step()

            if if_writer:
                # validation
                predictions = []
                net.eval()
                with torch.no_grad():
                    total = 0
                    top1_correct = 0
                    top2_correct = 0
                    top3_correct = 0
                    top5_correct = 0
                    val_loss = 0
                    for (pos, label, pwr) in val_loader:
                        pos = pos.to(device)
                        label = label.to(device)
                        optimizer.zero_grad()
                        
                        outputs = net(pos)
                        
                        val_loss += nn.CrossEntropyLoss(reduction="sum")(
                            outputs.view(-1, num_classes), label.flatten()
                        ).item()

                        total += label.cpu().numpy().size
                        prediction = torch.argmax(outputs, dim=-1)
                        top1_correct += torch.sum(prediction == label, dim=-1).cpu().numpy()
                        _, idx = torch.topk(outputs, 5, dim=-1)
                        idx = idx.cpu().numpy()
                        label = label.cpu().numpy()
                        for j in range(label.shape[0]):
                            top2_correct += np.isin(label[j], idx[j, :2]).sum()
                            top3_correct += np.isin(label[j], idx[j, :3]).sum()
                            top5_correct += np.isin(label[j], idx[j, :5]).sum()
                        predictions.append(prediction.cpu().numpy())

                    val_loss /= float(total)
                    val_top1_acc = top1_correct / float(total)
                    val_top2_acc = top2_correct / float(total)
                    val_top3_acc = top3_correct / float(total)
                    val_top5_acc = top5_correct / float(total)
                    print("val_loss={:.4f}".format(val_loss), flush=True)
                    print("accuracy", flush=True)
                    print(
                        np.stack(
                            [val_top1_acc, val_top2_acc, val_top3_acc, val_top5_acc], 0
                        ),
                        flush=True,
                    )
        if if_writer:
            writer.add_scalar("Loss/train", running_loss, epoch)
            writer.add_scalar("Loss/test", val_loss, epoch)
            writer.add_scalar("acc/train", running_acc, epoch)
            writer.add_scalar("acc/test", val_top1_acc, epoch)
    if if_writer:
        writer.close()
        torch.save(net.state_dict(), PATH)
    print("Finished Training")
    
    net.to(device)
    # net.load_state_dict(torch.load(PATH))

    # test
    # validation
    predictions = []
    raw_predictions = []
    true_label = []
    net.eval()
    with torch.no_grad():
        total = 0
        top1_correct = 0
        top2_correct = 0
        top3_correct = 0
        top5_correct = 0
        top1_pwr = 0
        top2_pwr = 0
        top3_pwr = 0
        top5_pwr = 0
        test_loss = 0
        for (pos, label, pwr) in test_loader:
            pos = pos.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            
            outputs = net(pos)

            test_loss += nn.CrossEntropyLoss(reduction="sum")(
                outputs.view(-1, num_classes), label.flatten()
            ).item()
            total += label.cpu().numpy().size
            prediction = torch.argmax(outputs, dim=-1)
            top1_correct += torch.sum(prediction == label, dim=-1).cpu().numpy()

            _, idx = torch.topk(outputs, 5, dim=-1)
            idx = idx.cpu().numpy()
            label = label.cpu().numpy()
            for j in range(label.shape[0]):
                top2_correct += np.isin(label[j], idx[j, :2]).sum()
                top3_correct += np.isin(label[j], idx[j, :3]).sum()
                top5_correct += np.isin(label[j], idx[j, :5]).sum()

            for j in range(label.shape[0]):
                max_pwr = pwr[j, label[j]]
                top1_pwr += pwr[j, idx[j, :1]].max() / max_pwr
                top2_pwr += pwr[j, idx[j, :2]].max() / max_pwr
                top3_pwr += pwr[j, idx[j, :3]].max() / max_pwr
                top5_pwr += pwr[j, idx[j, :5]].max() / max_pwr


            predictions.append(prediction.cpu().numpy())
            raw_predictions.append(outputs.cpu().numpy())
            true_label.append(label)

        test_loss /= float(total)
        test_top1_acc = top1_correct / float(total)
        test_top2_acc = top2_correct / float(total)
        test_top3_acc = top3_correct / float(total)
        test_top5_acc = top5_correct / float(total)

        test_top1_pwr = top1_pwr.cpu().numpy() / float(total)
        test_top2_pwr = top2_pwr.cpu().numpy() / float(total)
        test_top3_pwr = top3_pwr.cpu().numpy() / float(total)
        test_top5_pwr = top5_pwr.cpu().numpy() / float(total)

        predictions = np.concatenate(predictions, 0)
        raw_predictions = np.concatenate(raw_predictions, 0)
        true_label = np.concatenate(true_label, 0)

        test_acc = np.asarray([
            test_top1_acc,
            test_top2_acc,
            test_top3_acc,
            test_top5_acc
        ])

        test_pwr = np.asarray([
            test_top1_pwr,
            test_top2_pwr,
            test_top3_pwr,
            test_top5_pwr
        ])
        return test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, PATH


if __name__ == "__main__":
    torch.manual_seed(2022)
    num_epoch = 80 # 80
    test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, PATH = train_model(num_epoch, if_writer=False)
    print(test_loss)
    print(test_acc)
    print(test_pwr)
    print('done')
