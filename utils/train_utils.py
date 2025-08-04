import numpy as np
import torch
from torch import nn
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def print_time(t_i, t_f):
    elapsed_time_seconds = t_f - t_i
    hours = int(elapsed_time_seconds // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)
    print("Elapsed time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))


def cm_save(cm, cm_path, cm_title, clases, fontsiez=12, save=True):
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=fontsiez * 1.3333)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsiez)
    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, clases, rotation=0, fontsize=fontsiez)
    plt.yticks(tick_marks, clases, fontsize=fontsiez)
    plt.xlabel("VGG", fontsize=fontsiez * 1.166666)
    plt.ylabel("OVDAS", fontsize=fontsiez * 1.166666)
    if cm_title is not None:
        plt.title(cm_title, fontsize=fontsiez * 1.166666)
    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if cm_percent[i, j] > np.max(cm_percent) / 2 else "black"
            plt.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_percent[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=fontsiez,
            )  # , weight='bold')

    plt.tight_layout(pad=0)
    if save:
        plt.savefig(cm_path, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close()


def train(
    model_torch, train_loader, optimizer, criterion, device, epoch, batch_verbose
):
    running_loss = 0.0
    len_data = len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels, _ = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model_torch(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        if (i + 1) % batch_verbose == 0:
            print(
                f"running loss:{loss.item():.4f}, epoch {epoch}, batch {i+1}/{len_data} | lr: {current_lr}"
            )
    return running_loss


def validate(model, testloader, device, criterion=nn.CrossEntropyLoss()):
    # ---------------------
    # model = model_torch
    # testloader = val_loader
    # ---------------------
    eval_losses = []
    model.eval()
    total_validation_labels = []
    total_validation_predicted = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            # real = labels
            # total += labels.size(0)
            # correct += predicted.eq(real).sum().item()
            eval_losses.append(loss.item())
            labels = labels.cpu()
            predicted = predicted.cpu()
            total_validation_labels.extend(labels.tolist())
            total_validation_predicted.extend(predicted.tolist())
            if i % 10 == 0:
                print(f"batch {i} of {len(testloader)}")
        accuracy = accuracy_score(total_validation_labels, total_validation_predicted)
        recall = recall_score(
            total_validation_labels, total_validation_predicted, average="weighted"
        )
        f1 = f1_score(
            total_validation_labels, total_validation_predicted, average="weighted"
        )
        cm = confusion_matrix(total_validation_labels, total_validation_predicted)
        specificity = []
        for i in range(len(cm)):
            # Calculate specificity for the current class
            tn = sum(
                cm[j][j] for j in range(len(cm)) if j != i
            )  # true negatives excluding current class
            fp = sum(
                cm[j][i] for j in range(len(cm)) if j != i
            )  # false positives for current class
            specificity.append(tn / (tn + fp))
        mean_spec = np.mean(specificity)

    print(
        "Validation: Loss: %.3f | Accuracy: %.3f | Recall: %.3f | f1-score: %.3f | Specificity: %.3f"
        % (np.mean(eval_losses), accuracy, recall, f1, np.mean(specificity))
    )
    return (
        np.mean(eval_losses),
        accuracy,
        recall,
        f1,
        mean_spec,
        specificity,
        cm,
        # total_validation_labels,# estos hay que sacarlos
        # total_validation_predicted,
    )
