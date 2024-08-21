import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import tqdm
from mlps import MLP
    
# number of classes
num_classes = 3

# Daftar nilai untuk setiap hyperparameter
lr_values = [0.001]
num_epochs_values = [20]
batch_sizes = [8, 16]
hidden_layer_sizes = [[64, 32],[128, 64]]
activation_functions = [torch.relu, torch.sigmoid]
optimizers = [optim.Adam, optim.RMSprop, optim.SGD]
input_size = 320*320*3

# Kombinasi hyperparameter
hyperparameter_combinations = itertools.product(lr_values, num_epochs_values, batch_sizes, hidden_layer_sizes, activation_functions, optimizers)


results = []
i = 0

# Training model dengan setiap kombinasi hyperparameter
for lr, num_epochs, batch_size, hidden_size, activation_func, optimizer_type in hyperparameter_combinations:

    # Split dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    path = "results"
    folder_name = f"model_{i}"
    # create folder for each model
    if not os.path.exists(f"{path}/{folder_name}"):
        os.makedirs(f"{path}/{folder_name}")
    i += 1

    if activation_func == torch.relu:
        activation_func_str = "relu"
    elif activation_func == torch.sigmoid:
        activation_func_str = "sigmoid"
    elif activation_func == torch.tanh:
        activation_func_str = "tanh"

    if optimizer_type == optim.SGD:
        optimizer_type_str = "SGD"
    elif optimizer_type == optim.Adam:
        optimizer_type_str = "Adam"
    elif optimizer_type == optim.RMSprop:
        optimizer_type_str = "RMSprop"


    # create hyperparameter file
    path_hyperparameter = f"{path}/{folder_name}/hyperparameter.txt"
    with open(f"{path_hyperparameter}", "w") as f:
        f.write(f"lr={lr}\n")
        f.write(f"num_epochs={num_epochs}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"hidden_size={hidden_size}\n")
        f.write(f"activation_func={activation_func_str}\n")
        f.write(f"optimizer_type={optimizer_type_str}\n")


    # create csv file
    path_csv = f"{path}/{folder_name}/results.csv"
    with open(f"{path_csv}", "w") as f:
        f.write("epoch,loss,precision,recall,f1_score,accuracy\n")


    # Inisialisasi model, loss function, dan optimizer
    print(f'Hyperparameter: lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}, hidden_size={hidden_size}, activation_func={activation_func_str}, optimizer_type={optimizer_type_str}')
    print("=====================================")
    model = MLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_type(model.parameters(), lr=lr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_epochs = num_epochs
    best_acc = 0

    # Training model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, targets in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # Evaluasi model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                incorrect = (predicted != labels).sum().item()


        precision = correct / total
        recall = correct / (correct + incorrect)
        f1 = 2 * (precision * recall) / (precision + recall)

        accuracy = correct/total

        print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

        if best_acc < accuracy :
            torch.save(model.state_dict(), f"{path}/{folder_name}/best.pth")

         # save results
        with open(f"{path_csv}", "a") as f:
            f.write(f"{epoch+1},{running_loss/len(train_loader)},{precision},{recall},{f1},{accuracy}\n")

    torch.save(model.state_dict(), f"{path}/{folder_name}/last.pth")

    results.append({
        "model": "model_" + str(i),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "best_accuracy": best_acc,
        "last_accuracy": accuracy,
    })

# save results
with open(f"results.csv", "w") as f:
    f.write("model,precision,recall,f1_score,best_accuracy,last_accuracy\n")
    for result in results:
        f.write(f"{result['model']},{result['precision']},{result['recall']},{result['f1_score']},{result['best_accuracy'],result['last_accuracy']}\n")


print("Training selesai")
print("=====================================")


# Load image
from PIL import Image

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        incorrect = (predicted != labels).sum().item()

precision = correct / total
recall = correct / (correct + incorrect)
f1 = 2 * (precision * recall) / (precision + recall)

accuracy = correct/total

print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
print(f'Test Accuracy: {100 * correct / total:.2f}%')
