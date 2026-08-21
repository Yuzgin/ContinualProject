# Transformer - Different weight decays.

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.models.vision_transformer import VisionTransformer

wds = [0.0005, 0.005, 0.05]
batch_size = 128
epochs = 30
seed = 23
lr = 0.001
selection = "random"
data_folder = "../data"

dataset_path = "../data/cifar100"
buffer_size = 10000


# dataset_path = "../data/cifar10"
# buffer_size = 2500

# 20 tasks (5 classes per task)
# tasks = []
# for i in range(0, 100, 5):
#     task = list(range(i, i + 5))
#     tasks.append(task)

# 10 tasks (10 classes per task)
tasks = []
for i in range(0, 100, 10):
    task = list(range(i, i + 10))
    tasks.append(task)

# 5 tasks (20 classes per task)
# tasks = []
# for i in range(0, 100, 20):
#     task = list(range(i, i + 20))
#     tasks.append(task)

norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))



# GPU selection for parralelisation]
device_ids = [0, 1, 2, 3]
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm,
])
test_transform = transforms.Compose([transforms.ToTensor(), norm,
])

class CifarDataset(Dataset):
    def __init__(self, images_folder, csv_file, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.targets = list(self.data["label"])

        self.images = []
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            img_path = os.path.join(images_folder, row["image"])
            image = Image.open(img_path).convert('RGB')
            self.images.append(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    if not os.path.exists(dataset_path):
        print("CIFAR-10 not found")
        return

    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(dataset_path, "train_labels.csv")
    test_csv = os.path.join(dataset_path, "test_labels.csv")
    train_folder = os.path.join(dataset_path, "train")
    test_folder = os.path.join(dataset_path, "test")

    train_dataset = CifarDataset(train_folder, train_csv, train_transform)
    test_dataset = CifarDataset(test_folder, test_csv, test_transform)
    store_dataset = CifarDataset(train_folder, train_csv, test_transform)

    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")


    for wd in wds:
        print(f"Running weight decay {wd}")


        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Transformer model
        model = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=12,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,

            # num_classes=10,
            num_classes=100,
        )
        model = nn.DataParallel(model, device_ids=device_ids)
        print("Using GPUs " + str(device_ids))
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {n_params}")

        criterion = nn.CrossEntropyLoss()

        replay_images = []
        replay_labels = []
        samples_per_task = buffer_size // len(tasks)

        # Continual learning results
        accuracy_matrix = []
        for i in range(len(tasks)):
            accuracy_matrix.append([0.0] * len(tasks))

        for task_id in range(len(tasks)):
            classes = tasks[task_id]

            indices = []
            for i in range(len(train_dataset.targets)):
                if train_dataset.targets[i] in classes:
                    indices.append(i)

            train_loader = DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True, num_workers=16)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            # Stack replay
            if len(replay_images) > 0:
                replay_x = torch.stack(replay_images)
                replay_y = torch.tensor(replay_labels)
            else:
                replay_x = None
                replay_y = None

            epoch_bar = tqdm(total=epochs, desc=f"Task {task_id + 1}")
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # Add replay examples to batch
                    if replay_x is not None:
                        n = min(images.size(0), len(replay_images))
                        idx = torch.randint(0, len(replay_images), (n,))
                        images = torch.cat([images, replay_x[idx].to(device)])
                        labels = torch.cat([labels, replay_y[idx].to(device)])

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                scheduler.step()
                epoch_bar.set_postfix(epoch=epoch + 1, loss=round(running_loss / len(train_loader), 4))
                epoch_bar.update(1)
            epoch_bar.close()

            # Add prev task examples to buffer
            if samples_per_task > 0:
                store_indices = []
                for i in range(len(store_dataset.targets)):
                    if store_dataset.targets[i] in classes:
                        store_indices.append(i)

                n_take = min(samples_per_task, len(store_indices))
                chosen = random.sample(store_indices, n_take)

                for i in chosen:
                    image, label = store_dataset[i]
                    replay_images.append(image)
                    replay_labels.append(label)

            # Evaluation
            model.eval()
            for eval_id in range(task_id + 1):
                eval_classes = tasks[eval_id]
                eval_indices = []
                for i in range(len(test_dataset.targets)):
                    if test_dataset.targets[i] in eval_classes:
                        eval_indices.append(i)

                test_loader = DataLoader(Subset(test_dataset, eval_indices), batch_size=batch_size, shuffle=False, num_workers=16)

                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        preds = model(images).argmax(dim=1)
                        correct = correct + (preds == labels).sum().item()
                        total = total + labels.size(0)

                acc = 100 * correct / total
                accuracy_matrix[task_id][eval_id] = acc
                print(f"After task {task_id+1} on task {eval_id+1}: {acc:.2f}%")


        # Print results
        print("buffer =", buffer_size, "selection =", selection, "lr =", lr, "wd =", wd, "seed =", seed)
        print("parameters =", n_params)
        print("accuracy matrix:")
        for i in range(len(tasks)):
            print("after task", i + 1, ":", accuracy_matrix[i][:i + 1])

        forget_vals = []
        print("forgetting:")
        for task_id in range(len(tasks) - 1):
            peak = accuracy_matrix[task_id][task_id]
            final = accuracy_matrix[len(tasks) - 1][task_id]
            forget = peak - final
            forget_vals.append(forget)
            print("task", task_id + 1, "=", forget)

        final_row = accuracy_matrix[len(tasks) - 1]
        avg_acc = sum(final_row) / len(tasks)
        avg_forget = sum(forget_vals) / len(forget_vals) if forget_vals else 0.0
        print("average accuracy =", avg_acc)
        print("average forgetting =", avg_forget)

        family = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        kind = os.path.splitext(os.path.basename(__file__))[0].split("_")[-1]
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{family}_{kind}_{len(tasks)}task.txt")

        with open(results_path, "a") as f:
            f.write("=" * 60 + "\n")
            f.write(f"buffer = {buffer_size}\n")
            f.write(f"selection = {selection}\n")
            f.write(f"lr = {lr}\n")
            f.write(f"wd = {wd}\n")
            f.write(f"seed = {seed}\n")
            f.write(f"parameters = {n_params}\n")
            f.write(f"num_tasks = {len(tasks)}\n")
            f.write("accuracy matrix:\n")
            for i in range(len(tasks)):
                f.write(f"after task {i + 1} : {accuracy_matrix[i][:i + 1]}\n")
            f.write("forgetting:\n")
            for task_id, forget in enumerate(forget_vals):
                f.write(f"task {task_id + 1} = {forget}\n")
            f.write(f"average accuracy = {avg_acc}\n")
            f.write(f"average forgetting = {avg_forget}\n")
            f.write("\n")
        print("saved results to", results_path)
        print()


if __name__ == "__main__":
    main()
