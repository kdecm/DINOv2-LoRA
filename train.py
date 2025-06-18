import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import torch.hub
import ssl
import argparse
import time
from thop import profile
from thop import clever_format
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  

ssl._create_default_https_context = ssl._create_unverified_context

# Configuration
CONFIG = {
    "model_path": "facebookresearch/dinov2",
    "model_name": "dinov2_vits14",
    "train_dir": "/home/UserData/dino/lora/apple_dataset/train/",
    "val_dir": "/home/UserData/dino/lora/apple_dataset/val/",
    "test_dir": "/home/UserData/dino/lora/apple_dataset/test/",
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "lora_rank": 32,
    "lora_alpha": 16,
    "seed": 42
}

os.environ['TORCH_HOME'] = '/path/to/weights'

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_output_dir(base_dir="output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class ParameterAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.param_log = os.path.join(output_dir, "parameter_analysis.txt")
        with open(self.param_log, 'w') as f:
            f.write("Parameter Analysis Log\n")
            f.write("="*40 + "\n\n")
    
    def log_parameters(self, model, stage_name):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
        classifier_params = sum(p.numel() for n, p in model.named_parameters() if 'classifier' in n)
        
        analysis = (
            f"\n{stage_name} Parameter Analysis:\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)\n"
            f"  - LoRA params: {lora_params:,} ({100*lora_params/total_params:.2f}%)\n"
            f"  - Classifier params: {classifier_params:,} ({100*classifier_params/total_params:.2f}%)\n"
            f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)\n\n"
            "Detailed Breakdown:\n"
        )
        
        param_details = {}
        for name, param in model.named_parameters():
            module = name.split('.')[0]
            if 'lora_' in name.lower():
                module = 'LoRA_params'
            param_details[module] = param_details.get(module, 0) + param.numel()
        
        for module, count in param_details.items():
            analysis += f"{module:20}: {count:12,} params ({100*count/total_params:.2f}%)\n"
        
        with open(self.param_log, 'a') as f:
            f.write(analysis + "\n")
        print(analysis)

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, few_shot_k=None):
        self.root_dir = root_dir
        self.transform = transform
        self.few_shot_k = few_shot_k
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()
    
    def _load_images(self):
        images = []
        rng = random.Random(CONFIG["seed"])
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            img_names = [img_name for img_name in os.listdir(class_dir) 
                         if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.few_shot_k is not None and self.few_shot_k > 0:
                rng.shuffle(img_names)
                img_names = img_names[:self.few_shot_k]
            for img_name in img_names:
                img_path = os.path.join(class_dir, img_name)
                images.append((img_path, self.class_to_idx[class_name]))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class LoRA_qkv(nn.Module):
    def __init__(self, qkv: nn.Module, rank: int, lora_alpha: float):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.rank = rank
        
        for param in self.qkv.parameters():
            param.requires_grad = False
            
        self.lora_A_q = nn.Linear(self.dim, rank, bias=False)
        self.lora_B_q = nn.Linear(rank, self.dim, bias=False)
        self.lora_A_k = nn.Linear(self.dim, rank, bias=False)
        self.lora_B_k = nn.Linear(rank, self.dim, bias=False)
        
        self.scaling_q = lora_alpha / rank
        self.scaling_k = lora_alpha / rank
        
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.kaiming_uniform_(self.lora_A_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        delta_q = self.lora_B_q(self.lora_A_q(x)) * self.scaling_q
        delta_k = self.lora_B_k(self.lora_A_k(x)) * self.scaling_k
        return torch.cat([q + delta_q, k + delta_k, v], dim=-1)

class DINOv2Model(nn.Module):
    VIT_CONFIGS = {
        'vits14': {'dim': 384, 'patch_size': 14},
        'vitb14': {'dim': 768, 'patch_size': 14},
        'vitl14': {'dim': 1024, 'patch_size': 14},
        'vitg14': {'dim': 1536, 'patch_size': 14}
    }
    
    def __init__(self, num_classes: int, model_name: str = 'dinov2_vits14',
                 train_method: str = 'lora', output_dir: str = None):
        super().__init__()
        self.output_dir = output_dir
        self.analyzer = ParameterAnalyzer(output_dir) if output_dir else None
        self.train_method = train_method
        self.model_name = model_name
        
        self.vit_variant = model_name.split('_')[-1]
        if self.vit_variant not in self.VIT_CONFIGS:
            raise ValueError(f"Invalid ViT variant: {self.vit_variant}")
            
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Freeze parameters based on training method
        if self.train_method in ['lora', 'linear']:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            
        # Apply LoRA if using LoRA method
        if self.train_method == 'lora':
            for block in self.dinov2.blocks:
                orig_qkv = block.attn.qkv
                block.attn.qkv = LoRA_qkv(
                    orig_qkv,
                    rank=CONFIG["lora_rank"],
                    lora_alpha=CONFIG["lora_alpha"]
                )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.VIT_CONFIGS[self.vit_variant]['dim'], 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        if self.analyzer:
            self.analyzer.log_parameters(self, "DINOv2 Model Initialization")
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dinov2.prepare_tokens_with_masks(x)
        for blk in self.dinov2.blocks:
            x = blk(x)
        x = self.dinov2.norm(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(features[:, 0])
    
    def save_trainable_weights(self, path: str):
        if self.train_method == 'full':
            torch.save(self.state_dict(), path)
        else:
            trainable_state = {
                k: v for k, v in self.state_dict().items()
                if any(s in k for s in ['lora_', 'classifier'])
            }
            torch.save(trainable_state, path)

def compute_metrics(model, dataloader, device, class_names, output_dir=None, phase="val"):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            scores = torch.softmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = running_loss / len(dataloader)
    
    if output_dir:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 14}, linewidths=0.5, linecolor='gray')
        plt.title(f'', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(fontsize=13, rotation=45, ha='right')
        plt.yticks(fontsize=13, rotation=45, va='center_baseline')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{phase}_confusion_matrix.jpg'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(6, 6))
        colors = plt.colormaps.get_cmap('tab10').resampled(len(class_names))
        
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve((np.array(all_labels) == i).astype(int), 
                                  np.array(all_scores)[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors(i), lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curves', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{phase}_roc_curve.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(6, 6))
        for i in range(len(class_names)):
            precision, recall, _ = precision_recall_curve(
                (np.array(all_labels) == i).astype(int),
                np.array(all_scores)[:, i]
            )
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, color=colors(i), lw=2,
                     label=f'{class_names[i]} (AUC = {pr_auc:.2f})')
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(f'Precision-Recall Curves', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{phase}_precision_recall_curve.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return avg_loss, accuracy, all_labels, all_preds, all_scores
    
def plot_tsne(model, dataloader, device, class_names, output_dir, phase="train", n_samples=1000):
    """Generate t-SNE visualization of features"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(dataloader):
            if i * dataloader.batch_size >= n_samples:
                break
            images = images.to(device)
            batch_features = model.forward_features(images)[:, 0]
            features.append(batch_features.cpu())
            labels.append(batch_labels.cpu())
    
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    tsne = TSNE(n_components=2, random_state=CONFIG["seed"], perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(6, 6))
    palette = sns.color_palette("hsv", len(class_names))
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=[class_names[l] for l in labels],
        palette=palette,
        alpha=0.7,
        s=50,
        edgecolor='none'
    )
    
    plt.title(f'', fontsize=14)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize=14, ncol=2)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{phase}_tsne.jpg")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE plot to {plot_path}")

def train_model(train_dir, val_dir, test_dir, model_name, num_epochs=10,
               batch_size=32, lr=1e-4, method='lora', few_shot_k=None):
    set_seed(CONFIG["seed"])
    output_dir = create_output_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ImageFolderDataset(train_dir, transform=transform, few_shot_k=few_shot_k)
    val_dataset = ImageFolderDataset(val_dir, transform=transform)
    test_dataset = ImageFolderDataset(test_dir, transform=transform)
    class_names = train_dataset.classes
    
    print(f"\nTraining Configuration:")
    print(f"Method: {method.upper()}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Device: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(CONFIG["seed"]))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = DINOv2Model(
        num_classes=len(class_names),
        model_name=model_name,
        train_method=method,
        output_dir=output_dir
    ).to(device)
    
    # Parameter analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n=== Parameter Analysis ===")
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    print("\nGenerating initial t-SNE...")
    plot_tsne(model, train_loader, device, class_names, output_dir, "pretrain_train")
    plot_tsne(model, val_loader, device, class_names, output_dir, "pretrain_val")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'epoch': [], 
        'train_loss': [], 
        'val_loss': [],
        'train_acc': [], 
        'val_acc': [],
        'epoch_time': []
    }
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        train_loss, train_acc, _, _, _ = compute_metrics(model, train_loader, device, class_names)
        val_loss, val_acc, _, _, _ = compute_metrics(model, val_loader, device, class_names, output_dir, "val")
        
        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_trainable_weights(best_model_path)
            print(f"New best model saved with val accuracy: {val_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    model.load_state_dict(torch.load(best_model_path), strict=False)
    
    print("\nGenerating post-training visualizations:")
    try:
        print("  - Training set t-SNE...")
        plot_tsne(model, train_loader, device, class_names, output_dir, "posttrain_train")
        
        print("  - Validation set t-SNE...")
        plot_tsne(model, val_loader, device, class_names, output_dir, "posttrain_val")
        
        print("  - Test set t-SNE...")
        plot_tsne(model, test_loader, device, class_names, output_dir, "posttrain_test")
        
    except Exception as e:
        print(f"t-SNE visualization error: {str(e)}")
    
    test_loss, test_acc, test_labels, test_preds, _ = compute_metrics(
        model, test_loader, device, class_names, output_dir, "test"
    )
    
    test_report = classification_report(
        test_labels, test_preds, target_names=class_names, output_dict=True
    )
    pd.DataFrame(test_report).transpose().to_csv(
        os.path.join(output_dir, "detailed_classification_report.csv"), index=True
    )

    final_metrics = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'method': method,
        'model': model_name,
        'total_training_time_sec': total_time,
        'avg_epoch_time_sec': total_time/num_epochs,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'batch_size': batch_size,
        'few_shot_k': few_shot_k if few_shot_k else 'full_dataset'
    }

    pd.DataFrame(final_metrics, index=[0]).to_csv(
        os.path.join(output_dir, "final_metrics.csv"), index=False
    )
    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"), index=False
    )

    print("\n=== Final Results ===")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv2 with different methods")
    parser.add_argument('--method', type=str, choices=['lora', 'linear', 'full'], required=True)
    parser.add_argument('--model', type=str, default='dinov2_vits14',
                      choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--few_shot_k', type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(CONFIG["seed"])
    
    print(f"Initializing {args.method.upper()} training...")
    train_model(
        train_dir=CONFIG["train_dir"],
        val_dir=CONFIG["val_dir"],
        test_dir=CONFIG["test_dir"],
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        method=args.method,
        few_shot_k=args.few_shot_k
    )