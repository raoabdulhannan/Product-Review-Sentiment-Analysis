import os
import argparse
import GPUtil
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datasets import load_dataset, Dataset
import json

def check_gpu_usage():
    """Check if GPU is available and not too busy"""
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f'GPU {gpu.id} - {gpu.name}:')
        print(f'Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB')
        print(f'GPU Utilization: {gpu.load*100}%')
    
    if not gpus:
        raise RuntimeError("No GPU found!")
    
    return gpus[0].memoryUsed / gpus[0].memoryTotal

def parse_args():
    parser = argparse.ArgumentParser(description='Train a sentiment classifier on Amazon reviews')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                      help='Name of the pretrained model to use')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=3,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--n_splits', type=int, default=5,
                      help='Number of folds for cross-validation')
    parser.add_argument('--output_dir', type=str, default='model_outputs',
                      help='Directory to save model outputs')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    parser.add_argument('--accumulate_grad_batches', type=int, default=4,
                      help='Number of batches to accumulate gradients')
    parser.add_argument('--use_wandb', action='store_true',
                      help='Use Weights & Biases for logging')
    return parser.parse_args()

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = f"{item['review_title']} {item['review_text']}"
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        label = item['class_index'] - 1

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(pl.LightningModule):
    def __init__(self, model_name='roberta-base', num_classes=5, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Load config and model with gradient checkpointing enabled
        config = AutoConfig.from_pretrained(model_name)
        config.gradient_checkpointing = True
        self.model = AutoModel.from_pretrained(model_name, config=config)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
        self.val_predictions = []
        self.val_labels = []

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def training_step(self, batch, batch_idx):
        # Clear cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(outputs, batch['labels'])
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == batch['labels']).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(outputs, batch['labels'])
        preds = torch.argmax(outputs, dim=1)
        
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_labels.extend(batch['labels'].cpu().numpy())
        
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        val_accuracy = accuracy_score(self.val_labels, self.val_predictions)
        val_f1 = f1_score(self.val_labels, self.val_predictions, average='weighted')
        
        self.log('val_accuracy', val_accuracy, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        
        self.val_predictions = []
        self.val_labels = []

    def configure_optimizers(self):
        # Use AdamW with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        return optimizer

def main(args):
    # Check GPU usage but don't error out
    gpu_usage = check_gpu_usage()
    print(f"Current GPU memory usage: {gpu_usage*100:.2f}%")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes")
    
    # Split into train and test
    splits = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = splits['train']
    test_dataset = splits['test']
    
    # Save test dataset for later use
    test_path = os.path.join(args.output_dir, 'test_dataset')
    test_dataset.save_to_disk(test_path)
    
    # Initialize tokenizer
    print(f"Initializing tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Convert train dataset to pandas for k-fold
    train_df = train_dataset.to_pandas()
    
    # Initialize K-fold
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    # Store best model info
    best_model_info = {
        'fold': None,
        'accuracy': 0.0,
        'path': None
    }
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df), 1):
        print(f"\nTraining Fold {fold}/{args.n_splits}")
        
        # Split data
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]
        
        # Create datasets
        fold_train_dataset = SentimentDataset(
            Dataset.from_pandas(fold_train_df),
            tokenizer,
            max_length=args.max_length
        )
        fold_val_dataset = SentimentDataset(
            Dataset.from_pandas(fold_val_df),
            tokenizer,
            max_length=args.max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            fold_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,  # Reduced number of workers
            pin_memory=True
        )
        val_loader = DataLoader(
            fold_val_dataset,
            batch_size=args.batch_size,
            num_workers=1,  # Reduced number of workers
            pin_memory=True
        )
        
        # Initialize model
        model = SentimentClassifier(
            model_name=args.model_name,
            learning_rate=args.learning_rate
        )
        
        # Setup callbacks
        fold_dir = os.path.join(args.output_dir, f'fold_{fold}')
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_dir,
            filename='model-{epoch:02d}-{val_accuracy:.4f}',
            monitor='val_accuracy',
            mode='max',
            save_top_k=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            mode='max'
        )
        
        # Initialize trainer with memory optimizations
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator='gpu',
            devices=1,
            callbacks=[checkpoint_callback, early_stopping],
            enable_progress_bar=True,
            logger=WandbLogger(project='amazon-reviews') if args.use_wandb else True,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            precision='16-mixed'  # Use mixed precision training
        )
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        # Update best model info
        if checkpoint_callback.best_model_score > best_model_info['accuracy']:
            best_model_info = {
                'fold': fold,
                'accuracy': checkpoint_callback.best_model_score.item(),
                'path': checkpoint_callback.best_model_path
            }
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
    
    # Save best model info
    best_model_info_path = os.path.join(args.output_dir, 'best_model_info.json')
    with open(best_model_info_path, 'w') as f:
        json.dump(best_model_info, f)
    
    print("\nTraining completed!")
    print(f"Best model from fold {best_model_info['fold']}")
    print(f"Best validation accuracy: {best_model_info['accuracy']:.4f}")
    print(f"Best model saved at: {best_model_info['path']}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
