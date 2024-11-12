import os
import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pytorch_lightning as pl
import json
from review_classification_train import SentimentClassifier, SentimentDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Test the trained sentiment classifier')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing the trained model and configuration')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for testing')
    return parser.parse_args()

def load_config(model_dir):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def load_best_model_info(model_dir):
    info_path = os.path.join(model_dir, 'best_model_info.json')
    with open(info_path, 'r') as f:
        return json.load(f)

def main(args):
    # Load training configuration
    config = load_config(args.model_dir)
    best_model_info = load_best_model_info(args.model_dir)
    
    # Load tokenizer
    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_from_disk(os.path.join(args.model_dir, 'test_dataset'))
    
    # Create test dataloader
    test_data = SentimentDataset(test_dataset, tokenizer, max_length=config['max_length'])
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Load best model
    print(f"Loading best model from fold {best_model_info['fold']}...")
    model = SentimentClassifier.load_from_checkpoint(
        best_model_info['path'],
        model_name=config['model_name']
    )
    model.eval()
    model.cuda()
    
    # Test the model
    print("Testing model...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to GPU
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels']
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute and print metrics
    print("\nTest Results:")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Save results
    results = {
        'classification_report': classification_report(all_labels, all_preds, output_dict=True),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    results_path = os.path.join(args.model_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    print(f"\nTest results saved to: {results_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)