#!/usr/bin/env python
"""
Create Confusion Matrix Visualization from Test Report JSON
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path

def load_test_report(json_path: str):
    """Load test report from JSON file"""
    try:
        with open(json_path, 'r') as f:
            report = json.load(f)
        print(f"✅ Loaded test report from: {json_path}")
        return report
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return None

def extract_confusion_matrix_data(report):
    """Extract confusion matrix and class information from report"""
    try:
        # Get confusion matrix
        confusion_matrix = np.array(report['metrics']['confusion_matrix'])
        
        # Get class names from class_metrics
        class_metrics = report['metrics']['class_metrics']
        
        # Create class labels list
        class_labels = []
        for class_id in range(len(confusion_matrix)):
            if str(class_id) in class_metrics:
                class_name = class_metrics[str(class_id)]['name']
                class_labels.append(f"{class_id}: {class_name}")
            else:
                class_labels.append(f"Class {class_id}")
        
        print(f"✅ Extracted confusion matrix: {confusion_matrix.shape}")
        print(f"   Classes: {len(class_labels)}")
        
        return confusion_matrix, class_labels
        
    except Exception as e:
        print(f"❌ Error extracting confusion matrix data: {e}")
        return None, None

def create_confusion_matrix_plot(confusion_matrix, class_labels, output_path="confusion_matrix.png", 
                                title="Confusion Matrix", figsize=(12, 10)):
    """Create and save confusion matrix visualization"""
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Number of Pixels'}
        )
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {output_path}")
        
        # Show plot
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating confusion matrix plot: {e}")

def create_normalized_confusion_matrix(confusion_matrix, class_labels, output_path="confusion_matrix_normalized.png"):
    """Create normalized confusion matrix (percentages)"""
    try:
        # Normalize by rows (true classes)
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Handle NaN values (classes with no samples)
        cm_normalized = np.nan_to_num(cm_normalized)
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap with percentages
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Percentage of True Class'}
        )
        
        plt.title('Normalized Confusion Matrix (Percentages)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Normalized confusion matrix saved to: {output_path}")
        
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating normalized confusion matrix: {e}")

def print_confusion_matrix_stats(confusion_matrix, class_labels):
    """Print confusion matrix statistics"""
    try:
        print(f"\n📊 Confusion Matrix Statistics:")
        print(f"   Shape: {confusion_matrix.shape}")
        print(f"   Total predictions: {confusion_matrix.sum():,}")
        print(f"   Number of classes: {len(class_labels)}")
        
        # Calculate per-class accuracy
        print(f"\n🎯 Per-Class Accuracy:")
        for i, class_label in enumerate(class_labels):
            if confusion_matrix[i].sum() > 0:
                accuracy = confusion_matrix[i, i] / confusion_matrix[i].sum()
                print(f"   {class_label}: {accuracy:.2%} ({confusion_matrix[i, i]:,}/{confusion_matrix[i].sum():,})")
            else:
                print(f"   {class_label}: No samples")
        
        # Overall accuracy
        overall_accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()
        print(f"\n🎯 Overall Accuracy: {overall_accuracy:.2%}")
        
        # Most confused classes
        print(f"\n❓ Most Confused Class Pairs:")
        cm_copy = confusion_matrix.copy()
        np.fill_diagonal(cm_copy, 0)  # Remove diagonal (correct predictions)
        
        # Find top 5 confusion pairs
        top_confusions = []
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                if i != j and cm_copy[i, j] > 0:
                    top_confusions.append((cm_copy[i, j], i, j))
        
        top_confusions.sort(reverse=True)
        for count, true_idx, pred_idx in top_confusions[:5]:
            true_class = class_labels[true_idx].split(': ')[1] if ': ' in class_labels[true_idx] else class_labels[true_idx]
            pred_class = class_labels[pred_idx].split(': ')[1] if ': ' in class_labels[pred_idx] else class_labels[pred_idx]
            print(f"   {true_class} → {pred_class}: {count:,} pixels")
        
    except Exception as e:
        print(f"❌ Error calculating statistics: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create confusion matrix from test report JSON")
    parser.add_argument("--json", "-j", 
                       default="test_results/test_report.json",
                       help="Path to test report JSON file")
    parser.add_argument("--output", "-o", 
                       default="confusion_matrix_from_json.png",
                       help="Output path for confusion matrix image")
    parser.add_argument("--normalized", "-n", 
                       action="store_true",
                       help="Also create normalized confusion matrix")
    parser.add_argument("--stats", "-s", 
                       action="store_true", 
                       default=True,
                       help="Print confusion matrix statistics")
    
    args = parser.parse_args()
    
    print("🔍 Creating Confusion Matrix from JSON Report")
    print("=" * 50)
    
    # Load test report
    report = load_test_report(args.json)
    if report is None:
        return
    
    # Extract confusion matrix data
    confusion_matrix, class_labels = extract_confusion_matrix_data(report)
    if confusion_matrix is None:
        return
    
    # Print statistics
    if args.stats:
        print_confusion_matrix_stats(confusion_matrix, class_labels)
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create confusion matrix plot
    create_confusion_matrix_plot(
        confusion_matrix, 
        class_labels, 
        args.output,
        title=f"Confusion Matrix - {os.path.basename(args.json)}"
    )
    
    # Create normalized confusion matrix if requested
    if args.normalized:
        normalized_path = args.output.replace('.png', '_normalized.png')
        create_normalized_confusion_matrix(confusion_matrix, class_labels, normalized_path)
    
    print(f"\n✅ Confusion matrix visualization completed!")

if __name__ == "__main__":
    main() 