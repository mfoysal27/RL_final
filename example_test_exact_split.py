#!/usr/bin/env python
"""
Example: How to extract and test on the EXACT training test split
"""

import torch
import torch.nn.functional as F
from test_model import extract_training_test_split, ModelTester
import numpy as np
from sklearn.metrics import accuracy_score
import os

def example_extract_exact_test_split():
    """Example of extracting the exact training test split"""
    
    # Path to your training data directory (same as used during training)
    data_dir = "path/to/your/training/data"  # Update this path
    
    print("🎯 Example: Extracting EXACT training test split")
    print("="*50)
    
    try:
        # Extract the exact same test split used during training
        # Use the SAME ratios you used during training!
        test_loader, num_classes, class_names = extract_training_test_split(
            data_dir=data_dir,
            train_ratio=0.8,  # Same as training
            val_ratio=0.1,    # Same as training  
            test_ratio=0.1,   # Same as training
            batch_size=1
        )
        
        print(f"✅ Successfully extracted exact test split!")
        print(f"   Test samples: {len(test_loader.dataset)}")
        print(f"   Classes: {num_classes}")
        print(f"   Class names: {class_names}")
        
        return test_loader, num_classes, class_names
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure to:")
        print("   1. Update the data_dir path to your training data")
        print("   2. Use the exact same ratios as during training")
        print("   3. Ensure the data structure has images/ and masks/ folders")
        return None, None, None

def example_test_model_on_exact_split():
    """Example of testing a model on the exact training test split"""
    
    # Update these paths
    model_path = "path/to/your/trained/model.pth"
    data_dir = "path/to/your/training/data"
    
    print("\n🧠 Example: Testing model on EXACT training test split")
    print("="*60)
    
    try:
        # Create ModelTester with training split option
        tester = ModelTester(
            model_path=model_path,
            test_data_dir=data_dir,
            output_dir="exact_test_results",
            has_ground_truth=True,
            use_training_split=True  # 🎯 This enables exact training test split
        )
        
        # Run complete testing
        report_path = tester.run_complete_test(save_predictions=True)
        
        print(f"✅ Testing completed!")
        print(f"📊 Results: {report_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure to:")
        print("   1. Update model_path to your trained model")
        print("   2. Update data_dir to your training data")

def example_manual_evaluation():
    """Example of manual evaluation loop on exact test split"""
    
    data_dir = "path/to/your/training/data"
    model_path = "path/to/your/trained/model.pth"
    
    print("\n🔬 Example: Manual evaluation on exact test split")
    print("="*50)
    
    # Extract test split
    test_loader, num_classes, class_names = extract_training_test_split(data_dir)
    
    if test_loader is None:
        print("❌ Could not extract test split")
        return
    
    # Load your model (simplified example)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create your model here (adjust based on your model type)
        # model = YourModel(num_classes=num_classes)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model = model.to(device)
        # model.eval()
        
        print("⚠️  Model loading code needs to be customized for your model type")
        print("   See the full ModelTester class for complete model loading logic")
        
        # Example evaluation loop
        all_predictions = []
        all_ground_truths = []
        
        print(f"\n🔍 Evaluating on {len(test_loader)} test samples...")
        
        # with torch.no_grad():
        #     for batch_idx, (images, masks) in enumerate(test_loader):
        #         images = images.to(device)
        #         
        #         # Forward pass
        #         outputs = model(images)
        #         predictions = torch.argmax(outputs, dim=1)
        #         
        #         # Store results
        #         all_predictions.extend(predictions.cpu().numpy().flatten())
        #         all_ground_truths.extend(masks.numpy().flatten())
        #         
        #         if batch_idx % 10 == 0:
        #             print(f"   Processed {batch_idx+1}/{len(test_loader)} batches")
        
        # Calculate metrics
        # accuracy = accuracy_score(all_ground_truths, all_predictions)
        # print(f"\n✅ Test Accuracy: {accuracy:.4f}")
        
        print("💡 Uncomment and customize the evaluation code above")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    print("📚 Examples: Testing on EXACT training test split")
    print("="*60)
    
    print("\n🎯 Available examples:")
    print("1. extract_exact_test_split() - Extract test split only")
    print("2. test_model_on_exact_split() - Full testing with ModelTester")
    print("3. manual_evaluation() - Custom evaluation loop")
    
    print("\n💡 Update the file paths in the examples before running!")
    print("💡 Use the same ratios that were used during training!")
    
    # Uncomment to run examples:
    # example_extract_exact_test_split()
    # example_test_model_on_exact_split()
    # example_manual_evaluation() 