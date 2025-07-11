#!/usr/bin/env python
"""
Model Checkpoint Analysis Tool
Compare sizes and contents of different model checkpoints
"""

import torch
import torch.nn.functional as F
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_checkpoint(path):
    """Analyze a PyTorch checkpoint file"""
    if not os.path.exists(path):
        print(f'❌ File not found: {path}')
        return None
    
    # Get file size
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f'\n📁 File: {os.path.basename(path)}')
    print(f'💾 Size: {size_mb:.2f} MB')
    
    analysis = {
        'file_name': os.path.basename(path),
        'file_size_mb': size_mb,
        'file_path': path,
        'components': {}
    }
    
    # Load and analyze contents
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        print(f'🔍 Checkpoint Contents:')
        for key in checkpoint.keys():
            print(f'   📋 {key}')
            
            if key == 'model_state_dict':
                # Analyze model parameters
                model_state = checkpoint[key]
                param_count = sum(p.numel() for p in model_state.values())
                param_size_mb = sum(p.numel() * p.element_size() for p in model_state.values()) / (1024 * 1024)
                
                analysis['components']['model'] = {
                    'parameter_count': param_count,
                    'size_mb': param_size_mb,
                    'layer_count': len(model_state),
                    'layers': list(model_state.keys())
                }
                
                print(f'      🧠 Model parameters: {param_count:,}')
                print(f'      📊 Model size: {param_size_mb:.2f} MB')
                print(f'      🔧 Layers: {len(model_state)}')
                
                # Analyze layer types
                layer_types = {}
                for layer_name in model_state.keys():
                    layer_type = layer_name.split('.')[0] if '.' in layer_name else layer_name
                    layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
                
                print(f'      🏗️  Layer types: {dict(layer_types)}')
                analysis['components']['model']['layer_types'] = layer_types
                
            elif key == 'training_history':
                history = checkpoint[key]
                analysis['components']['training_history'] = {
                    'available': True,
                    'epochs': len(history) if isinstance(history, list) else 'N/A',
                    'keys': list(history.keys()) if isinstance(history, dict) else 'N/A'
                }
                print(f'      📈 Training history available')
                
            elif key == 'feedback_history':
                feedback = checkpoint[key]
                analysis['components']['feedback_history'] = {
                    'available': True,
                    'entries': len(feedback) if isinstance(feedback, list) else 'N/A'
                }
                print(f'      🎯 Feedback history: {len(feedback) if isinstance(feedback, list) else "N/A"} entries')
                
            elif key == 'rl_epoch':
                print(f'      🔄 RL epoch: {checkpoint[key]}')
                analysis['components']['rl_epoch'] = checkpoint[key]
                
            elif key == 'model_info':
                model_info = checkpoint[key]
                analysis['components']['model_info'] = model_info
                print(f'      ℹ️  Model info available')
                if 'model_architecture' in model_info:
                    print(f'          Architecture: {model_info["model_architecture"]}')
                if 'rl_training' in model_info:
                    rl_info = model_info['rl_training']
                    print(f'          RL updates: {rl_info.get("total_updates", "N/A")}')
                    print(f'          Learning rate: {rl_info.get("learning_rate", "N/A")}')
                
            elif key in ['num_classes', 'model_type', 'config']:
                analysis['components'][key] = checkpoint[key]
                print(f'      📝 {key}: {checkpoint[key]}')
        
        return analysis
        
    except Exception as e:
        print(f'❌ Error loading checkpoint: {e}')
        return None

def compare_models(model1_path, model2_path):
    """Compare two model checkpoints"""
    print(f'\n🔍 COMPARING MODELS')
    print('=' * 50)
    
    # Load both models
    try:
        checkpoint1 = torch.load(model1_path, map_location='cpu')
        checkpoint2 = torch.load(model2_path, map_location='cpu')
        
        # Extract model states
        state1 = checkpoint1.get('model_state_dict', checkpoint1)
        state2 = checkpoint2.get('model_state_dict', checkpoint2)
        
        if not isinstance(state1, dict) or not isinstance(state2, dict):
            print('❌ Could not extract model states for comparison')
            return
        
        print(f'📊 PARAMETER COMPARISON:')
        
        # Compare parameter counts
        params1 = sum(p.numel() for p in state1.values())
        params2 = sum(p.numel() for p in state2.values())
        
        print(f'   Model 1: {params1:,} parameters')
        print(f'   Model 2: {params2:,} parameters')
        print(f'   Difference: {params2 - params1:,} parameters')
        
        # Compare architectures
        layers1 = set(state1.keys())
        layers2 = set(state2.keys())
        
        common_layers = layers1 & layers2
        unique_to_1 = layers1 - layers2
        unique_to_2 = layers2 - layers1
        
        print(f'\n🏗️  ARCHITECTURE COMPARISON:')
        print(f'   Common layers: {len(common_layers)}')
        print(f'   Unique to model 1: {len(unique_to_1)}')
        print(f'   Unique to model 2: {len(unique_to_2)}')
        
        if unique_to_1:
            print(f'   Model 1 only: {list(unique_to_1)[:5]}...' if len(unique_to_1) > 5 else f'   Model 1 only: {list(unique_to_1)}')
        if unique_to_2:
            print(f'   Model 2 only: {list(unique_to_2)[:5]}...' if len(unique_to_2) > 5 else f'   Model 2 only: {list(unique_to_2)}')
        
        # Compare weights for common layers
        if common_layers:
            print(f'\n🔄 WEIGHT CHANGE ANALYSIS:')
            
            total_change = 0
            max_change = 0
            max_change_layer = ''
            layer_changes = {}
            
            for layer_name in common_layers:
                if layer_name in state1 and layer_name in state2:
                    w1 = state1[layer_name]
                    w2 = state2[layer_name]
                    
                    if w1.shape == w2.shape:
                        # Calculate change metrics
                        change = torch.norm(w2 - w1).item()
                        relative_change = change / torch.norm(w1).item() if torch.norm(w1).item() > 0 else 0
                        
                        # Cosine similarity
                        similarity = F.cosine_similarity(w1.flatten().unsqueeze(0), w2.flatten().unsqueeze(0)).item()
                        
                        layer_changes[layer_name] = {
                            'absolute_change': change,
                            'relative_change': relative_change,
                            'cosine_similarity': similarity
                        }
                        
                        total_change += change
                        
                        if relative_change > max_change:
                            max_change = relative_change
                            max_change_layer = layer_name
            
            avg_change = total_change / len(common_layers) if common_layers else 0
            
            print(f'   Average weight change: {avg_change:.6f}')
            print(f'   Maximum change: {max_change:.6f} in layer "{max_change_layer}"')
            
            # Show top 5 most changed layers
            sorted_changes = sorted(layer_changes.items(), key=lambda x: x[1]['relative_change'], reverse=True)
            print(f'   Top 5 most changed layers:')
            for i, (layer, metrics) in enumerate(sorted_changes[:5]):
                print(f'      {i+1}. {layer}: {metrics["relative_change"]:.6f} change, {metrics["cosine_similarity"]:.4f} similarity')
            
            # Interpretation
            print(f'\n💡 INTERPRETATION:')
            if avg_change < 0.001:
                print('   ✅ Very small changes - models are very similar')
            elif avg_change < 0.01:
                print('   ⚠️ Moderate changes - some learning occurred')
            else:
                print('   🚨 Large changes - significant model modification')
                print('   ⚠️ Possible catastrophic forgetting if this is an RL model')
        
        # Compare other metadata
        print(f'\n📋 METADATA COMPARISON:')
        
        # Training info
        if 'training_history' in checkpoint1 and 'training_history' in checkpoint2:
            hist1 = checkpoint1['training_history']
            hist2 = checkpoint2['training_history']
            print(f'   Training history: Model 1 has {len(hist1) if isinstance(hist1, list) else "N/A"}, Model 2 has {len(hist2) if isinstance(hist2, list) else "N/A"}')
        
        # RL info
        if 'rl_epoch' in checkpoint1 or 'rl_epoch' in checkpoint2:
            rl1 = checkpoint1.get('rl_epoch', 'N/A')
            rl2 = checkpoint2.get('rl_epoch', 'N/A')
            print(f'   RL epochs: Model 1: {rl1}, Model 2: {rl2}')
        
        # Model info
        if 'model_info' in checkpoint1 or 'model_info' in checkpoint2:
            info1 = checkpoint1.get('model_info', {})
            info2 = checkpoint2.get('model_info', {})
            
            if 'rl_training' in info1 and 'rl_training' in info2:
                rl_info1 = info1['rl_training']
                rl_info2 = info2['rl_training']
                print(f'   RL updates: Model 1: {rl_info1.get("total_updates", "N/A")}, Model 2: {rl_info2.get("total_updates", "N/A")}')
                print(f'   Learning rates: Model 1: {rl_info1.get("learning_rate", "N/A")}, Model 2: {rl_info2.get("learning_rate", "N/A")}')
        
    except Exception as e:
        print(f'❌ Error comparing models: {e}')

def save_analysis_report(analyses, output_path):
    """Save analysis report to JSON file"""
    try:
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'models_analyzed': len(analyses),
            'analyses': analyses
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f'✅ Analysis report saved to: {output_path}')
        
    except Exception as e:
        print(f'❌ Error saving report: {e}')

def main():
    """Main analysis function"""
    print('🔍 Model Checkpoint Analysis Tool')
    print('=' * 50)
    
    # Get model files from current directory
    model_files = []
    for ext in ['*.pth', '*.pt']:
        model_files.extend(Path('.').glob(ext))
    
    if not model_files:
        print('❌ No model files found in current directory')
        print('   Looking for files with extensions: .pth, .pt')
        return
    
    print(f'📁 Found {len(model_files)} model files:')
    for i, file in enumerate(model_files):
        print(f'   {i+1}. {file.name}')
    
    # Analyze each model
    analyses = []
    for model_file in model_files:
        analysis = analyze_checkpoint(str(model_file))
        if analysis:
            analyses.append(analysis)
    
    # Compare models if we have more than one
    if len(analyses) >= 2:
        print(f'\n🔍 Would you like to compare models? (y/n)')
        if input().lower().startswith('y'):
            print('Select two models to compare:')
            for i, analysis in enumerate(analyses):
                print(f'   {i+1}. {analysis["file_name"]}')
            
            try:
                idx1 = int(input('First model (number): ')) - 1
                idx2 = int(input('Second model (number): ')) - 1
                
                if 0 <= idx1 < len(analyses) and 0 <= idx2 < len(analyses) and idx1 != idx2:
                    compare_models(analyses[idx1]['file_path'], analyses[idx2]['file_path'])
                else:
                    print('❌ Invalid selection')
                    
            except ValueError:
                print('❌ Invalid input')
    
    # Save report
    if analyses:
        save_analysis_report(analyses, 'model_analysis_report.json')
    
    print(f'\n✅ Analysis complete!')

if __name__ == '__main__':
    main() 