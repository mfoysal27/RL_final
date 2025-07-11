"""
Model Factory for Biological Image Segmentation
Supports multiple architectures: UNet, HybridUNetGNN, TransUNet, SwinUNETR, SAM
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Import models conditionally to handle missing dependencies
AVAILABLE_MODEL_CLASSES = {}

# Always available models
try:
    from models.unet import UNet
    AVAILABLE_MODEL_CLASSES['unet'] = UNet
except ImportError as e:
    print(f"Warning: Could not import UNet: {e}")

# Conditional imports for models with dependencies
try:
    from models.hybrid_model import HybridUNetGNN
    AVAILABLE_MODEL_CLASSES['hybrid_unet_gnn'] = HybridUNetGNN
except ImportError as e:
    print(f"Info: HybridUNetGNN not available (missing torch_geometric): {e}")

try:
    from models.transunet import TransUNet
    AVAILABLE_MODEL_CLASSES['transunet'] = TransUNet
except ImportError as e:
    print(f"Info: TransUNet not available: {e}")

try:
    from models.swin_unetr import SwinUNETR
    AVAILABLE_MODEL_CLASSES['swin_unetr'] = SwinUNETR
except ImportError as e:
    print(f"Info: SwinUNETR not available: {e}")

try:
    from models.cellpose_sam_model import CellposeSAMModel
    AVAILABLE_MODEL_CLASSES['cellpose_sam'] = CellposeSAMModel
except ImportError as e:
    print(f"Info: CellposeSAM not available: {e}")

class ModelFactory:
    """Factory for creating and managing different model architectures"""
    
    AVAILABLE_MODELS = AVAILABLE_MODEL_CLASSES
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get list of available models with descriptions"""
        model_descriptions = {
            'unet': 'UNet - Classic architecture (Fast, Low Memory: ~2GB VRAM)',
            'hybrid_unet_gnn': 'HybridUNetGNN - UNet + Graph Neural Networks (Medium: ~3GB VRAM)',
            'transunet': 'TransUNet - Transformer + CNN backbone (High Performance: ~8GB VRAM)',
            'swin_unetr': 'SwinUNETR - Swin Transformer (State-of-the-art: ~6GB VRAM)',
            'cellpose_sam': 'CellposeSAM - Segment Anything Model (Very Advanced: ~12GB VRAM)'
        }
        
        available = {}
        for model_type in cls.AVAILABLE_MODELS.keys():
            if model_type in model_descriptions:
                available[model_type] = model_descriptions[model_type]
        
        return available
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Create a model based on type and configuration"""
        
        if model_type not in cls.AVAILABLE_MODELS:
            available_models = list(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Model type '{model_type}' not available. Available: {available_models}")
        
        model_class = cls.AVAILABLE_MODELS[model_type]
        
        # Extract common parameters
        common_params = {
            'in_channels': config.get('in_channels', 1),
            'num_classes': config.get('num_classes', 6),
        }
        
        # Model-specific parameter handling
        if model_type == 'unet':
            params = {
                **common_params,
                'features': config.get('features', [64, 128, 256, 512])
            }
        
        elif model_type == 'hybrid_unet_gnn':
            params = {
                **common_params,
                'features': config.get('features', [64, 128, 256, 512]),
                'gnn_hidden_channels': config.get('gnn_hidden_channels', 64)
            }
        
        elif model_type == 'transunet':
            params = {
                **common_params,
                'img_size': config.get('img_size', 256),
                'patch_size': config.get('patch_size', 16),
                'embed_dim': config.get('embed_dim', 768),
                'depth': config.get('depth', 12),
                'num_heads': config.get('num_heads', 12),
                'mlp_ratio': config.get('mlp_ratio', 4.0),
                'dropout': config.get('dropout', 0.1),
                'cnn_features': config.get('cnn_features', [64, 128, 256, 512])
            }
        
        elif model_type == 'swin_unetr':
            params = {
                **common_params,
                'img_size': config.get('img_size', 256),
                'patch_size': config.get('patch_size', 4),
                'embed_dim': config.get('embed_dim', 96),
                'depths': config.get('depths', [2, 2, 6, 2]),
                'num_heads': config.get('num_heads', [3, 6, 12, 24]),
                'window_size': config.get('window_size', 7),
                'mlp_ratio': config.get('mlp_ratio', 4.0),
                'drop_rate': config.get('drop_rate', 0.0),
                'attn_drop_rate': config.get('attn_drop_rate', 0.0),
                'drop_path_rate': config.get('drop_path_rate', 0.1)
            }
        
        elif model_type == 'cellpose_sam':
            params = {
                'num_classes': config.get('num_classes', 6),
                'sam_model_type': config.get('sam_model_type', 'vit_b'),
                'freeze_sam': config.get('freeze_sam', True),
                'gnn_hidden_channels': config.get('gnn_hidden_channels', 128),
                'device': config.get('device', None)
            }
        
        try:
            model = model_class(**params)
            print(f"Successfully created {model_type} with parameters: {params}")
            return model
        except Exception as e:
            print(f"Error creating {model_type}: {e}")
            raise
    
    @classmethod
    def get_default_config(cls, model_type: str) -> Dict[str, Any]:
        """Get default configuration for a model type"""
        
        base_config = {
            'in_channels': 1,
            'num_classes': 6,
            'img_size': 256,
        }
        
        model_configs = {
            'unet': {
                **base_config,
                'features': [64, 128, 256, 512]
            },
            
            'hybrid_unet_gnn': {
                **base_config,
                'features': [64, 128, 256, 512],
                'gnn_hidden_channels': 64
            },
            
            'transunet': {
                **base_config,
                'patch_size': 16,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4.0,
                'dropout': 0.1,
                'cnn_features': [64, 128, 256, 512]
            },
            
            'swin_unetr': {
                **base_config,
                'patch_size': 4,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'mlp_ratio': 4.0,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.1
            },
            
            'cellpose_sam': {
                **base_config,
                'sam_model_type': 'vit_b',
                'freeze_sam': True,
                'gnn_hidden_channels': 128
            }
        }
        
        return model_configs.get(model_type, base_config)
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """Get information about a model type"""
        
        info = {
            'unet': {
                'name': 'U-Net',
                'description': 'Classic U-Net architecture for biological image segmentation',
                'complexity': 'Low',
                'memory_usage': 'Low (2GB VRAM)',
                'training_time': 'Fast (2-3 hours)',
                'best_for': 'Quick experiments, baseline comparisons',
                'accuracy': 'Good'
            },
            
            'hybrid_unet_gnn': {
                'name': 'Hybrid UNet-GNN',
                'description': 'U-Net enhanced with Graph Neural Networks for spatial relationships',
                'complexity': 'Medium',
                'memory_usage': 'Medium (3GB VRAM)',
                'training_time': 'Medium (4-5 hours)',
                'best_for': 'Capturing spatial relationships between biological structures',
                'accuracy': 'Very Good'
            },
            
            'transunet': {
                'name': 'TransUNet',
                'description': 'Transformer-based architecture with CNN backbone',
                'complexity': 'High',
                'memory_usage': 'High (8GB VRAM)',
                'training_time': 'Slow (10-12 hours)',
                'best_for': 'Complex biological structures, long-range dependencies',
                'accuracy': 'Excellent'
            },
            
            'swin_unetr': {
                'name': 'Swin-UNETR',
                'description': 'Swin Transformer architecture for segmentation',
                'complexity': 'High',
                'memory_usage': 'High (6GB VRAM)',
                'training_time': 'Slow (8-10 hours)',
                'best_for': 'Multi-scale biological features, hierarchical structures',
                'accuracy': 'Excellent'
            },
            
            'cellpose_sam': {
                'name': 'Cellpose-SAM',
                'description': 'Segment Anything Model adapted for biological images',
                'complexity': 'Very High',
                'memory_usage': 'Very High (12GB VRAM)',
                'training_time': 'Very Slow (15+ hours)',
                'best_for': 'Few-shot learning, diverse biological domains',
                'accuracy': 'Outstanding'
            }
        }
        
        return info.get(model_type, {'name': 'Unknown', 'description': 'No information available'})
    
    @classmethod
    def get_optimizer_params(cls, model: nn.Module, model_type: str, config: Dict[str, Any]) -> Dict:
        """Get optimizer parameters for different model components"""
        
        if model_type == 'unet':
            return {
                'main': {
                    'params': model.parameters(),
                    'lr': config.get('lr', 1e-3)
                }
            }
        
        elif model_type == 'hybrid_unet_gnn':
            return {
                'unet': {
                    'params': model.get_unet_parameters() if hasattr(model, 'get_unet_parameters') else model.parameters(),
                    'lr': config.get('unet_lr', 1e-3)
                },
                'gnn': {
                    'params': model.get_gnn_parameters() if hasattr(model, 'get_gnn_parameters') else [],
                    'lr': config.get('gnn_lr', 5e-4)
                }
            }
        
        elif model_type == 'transunet':
            return {
                'transformer': {
                    'params': model.get_transformer_parameters() if hasattr(model, 'get_transformer_parameters') else model.parameters(),
                    'lr': config.get('transformer_lr', 1e-4)
                },
                'cnn': {
                    'params': model.get_cnn_parameters() if hasattr(model, 'get_cnn_parameters') else [],
                    'lr': config.get('cnn_lr', 1e-3)
                }
            }
        
        elif model_type == 'swin_unetr':
            return {
                'encoder': {
                    'params': model.get_encoder_parameters() if hasattr(model, 'get_encoder_parameters') else model.parameters(),
                    'lr': config.get('encoder_lr', 1e-4)
                },
                'decoder': {
                    'params': model.get_decoder_parameters() if hasattr(model, 'get_decoder_parameters') else [],
                    'lr': config.get('decoder_lr', 1e-3)
                }
            }
        
        elif model_type == 'cellpose_sam':
            optimizer_params = {}
            
            if hasattr(model, 'get_sam_parameters'):
                optimizer_params['sam'] = {
                    'params': model.get_sam_parameters(),
                    'lr': config.get('sam_lr', 1e-5)  # Very low for pre-trained
                }
            
            if hasattr(model, 'get_gnn_parameters'):
                optimizer_params['gnn'] = {
                    'params': model.get_gnn_parameters(),
                    'lr': config.get('gnn_lr', 1e-3)  # Higher for new components
                }
            
            if not optimizer_params:
                optimizer_params['main'] = {
                    'params': model.parameters(),
                    'lr': config.get('lr', 1e-3)
                }
            
            return optimizer_params
        
        else:
            return {
                'main': {
                    'params': model.parameters(),
                    'lr': config.get('lr', 1e-3)
                }
            }
    
    @classmethod
    def setup_optimizers(cls, model: nn.Module, model_type: str, config: Dict[str, Any]) -> Dict:
        """Setup optimizers for different model components"""
        
        optimizer_params = cls.get_optimizer_params(model, model_type, config)
        optimizers = {}
        
        for name, params in optimizer_params.items():
            if not params['params']:  # Skip empty parameter groups
                continue
                
            optimizer_type = config.get('optimizer', 'adam').lower()
            
            if optimizer_type == 'adam':
                optimizers[name] = torch.optim.Adam(
                    params['params'],
                    lr=params['lr'],
                    weight_decay=config.get('weight_decay', 1e-4)
                )
            elif optimizer_type == 'adamw':
                optimizers[name] = torch.optim.AdamW(
                    params['params'],
                    lr=params['lr'],
                    weight_decay=config.get('weight_decay', 1e-4)
                )
            elif optimizer_type == 'sgd':
                optimizers[name] = torch.optim.SGD(
                    params['params'],
                    lr=params['lr'],
                    momentum=config.get('momentum', 0.9),
                    weight_decay=config.get('weight_decay', 1e-4)
                )
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return optimizers
    
    @classmethod
    def setup_schedulers(cls, optimizers: Dict, config: Dict[str, Any]) -> Dict:
        """Setup learning rate schedulers"""
        
        schedulers = {}
        scheduler_type = config.get('scheduler', 'cosine').lower()
        
        for name, optimizer in optimizers.items():
            if scheduler_type == 'cosine':
                schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=config.get('T_0', 20),
                    T_mult=config.get('T_mult', 2)
                )
            elif scheduler_type == 'plateau':
                schedulers[name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=config.get('factor', 0.5),
                    patience=config.get('patience', 10)
                )
            elif scheduler_type == 'step':
                schedulers[name] = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.get('step_size', 30),
                    gamma=config.get('gamma', 0.1)
                )
            elif scheduler_type == 'exponential':
                schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=config.get('gamma', 0.95)
                )
        
        return schedulers
    
    @classmethod
    def get_model_memory_estimate(cls, model_type: str, config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory usage for different models"""
        
        # Base memory estimates in MB (approximate)
        estimates = {
            'unet': {
                'model_size': 50,
                'training_memory': 2000,
                'inference_memory': 500
            },
            
            'hybrid_unet_gnn': {
                'model_size': 75,
                'training_memory': 3000,
                'inference_memory': 750
            },
            
            'transunet': {
                'model_size': 300,
                'training_memory': 8000,
                'inference_memory': 2000
            },
            
            'swin_unetr': {
                'model_size': 200,
                'training_memory': 6000,
                'inference_memory': 1500
            },
            
            'cellpose_sam': {
                'model_size': 500,
                'training_memory': 12000,
                'inference_memory': 3000
            }
        }
        
        base_estimate = estimates.get(model_type, estimates['unet'])
        
        # Scale based on batch size and image size
        batch_size = config.get('batch_size', 4)
        img_size = config.get('img_size', 256)
        scale_factor = (batch_size / 4) * ((img_size / 256) ** 2)
        
        return {
            'model_size_mb': base_estimate['model_size'],
            'training_memory_mb': base_estimate['training_memory'] * scale_factor,
            'inference_memory_mb': base_estimate['inference_memory'] * scale_factor
        }
    
    @classmethod
    def validate_config(cls, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust configuration for a model type"""
        
        validated_config = config.copy()
        
        # Common validations
        validated_config['in_channels'] = max(1, config.get('in_channels', 1))
        validated_config['num_classes'] = max(2, config.get('num_classes', 6))
        validated_config['img_size'] = max(64, config.get('img_size', 256))
        
        # Model-specific validations
        if model_type == 'transunet':
            # Ensure patch size divides image size
            patch_size = config.get('patch_size', 16)
            img_size = validated_config['img_size']
            if img_size % patch_size != 0:
                # Adjust patch size to nearest divisor
                divisors = [i for i in range(4, 65) if img_size % i == 0]
                validated_config['patch_size'] = min(divisors, key=lambda x: abs(x - patch_size))
                print(f"Adjusted patch_size to {validated_config['patch_size']} for img_size {img_size}")
        
        elif model_type == 'swin_unetr':
            # Ensure image size is compatible with patch merging
            img_size = validated_config['img_size']
            patch_size = config.get('patch_size', 4)
            
            # Check if image size allows proper patch merging
            levels = len(config.get('depths', [2, 2, 6, 2]))
            min_size = patch_size * (2 ** levels)
            
            if img_size < min_size:
                validated_config['img_size'] = min_size
                print(f"Adjusted img_size to {min_size} for Swin-UNETR with {levels} levels")
        
        return validated_config 