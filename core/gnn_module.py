import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.utils import grid

class SegmentationGNN(nn.Module):
    """GNN module for refining segmentation features"""
    def __init__(self, in_channels, hidden_channels=64, num_layers=3):
        super(SegmentationGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Initial feature transformation
        self.feature_transform = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                gnn.GCNConv(hidden_channels, hidden_channels)
            )
        
        # Final projection
        self.final_proj = nn.Conv2d(hidden_channels, in_channels, 1)
        
    def _create_graph_from_features(self, x, batch_size, height, width):
        """Create a graph from feature map"""
        # Create grid-like edge connections (8-neighborhood)
        edge_index = []
        for i in range(height):
            for j in range(width):
                node = i * width + j
                # Add edges to 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor = ni * width + nj
                            edge_index.append([node, neighbor])
                            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = edge_index.to(x.device)
        
        # Repeat edge_index for each batch
        if batch_size > 1:
            edge_index_list = []
            for b in range(batch_size):
                offset = b * (height * width)
                batch_edge_index = edge_index + offset
                edge_index_list.append(batch_edge_index)
            edge_index = torch.cat(edge_index_list, dim=1)
        
        return edge_index
        
    def _features_to_graph(self, features):
        """Convert feature map to graph representation"""
        batch_size, channels, height, width = features.shape
        
        # Reshape features to node features
        x = features.view(batch_size, channels, -1)  # [B, C, H*W]
        x = x.permute(0, 2, 1)  # [B, H*W, C]
        x = x.reshape(-1, channels)  # [B*H*W, C]
        
        # Create graph structure
        edge_index = self._create_graph_from_features(x, batch_size, height, width)
        
        return Data(x=x, edge_index=edge_index), (batch_size, channels, height, width)
        
    def _graph_to_features(self, x, original_shape):
        """Convert graph representation back to feature map"""
        batch_size, channels, height, width = original_shape
        
        # Reshape node features back to feature map
        x = x.view(batch_size, height * width, -1)  # [B, H*W, C]
        x = x.permute(0, 2, 1)  # [B, C, H*W]
        x = x.view(batch_size, -1, height, width)  # [B, C, H, W]
        
        return x
        
    def forward(self, features):
        """Forward pass"""
        # Transform features
        x = self.feature_transform(features)
        
        # Convert to graph
        graph, original_shape = self._features_to_graph(x)
        
        # Apply GNN layers
        x = graph.x
        edge_index = graph.edge_index
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = torch.relu(x)
        
        # Convert back to feature map
        x = self._graph_to_features(x, original_shape)
        
        # Final projection
        x = self.final_proj(x)
        
        # Residual connection
        x = x + features
        
        return x 