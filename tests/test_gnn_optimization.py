"""
Tests for GNN Optimization Module - Session 22
===============================================
"""

import pytest
import torch
import torch.nn as nn
from src.compute.gnn_optimization import (
    GraphBatch,
    BenchmarkResult,
    MessagePassing,
    GCNConv,
    GATConv,
    GraphSAGEConv,
    OptimizedGCN,
    add_self_loops,
    gcn_norm,
    softmax,
    create_karate_club_graph
)


# Test fixtures

@pytest.fixture
def small_graph():
    """Small test graph."""
    # Triangle graph: 0-1, 1-2, 2-0
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    x = torch.randn(3, 8)
    return GraphBatch(x=x, edge_index=edge_index, num_graphs=1)


@pytest.fixture
def karate_graph():
    """Karate Club graph."""
    return create_karate_club_graph()


# Test GraphBatch

def test_graph_batch_creation():
    """Test GraphBatch creation."""
    x = torch.randn(10, 5)
    edge_index = torch.randint(0, 10, (2, 20))
    
    batch = GraphBatch(x=x, edge_index=edge_index)
    
    assert batch.x.shape == (10, 5)
    assert batch.edge_index.shape == (2, 20)
    assert batch.num_graphs == 1


def test_graph_batch_with_edge_attr():
    """Test GraphBatch with edge attributes."""
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 2)
    
    batch = GraphBatch(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    assert batch.edge_attr.shape == (3, 2)


# Test MessagePassing base class

def test_message_passing_init():
    """Test MessagePassing initialization."""
    mp = MessagePassing(aggr='add')
    assert mp.aggr == 'add'


def test_message_passing_aggregation_add(small_graph):
    """Test add aggregation."""
    mp = MessagePassing(aggr='add')
    
    messages = torch.ones(small_graph.edge_index.size(1), 4)
    index = small_graph.edge_index[1]  # Target nodes
    
    out = mp.aggregate(messages, index, num_nodes=3)
    
    assert out.shape == (3, 4)


def test_message_passing_aggregation_mean(small_graph):
    """Test mean aggregation."""
    mp = MessagePassing(aggr='mean')
    
    messages = torch.ones(small_graph.edge_index.size(1), 4)
    index = small_graph.edge_index[1]
    
    out = mp.aggregate(messages, index, num_nodes=3)
    
    assert out.shape == (3, 4)
    # Mean of ones should be ones
    assert torch.allclose(out, torch.ones(3, 4), atol=1e-6)


def test_message_passing_aggregation_max(small_graph):
    """Test max aggregation."""
    mp = MessagePassing(aggr='max')
    
    messages = torch.randn(small_graph.edge_index.size(1), 4)
    index = small_graph.edge_index[1]
    
    out = mp.aggregate(messages, index, num_nodes=3)
    
    assert out.shape == (3, 4)


# Test GCNConv

def test_gcnconv_creation():
    """Test GCNConv layer creation."""
    conv = GCNConv(8, 16)
    
    assert conv.in_channels == 8
    assert conv.out_channels == 16
    assert conv.weight.shape == (8, 16)


def test_gcnconv_forward(small_graph):
    """Test GCNConv forward pass."""
    conv = GCNConv(8, 16)
    
    out = conv(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 16)


def test_gcnconv_no_bias():
    """Test GCNConv without bias."""
    conv = GCNConv(8, 16, bias=False)
    
    assert conv.bias is None


def test_gcnconv_no_normalize(small_graph):
    """Test GCNConv without normalization."""
    conv = GCNConv(8, 16, normalize=False)
    
    out = conv(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 16)


# Test GATConv

def test_gatconv_creation():
    """Test GATConv layer creation."""
    conv = GATConv(8, 16, heads=4)
    
    assert conv.in_channels == 8
    assert conv.out_channels == 16
    assert conv.heads == 4


def test_gatconv_forward(small_graph):
    """Test GATConv forward pass."""
    conv = GATConv(8, 16, heads=4, concat=True)
    
    out = conv(small_graph.x, small_graph.edge_index)
    
    # With concat: out_channels * heads
    assert out.shape == (3, 16 * 4)


def test_gatconv_no_concat(small_graph):
    """Test GATConv with averaging instead of concatenation."""
    conv = GATConv(8, 16, heads=4, concat=False)
    
    out = conv(small_graph.x, small_graph.edge_index)
    
    # Without concat: average over heads
    assert out.shape == (3, 16)


def test_gatconv_dropout(small_graph):
    """Test GATConv with dropout."""
    conv = GATConv(8, 16, heads=2, dropout=0.5)
    conv.train()
    
    out = conv(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 16 * 2)


# Test GraphSAGEConv

def test_graphsage_creation():
    """Test GraphSAGE layer creation."""
    conv = GraphSAGEConv(8, 16)
    
    assert conv.in_channels == 8
    assert conv.out_channels == 16


def test_graphsage_forward(small_graph):
    """Test GraphSAGE forward pass."""
    conv = GraphSAGEConv(8, 16)
    
    out = conv(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 16)


def test_graphsage_aggregation_max(small_graph):
    """Test GraphSAGE with max aggregation."""
    conv = GraphSAGEConv(8, 16, aggr='max')
    
    out = conv(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 16)


def test_graphsage_no_normalize(small_graph):
    """Test GraphSAGE without L2 normalization."""
    conv = GraphSAGEConv(8, 16, normalize=False)
    
    out = conv(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 16)


# Test OptimizedGCN

def test_optimized_gcn_creation():
    """Test OptimizedGCN creation."""
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=3,
        out_channels=4
    )
    
    assert len(model.convs) == 3
    assert model.num_layers == 3


def test_optimized_gcn_forward(small_graph):
    """Test OptimizedGCN forward pass."""
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=2,
        out_channels=4
    )
    
    out = model(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 4)


def test_optimized_gcn_single_layer(small_graph):
    """Test OptimizedGCN with single layer."""
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1
    )
    
    out = model(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 16)


def test_optimized_gcn_different_activations(small_graph):
    """Test OptimizedGCN with different activation functions."""
    for activation in ['relu', 'elu', 'leaky_relu']:
        model = OptimizedGCN(
            in_channels=8,
            hidden_channels=16,
            num_layers=2,
            activation=activation
        )
        
        out = model(small_graph.x, small_graph.edge_index)
        assert out.shape == (3, 16)


def test_optimized_gcn_dropout(small_graph):
    """Test OptimizedGCN with dropout."""
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=2,
        dropout=0.5
    )
    model.train()
    
    out = model(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 16)


def test_optimized_gcn_optimization_levels(small_graph):
    """Test different optimization levels."""
    for level in [0, 1, 2]:
        model = OptimizedGCN(
            in_channels=8,
            hidden_channels=16,
            num_layers=2,
            optimization_level=level
        )
        
        out = model(small_graph.x, small_graph.edge_index)
        assert out.shape == (3, 16)


# Test benchmark functionality

def test_benchmark(small_graph):
    """Test benchmark functionality."""
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=2
    )
    
    result = model.benchmark(small_graph, num_iterations=10)
    
    assert isinstance(result, BenchmarkResult)
    assert result.throughput > 0
    assert result.latency > 0
    assert result.num_parameters > 0


def test_benchmark_karate(karate_graph):
    """Test benchmark on Karate Club graph."""
    model = OptimizedGCN(
        in_channels=34,
        hidden_channels=16,
        num_layers=2
    )
    
    result = model.benchmark(karate_graph, num_iterations=10)
    
    assert result.throughput > 0
    assert result.latency > 0


# Test helper functions

def test_add_self_loops():
    """Test adding self-loops to edge_index."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    
    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=3)
    
    # Should have 3 original + 3 self-loops
    assert edge_index_with_loops.size(1) == 6
    
    # Check self-loops
    assert (0, 0) in zip(edge_index_with_loops[0].tolist(), edge_index_with_loops[1].tolist())
    assert (1, 1) in zip(edge_index_with_loops[0].tolist(), edge_index_with_loops[1].tolist())
    assert (2, 2) in zip(edge_index_with_loops[0].tolist(), edge_index_with_loops[1].tolist())


def test_add_self_loops_with_weights():
    """Test adding self-loops with edge weights."""
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_weight = torch.tensor([0.5, 0.5])
    
    edge_index_with_loops, edge_weight_with_loops = add_self_loops(edge_index, edge_weight, num_nodes=2)
    
    assert edge_index_with_loops.size(1) == 4  # 2 original + 2 self-loops
    assert edge_weight_with_loops.size(0) == 4


def test_gcn_norm():
    """Test GCN normalization."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    
    edge_weight = gcn_norm(edge_index, num_nodes=3)
    
    assert edge_weight.size(0) == 3
    assert torch.all(edge_weight > 0)


def test_gcn_norm_with_weights():
    """Test GCN normalization with existing weights."""
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_weight = torch.tensor([2.0, 2.0])
    
    normalized = gcn_norm(edge_index, edge_weight, num_nodes=2)
    
    assert normalized.size(0) == 2


def test_softmax_grouped():
    """Test grouped softmax."""
    src = torch.tensor([[1.0], [2.0], [3.0], [1.0]])
    index = torch.tensor([0, 0, 0, 1])  # First 3 in group 0, last in group 1
    
    out = softmax(src, index, num_nodes=2)
    
    assert out.shape == src.shape
    # Check softmax property: sum to 1 per group
    group0_sum = out[:3].sum().item()
    assert abs(group0_sum - 1.0) < 1e-6


def test_karate_club_graph():
    """Test Karate Club graph creation."""
    graph = create_karate_club_graph()
    
    assert graph.x.shape == (34, 34)  # 34 nodes, identity features
    assert graph.edge_index.size(0) == 2
    assert graph.num_graphs == 1


# Test edge cases

def test_gcnconv_empty_graph():
    """Test GCNConv with minimal graph."""
    conv = GCNConv(4, 8)
    
    x = torch.randn(1, 4)
    edge_index = torch.tensor([[0], [0]])  # Single self-loop
    
    out = conv(x, edge_index)
    
    assert out.shape == (1, 8)


def test_gatconv_single_node():
    """Test GATConv with single node."""
    conv = GATConv(4, 8, heads=2)
    
    x = torch.randn(1, 4)
    edge_index = torch.tensor([[0], [0]])
    
    out = conv(x, edge_index)
    
    assert out.shape == (1, 16)  # concat=True by default


def test_graphsage_isolated_node():
    """Test GraphSAGE with isolated node."""
    conv = GraphSAGEConv(4, 8)
    
    # 2 nodes: 0 connected to itself, 1 isolated
    x = torch.randn(2, 4)
    edge_index = torch.tensor([[0], [0]])
    
    out = conv(x, edge_index)
    
    assert out.shape == (2, 8)


# Integration tests

def test_full_gnn_pipeline(small_graph):
    """Test complete GNN pipeline."""
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=3,
        out_channels=4,
        dropout=0.5
    )
    
    # Forward pass
    model.train()
    out_train = model(small_graph.x, small_graph.edge_index)
    
    model.eval()
    out_eval = model(small_graph.x, small_graph.edge_index)
    
    assert out_train.shape == (3, 4)
    assert out_eval.shape == (3, 4)


def test_gradient_flow(small_graph):
    """Test gradient flow through GNN."""
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=2,
        out_channels=4
    )
    
    out = model(small_graph.x, small_graph.edge_index)
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None


def test_multiple_layers(small_graph):
    """Test GNN with many layers."""
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=5,
        out_channels=4
    )
    
    out = model(small_graph.x, small_graph.edge_index)
    
    assert out.shape == (3, 4)


# Performance tests

def test_large_graph_handling():
    """Test handling of larger graphs."""
    # Create larger graph
    num_nodes = 100
    num_edges = 500
    
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    graph = GraphBatch(x=x, edge_index=edge_index, num_graphs=1)
    
    model = OptimizedGCN(
        in_channels=16,
        hidden_channels=32,
        num_layers=2
    )
    
    out = model(x, edge_index)
    
    assert out.shape == (num_nodes, 32)


def test_batched_graphs():
    """Test with multiple graphs in batch."""
    # 2 graphs with 3 nodes each
    x = torch.randn(6, 8)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],  # Sources
        [1, 2, 0, 4, 5, 3]   # Targets
    ])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])  # Graph assignment
    
    graph = GraphBatch(x=x, edge_index=edge_index, batch=batch, num_graphs=2)
    
    model = OptimizedGCN(
        in_channels=8,
        hidden_channels=16,
        num_layers=2
    )
    
    out = model(x, edge_index)
    
    assert out.shape == (6, 16)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
