"""
Comprehensive test suite for Protein Pre-Cancer Prediction project.
Tests core dependencies and basic functionality.
"""

import pytest
import torch
import torch.nn as nn


class TestDependencies:
    """Test that core dependencies are available."""
    
    def test_torch_import(self):
        """Verify PyTorch is installed."""
        assert torch is not None
        
    def test_torch_version(self):
        """Check PyTorch version is available."""
        version = torch.__version__
        assert version is not None
        assert len(version) > 0
        

class TestDevice:
    """Test device availability (CPU/GPU)."""
    
    def test_cpu_available(self):
        """Verify CPU is available."""
        assert torch.device("cpu") is not None
        
    def test_device_creation(self):
        """Test creating tensors on available device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([1, 2, 3], device=device)
        assert tensor is not None
        assert tensor.device.type in ["cpu", "cuda"]


class TestTensorOperations:
    """Test basic tensor operations."""
    
    def test_tensor_creation(self):
        """Test creating basic tensors."""
        tensor = torch.randn(3, 4)
        assert tensor.shape == torch.Size([3, 4])
        
    def test_tensor_operations(self):
        """Test basic tensor arithmetic."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        c = a + b
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(c, expected)


class TestNeuralNetwork:
    """Test basic neural network operations."""
    
    def test_simple_model_creation(self):
        """Test creating a simple neural network."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        assert model is not None
        
    def test_model_forward_pass(self):
        """Test forward pass through a simple model."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        assert output.shape == torch.Size([1, 2])
        
    def test_model_parameters(self):
        """Test that model has learnable parameters."""
        model = nn.Linear(10, 5)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestGradients:
    """Test gradient computation."""
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = x ** 2
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.tensor([4.0, 6.0]))


def test_pytorch_basic():
    """Basic sanity test for PyTorch."""
    x = torch.randn(5, 3)
    assert x.shape == (5, 3)
    assert x.dtype == torch.float32
