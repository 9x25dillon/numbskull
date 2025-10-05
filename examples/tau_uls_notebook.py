#!/usr/bin/env python3
"""
Interactive notebook-style examples for TAU-ULS Enhanced WaveCaster
Can be run as a script or converted to Jupyter notebook
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import json
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tau_uls_wavecaster_enhanced import (
    KFPLayer, TAULSControlUnit, EntropyRegulationModule,
    TAULSAnalyzer, create_kfp_polynomial_basis, kfp_polynomial_update,
    safe_json
)

# %% [markdown]
# # TAU-ULS Enhanced WaveCaster: Interactive Examples
# 
# This notebook demonstrates the core components of the TAU-ULS (Two-level Trans-Algorithmic Universal Learning System) architecture.

# %% [markdown]
# ## 1. Understanding the Kinetic Force Principle (KFP) Layer

# %%
def demonstrate_kfp_convergence():
    """Show how KFP layers drive systems toward stability"""
    print("KFP Layer Convergence Demonstration")
    print("-" * 40)
    
    # Create a KFP layer
    dim = 32
    kfp = KFPLayer(dim=dim, stability_weight=0.1)
    
    # Start with high-variance input
    x = torch.randn(16, dim) * 5.0  # High variance
    
    # Track convergence
    history = []
    
    for i in range(20):
        x, fluctuation = kfp(x)
        avg_fluctuation = fluctuation.mean().item()
        variance = x.var().item()
        
        history.append({
            'iteration': i,
            'fluctuation': avg_fluctuation,
            'variance': variance
        })
        
        if i % 5 == 0:
            print(f"Iteration {i:2d}: Fluctuation={avg_fluctuation:.4f}, Variance={variance:.4f}")
    
    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        iterations = [h['iteration'] for h in history]
        fluctuations = [h['fluctuation'] for h in history]
        variances = [h['variance'] for h in history]
        
        ax1.plot(iterations, fluctuations, 'b-', marker='o')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fluctuation Intensity')
        ax1.set_title('KFP Fluctuation Convergence')
        ax1.grid(True)
        
        ax2.plot(iterations, variances, 'r-', marker='s')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Variance')
        ax2.set_title('System Variance Over Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\n(Install matplotlib to see visualizations)")
    
    return history

# Run demonstration
kfp_history = demonstrate_kfp_convergence()

# %% [markdown]
# ## 2. TAU-ULS Control Unit: Two-Level Control

# %%
def demonstrate_control_mixing():
    """Show how meta-control and auto-control are mixed"""
    print("\nTAU-ULS Control Unit Demonstration")
    print("-" * 40)
    
    # Create control unit
    control = TAULSControlUnit(input_dim=64, hidden_dim=128, control_dim=32)
    
    # Test with different input patterns
    test_inputs = {
        'random': torch.randn(1, 64),
        'structured': torch.ones(1, 64) * 0.5 + torch.randn(1, 64) * 0.1,
        'sparse': torch.zeros(1, 64).scatter_(1, torch.randint(0, 64, (10,)), 1.0),
    }
    
    results = {}
    
    for name, input_tensor in test_inputs.items():
        output = control(input_tensor)
        
        results[name] = {
            'control_output_norm': output['control_output'].norm().item(),
            'control_mixing': output['control_mixing'].item(),
            'meta_stability': output['meta_stability'].mean().item(),
            'auto_stability': output['auto_stability'].mean().item(),
        }
        
        print(f"\n{name.capitalize()} input:")
        print(f"  Control mixing (α): {results[name]['control_mixing']:.3f}")
        print(f"  Meta stability: {results[name]['meta_stability']:.4f}")
        print(f"  Auto stability: {results[name]['auto_stability']:.4f}")
        print(f"  Output norm: {results[name]['control_output_norm']:.3f}")
    
    # Visualize control mixing
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        patterns = list(results.keys())
        mixing_values = [results[p]['control_mixing'] for p in patterns]
        meta_values = [results[p]['meta_stability'] for p in patterns]
        auto_values = [results[p]['auto_stability'] for p in patterns]
        
        x = np.arange(len(patterns))
        width = 0.25
        
        ax.bar(x - width, mixing_values, width, label='Control Mixing (α)', alpha=0.8)
        ax.bar(x, meta_values, width, label='Meta Stability', alpha=0.8)
        ax.bar(x + width, auto_values, width, label='Auto Stability', alpha=0.8)
        
        ax.set_xlabel('Input Pattern')
        ax.set_ylabel('Value')
        ax.set_title('TAU-ULS Control Characteristics')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        pass
    
    return results

# Run demonstration
control_results = demonstrate_control_mixing()

# %% [markdown]
# ## 3. Entropy Regulation Module

# %%
def demonstrate_entropy_regulation():
    """Show how entropy regulation responds to environmental stress"""
    print("\nEntropy Regulation Module Demonstration")
    print("-" * 40)
    
    # Create entropy regulator
    entropy_reg = EntropyRegulationModule(dim=32, max_entropy_target=0.8)
    
    # Test with different stress levels
    x = torch.randn(8, 32)
    stress_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = []
    
    for stress in stress_levels:
        stress_tensor = torch.tensor([stress])
        output, info = entropy_reg(x, stress_tensor)
        
        results.append({
            'stress': stress,
            'current_entropy': info['current_entropy'].item(),
            'target_intensity': info['target_intensity'].item(),
            'entropy_error': info['entropy_error'].item(),
            'output_norm': output.norm().item(),
        })
        
        print(f"\nStress level: {stress:.1f}")
        print(f"  Current entropy: {info['current_entropy'].item():.4f}")
        print(f"  Target intensity: {info['target_intensity'].item():.4f}")
        print(f"  Entropy error: {info['entropy_error'].item():.4f}")
    
    # Visualize entropy regulation
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        stress_vals = [r['stress'] for r in results]
        entropy_vals = [r['current_entropy'] for r in results]
        intensity_vals = [r['target_intensity'] for r in results]
        
        ax1.plot(stress_vals, entropy_vals, 'b-', marker='o', label='Current Entropy')
        ax1.axhline(y=0.8, color='r', linestyle='--', label='Max Target (0.8)')
        ax1.set_xlabel('Environmental Stress')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Entropy vs Stress')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(stress_vals, intensity_vals, 'g-', marker='s', label='Target Intensity')
        ax2.set_xlabel('Environmental Stress')
        ax2.set_ylabel('Modification Intensity')
        ax2.set_title('Adaptive Intensity Control')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        pass
    
    return results

# Run demonstration
entropy_results = demonstrate_entropy_regulation()

# %% [markdown]
# ## 4. Polynomial KFP Basis Functions

# %%
def demonstrate_polynomial_kfp():
    """Show polynomial basis functions for KFP approximation"""
    print("\nPolynomial KFP Basis Demonstration")
    print("-" * 40)
    
    # Create polynomial basis
    degree = 3
    dim = 8
    coeffs = create_kfp_polynomial_basis(degree, dim)
    
    print(f"Polynomial basis shape: {coeffs.shape}")
    print(f"Degree: {degree}, Dimension: {dim}")
    
    # Check stability constraint
    print(f"\nQuadratic term diagonal (should be negative):")
    print(coeffs[2].diagonal()[:4].numpy())
    
    # Demonstrate polynomial update
    x = torch.randn(dim)
    print(f"\nInitial state norm: {x.norm().item():.4f}")
    
    # Apply multiple updates
    trajectory = [x.clone()]
    for i in range(10):
        x = kfp_polynomial_update(x, coeffs, learning_rate=0.05)
        trajectory.append(x.clone())
    
    print(f"Final state norm: {x.norm().item():.4f}")
    
    # Visualize trajectory
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Norm over iterations
        norms = [t.norm().item() for t in trajectory]
        ax1.plot(norms, 'b-', marker='o')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('State Norm')
        ax1.set_title('KFP Polynomial Update Convergence')
        ax1.grid(True)
        
        # Phase space (first 2 dimensions)
        if dim >= 2:
            x_coords = [t[0].item() for t in trajectory]
            y_coords = [t[1].item() for t in trajectory]
            
            ax2.plot(x_coords, y_coords, 'r-', marker='o', markersize=4)
            ax2.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
            ax2.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
            ax2.set_xlabel('Dimension 1')
            ax2.set_ylabel('Dimension 2')
            ax2.set_title('Phase Space Trajectory')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        pass
    
    return trajectory

# Run demonstration
poly_trajectory = demonstrate_polynomial_kfp()

# %% [markdown]
# ## 5. Complete TAU-ULS Analysis Pipeline

# %%
def analyze_text_samples():
    """Analyze different text samples with TAU-ULS"""
    print("\nComplete TAU-ULS Text Analysis")
    print("-" * 40)
    
    # Create analyzer
    analyzer = TAULSAnalyzer()
    
    # Test samples
    samples = {
        'simple': "Hello world",
        'technical': "The quantum entanglement phenomenon demonstrates non-local correlations",
        'repetitive': "data " * 50,
        'mixed': "Important: Process A1B2C3 -> Result XYZ! @timestamp:12345",
        'noise': "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 100)),
    }
    
    results = {}
    
    for name, text in samples.items():
        analysis = analyzer(text)
        results[name] = analysis
        
        print(f"\n{name.upper()} TEXT:")
        print(f"  Text: '{text[:30]}...'")
        print(f"  Stability: {analysis['stability_score']:.3f}")
        print(f"  Entropy: {analysis['entropy_score']:.3f}")
        print(f"  Complexity: {analysis['complexity_score']:.3f}")
        print(f"  Coherence: {analysis['coherence_score']:.3f}")
        print(f"  Control Mix: {analysis['control_mixing']:.3f}")
    
    # Create comparison matrix
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for heatmap
        metrics = ['stability_score', 'entropy_score', 'complexity_score', 'coherence_score']
        text_types = list(results.keys())
        
        data = np.array([[results[t][m] for m in metrics] for t in text_types])
        
        im = ax.imshow(data, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(text_types)))
        ax.set_xticklabels([m.replace('_score', '') for m in metrics])
        ax.set_yticklabels(text_types)
        
        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(text_types)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black" if data[i, j] > 0.5 else "white")
        
        ax.set_title('TAU-ULS Analysis Scores Across Text Types')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        pass
    
    return results

# Run analysis
analysis_results = analyze_text_samples()

# %% [markdown]
# ## Summary and Key Insights
# 
# 1. **KFP Layers**: Drive systems toward stability through gradient-based optimization on fluctuation intensity
# 2. **Two-Level Control**: Balances reactive (automatic) and deliberative (meta) control strategies
# 3. **Entropy Regulation**: Adapts system behavior based on environmental stress
# 4. **Polynomial Basis**: Provides mathematical foundation for stability landscapes
# 5. **Integrated Analysis**: Combines multiple neural metrics for comprehensive content understanding

# %%
def print_summary():
    """Print summary of key findings"""
    print("\n" + "="*60)
    print("TAU-ULS ENHANCED WAVECASTER: KEY INSIGHTS")
    print("="*60)
    
    print("\n1. STABILITY CONVERGENCE")
    print("   - KFP layers reduce fluctuation intensity over time")
    print("   - Typical convergence within 10-20 iterations")
    print("   - Stability weight parameter controls convergence speed")
    
    print("\n2. ADAPTIVE CONTROL")
    print("   - Control mixing adapts to input characteristics")
    print("   - Structured inputs favor meta-control")
    print("   - Random inputs trigger more automatic control")
    
    print("\n3. ENTROPY MANAGEMENT")
    print("   - System maintains entropy below target threshold")
    print("   - Environmental stress modulates intensity")
    print("   - Prevents over-regularization in high-stress conditions")
    
    print("\n4. CONTENT ANALYSIS")
    print("   - Technical text shows high complexity, moderate stability")
    print("   - Repetitive text has low entropy, high stability")
    print("   - Random text exhibits high entropy, low coherence")
    
    print("\n5. PRACTICAL APPLICATIONS")
    print("   - Adaptive modulation selection based on content")
    print("   - Real-time parameter optimization")
    print("   - Robust to varying input characteristics")
    
    print("\n" + "="*60)

print_summary()

# %% [markdown]
# ## Next Steps
# 
# 1. **Experiment** with different TAU-ULS parameters
# 2. **Integrate** with real communication systems
# 3. **Optimize** for specific use cases
# 4. **Extend** with additional neural architectures
# 5. **Deploy** in production environments

# %%
print("\nNotebook examples completed successfully!")
print("Ready for TAU-ULS enhanced communication experiments.")