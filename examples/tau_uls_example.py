#!/usr/bin/env python3
"""
Example usage of TAU-ULS Enhanced WaveCaster
Demonstrates various features and use cases
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path if running from examples folder
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tau_uls_wavecaster_enhanced import (
        TAULSAnalyzer, TAUEnhancedMirrorCast, TAUAdaptiveLinkPlanner,
        ModConfig, FrameConfig, SecurityConfig, ModulationScheme, FEC,
        full_tau_cast_and_save, safe_json
    )
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have installed: pip install numpy scipy torch matplotlib")
    sys.exit(1)

def example_1_basic_tau_analysis():
    """Example 1: Basic TAU-ULS neural analysis of text"""
    print("\n=== Example 1: Basic TAU-ULS Analysis ===")
    
    # Create analyzer
    analyzer = TAULSAnalyzer()
    
    # Test different text samples
    samples = [
        "Simple clear text with low complexity",
        "The quantum superposition principle states that any two (or more) quantum states can be added together ('superposed') and the result will be another valid quantum state",
        "A" * 100 + "B" * 100,  # High repetition
        "!@#$%^&*()_+" * 20,     # High entropy
    ]
    
    for i, text in enumerate(samples):
        print(f"\nSample {i+1}: '{text[:50]}...'")
        analysis = analyzer(text)
        
        print(f"  Stability: {analysis['stability_score']:.3f}")
        print(f"  Entropy: {analysis['entropy_score']:.3f}")
        print(f"  Complexity: {analysis['complexity_score']:.3f}")
        print(f"  Coherence: {analysis['coherence_score']:.3f}")
        print(f"  Control Mixing: {analysis['control_mixing']:.3f}")
        print(f"  Fluctuation: {analysis['fluctuation_intensity']:.3f}")

def example_2_adaptive_modulation():
    """Example 2: Adaptive modulation selection based on content"""
    print("\n=== Example 2: Adaptive Modulation Selection ===")
    
    # Create planner
    planner = TAUAdaptiveLinkPlanner()
    base_config = ModConfig()
    
    # Test different content types
    test_cases = [
        ("Simple message", "Hello world"),
        ("Technical document", "The implementation uses a novel approach to quantum error correction based on topological codes"),
        ("High entropy data", "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 100))),
        ("Structured data", '{"sensor_id": 42, "temperature": 23.5, "humidity": 65, "timestamp": 1234567890}'),
    ]
    
    for name, content in test_cases:
        print(f"\n{name}:")
        new_config, plan_info = planner.plan(content, base_config)
        
        print(f"  Recommended modulation: {plan_info['recommended_modulation'].upper()}")
        print(f"  Symbol rate: {base_config.symbol_rate} -> {new_config.symbol_rate}")
        print(f"  Amplitude: {base_config.amplitude:.2f} -> {new_config.amplitude:.2f}")
        print(f"  Stability score: {plan_info['tau_analysis']['stability_score']:.3f}")

def example_3_mirror_cast_comparison():
    """Example 3: Compare classical and TAU-enhanced analysis"""
    print("\n=== Example 3: TAU-Enhanced Mirror Cast ===")
    
    # Create enhanced mirror cast engine
    engine = TAUEnhancedMirrorCast()
    
    text = """
    In the realm of digital communication, the convergence of neural architectures 
    and symbolic reasoning opens new possibilities for adaptive systems that can 
    respond intelligently to varying channel conditions and content characteristics.
    """
    
    # Run full analysis
    result = engine.cast(text)
    
    print(f"\nText: '{text.strip()[:60]}...'")
    print(f"\nClassical Analysis:")
    print(f"  Entropy: {result['entropy']:.3f}")
    print(f"  Matrix Rank: {result['matrix']['projected_rank']}")
    print(f"  Structure: {result['matrix']['structure']}")
    
    print(f"\nTAU-ULS Neural Analysis:")
    tau = result['tau_uls']
    print(f"  Stability: {tau['stability_score']:.3f}")
    print(f"  Entropy: {tau['entropy_score']:.3f}")
    print(f"  Complexity: {tau['complexity_score']:.3f}")
    print(f"  Coherence: {tau['coherence_score']:.3f}")
    
    print(f"\nCombined Stability: {result['combined_stability']:.3f}")
    print(f"Recommended Modulation: {result['recommendation'].upper()}")

def example_4_full_transmission():
    """Example 4: Complete transmission pipeline with TAU-ULS"""
    print("\n=== Example 4: Full Transmission Pipeline ===")
    
    # Prepare content
    message = """
    TAU-ULS Enhanced Communication: This message demonstrates the full pipeline
    of neural analysis, adaptive modulation selection, and signal generation.
    The system analyzes content complexity and automatically selects optimal
    transmission parameters.
    """
    
    # Run TAU analysis
    analyzer = TAULSAnalyzer()
    tau_analysis = analyzer(message)
    
    # Get adaptive configuration
    planner = TAUAdaptiveLinkPlanner()
    base_config = ModConfig()
    mod_config, plan_info = planner.plan(message, base_config)
    
    # Determine modulation scheme
    scheme_name = plan_info['recommended_modulation']
    scheme = ModulationScheme[scheme_name.upper()]
    
    # Security and framing
    frame_config = FrameConfig()
    security_config = SecurityConfig(
        watermark="TAU-DEMO-2024"
    )
    
    # Generate transmission
    output_dir = Path("tau_demo_output")
    paths = full_tau_cast_and_save(
        text=message,
        outdir=output_dir,
        scheme=scheme,
        mcfg=mod_config,
        fcfg=frame_config,
        sec=security_config,
        fec_scheme=FEC.HAMMING74,
        want_wav=True,
        want_iq=True,
        tau_analysis=tau_analysis,
        title="TAU-ULS Demo"
    )
    
    print(f"\nTransmission Configuration:")
    print(f"  Modulation: {scheme.name}")
    print(f"  Symbol Rate: {mod_config.symbol_rate} Hz")
    print(f"  Sample Rate: {mod_config.sample_rate} Hz")
    print(f"  FEC: Hamming(7,4)")
    
    print(f"\nTAU-ULS Scores:")
    print(f"  Stability: {tau_analysis['stability_score']:.3f}")
    print(f"  Entropy: {tau_analysis['entropy_score']:.3f}")
    print(f"  Complexity: {tau_analysis['complexity_score']:.3f}")
    
    print(f"\nGenerated Files:")
    for key, path in paths.__dict__.items():
        if path:
            print(f"  {key}: {path}")

def example_5_visualize_tau_metrics():
    """Example 5: Visualize TAU-ULS metrics for different content types"""
    print("\n=== Example 5: TAU-ULS Metrics Visualization ===")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return
    
    # Create analyzer
    analyzer = TAULSAnalyzer()
    
    # Test various content types
    content_types = {
        "Simple": "Hello world",
        "Technical": "Quantum error correction using stabilizer codes",
        "Repetitive": "AAAA" * 25,
        "Random": "".join(np.random.choice(list("01"), 100)),
        "Structured": '{"data": [1, 2, 3, 4, 5]}',
        "Natural": "The quick brown fox jumps over the lazy dog",
    }
    
    # Collect metrics
    metrics = {
        'stability': [],
        'entropy': [],
        'complexity': [],
        'coherence': [],
    }
    labels = []
    
    for label, content in content_types.items():
        analysis = analyzer(content)
        labels.append(label)
        metrics['stability'].append(analysis['stability_score'])
        metrics['entropy'].append(analysis['entropy_score'])
        metrics['complexity'].append(analysis['complexity_score'])
        metrics['coherence'].append(analysis['coherence_score'])
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Stability scores
    ax1.bar(labels, metrics['stability'], color='blue', alpha=0.7)
    ax1.set_title('Stability Scores')
    ax1.set_ylim(0, 1)
    ax1.set_xticklabels(labels, rotation=45)
    
    # Entropy scores
    ax2.bar(labels, metrics['entropy'], color='red', alpha=0.7)
    ax2.set_title('Entropy Scores')
    ax2.set_ylim(0, 1)
    ax2.set_xticklabels(labels, rotation=45)
    
    # Complexity scores
    ax3.bar(labels, metrics['complexity'], color='green', alpha=0.7)
    ax3.set_title('Complexity Scores')
    ax3.set_ylim(0, 1)
    ax3.set_xticklabels(labels, rotation=45)
    
    # Coherence scores
    ax4.bar(labels, metrics['coherence'], color='purple', alpha=0.7)
    ax4.set_title('Coherence Scores')
    ax4.set_ylim(0, 1)
    ax4.set_xticklabels(labels, rotation=45)
    
    plt.suptitle('TAU-ULS Metrics Across Different Content Types')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("tau_demo_output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "tau_metrics_comparison.png")
    print(f"\nVisualization saved to: {output_dir / 'tau_metrics_comparison.png'}")
    
    # Also create a radar chart
    fig2, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Select one content type for radar
    content_key = "Technical"
    idx = labels.index(content_key)
    
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    values = [
        metrics['stability'][idx],
        metrics['entropy'][idx],
        metrics['complexity'][idx],
        metrics['coherence'][idx]
    ]
    values += values[:1]  # Complete the circle
    angles = np.concatenate([angles, [angles[0]]])
    
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Stability', 'Entropy', 'Complexity', 'Coherence'])
    ax.set_ylim(0, 1)
    ax.set_title(f'TAU-ULS Profile: {content_key}')
    ax.grid(True)
    
    plt.savefig(output_dir / "tau_radar_chart.png")
    print(f"Radar chart saved to: {output_dir / 'tau_radar_chart.png'}")

def main():
    """Run all examples"""
    print("TAU-ULS Enhanced WaveCaster Examples")
    print("=" * 50)
    
    examples = [
        example_1_basic_tau_analysis,
        example_2_adaptive_modulation,
        example_3_mirror_cast_comparison,
        example_4_full_transmission,
        example_5_visualize_tau_metrics,
    ]
    
    for i, example_func in enumerate(examples):
        try:
            example_func()
        except Exception as e:
            print(f"\nError in example {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    
    # Summary
    print("\nKey Takeaways:")
    print("1. TAU-ULS provides multi-dimensional content analysis")
    print("2. Adaptive modulation selection improves transmission reliability")
    print("3. Neural and symbolic approaches complement each other")
    print("4. The system adapts to content complexity automatically")
    print("5. Visualization helps understand system behavior")

if __name__ == "__main__":
    main()