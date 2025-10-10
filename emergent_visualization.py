"""
Emergent Cognitive Network Visualization System

Advanced visualization tools for emergent cognitive network dynamics,
quantum optimization processes, and swarm intelligence patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from emergent_cognitive_network import *

class EmergentVisualization:
    """
    Advanced visualization system for emergent cognitive networks
    """
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
    def plot_quantum_optimization_dynamics(self, quantum_results: Dict, save_path: str = None):
        """
        Visualize quantum optimization dynamics
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Quantum Optimization Protocol Dynamics', fontsize=16, fontweight='bold')
        
        # Extract data
        history = quantum_results['optimization_history']
        iterations = [h['iteration'] for h in history]
        objectives = [h['objective'] for h in history]
        betas = [h['beta'] for h in history]
        
        # Plot 1: Objective function evolution
        axes[0, 0].plot(iterations, objectives, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Objective Function Evolution')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Beta schedule
        axes[0, 1].plot(iterations, betas, 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Quantum Annealing Schedule')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Beta (Temperature)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Final quantum state
        final_state = quantum_results['final_state']
        state_magnitude = np.abs(final_state)
        state_phase = np.angle(final_state)
        
        x = np.arange(len(final_state))
        axes[1, 0].bar(x, state_magnitude, alpha=0.7, color='cyan')
        axes[1, 0].set_title('Final Quantum State Magnitude')
        axes[1, 0].set_xlabel('State Index')
        axes[1, 0].set_ylabel('Magnitude')
        
        # Plot 4: Phase distribution
        axes[1, 1].scatter(state_magnitude, state_phase, c=state_phase, 
                          cmap='hsv', alpha=0.7, s=50)
        axes[1, 1].set_title('Quantum State Phase Distribution')
        axes[1, 1].set_xlabel('Magnitude')
        axes[1, 1].set_ylabel('Phase (radians)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_swarm_intelligence_patterns(self, swarm_results: Dict, save_path: str = None):
        """
        Visualize swarm intelligence patterns and coordination
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Swarm Cognitive Network Patterns', fontsize=16, fontweight='bold')
        
        # Extract data
        coordination_matrix = swarm_results['coordination_matrix']
        pattern = swarm_results['pattern']
        intelligence_metric = swarm_results['intelligence_metric']
        
        # Plot 1: Coordination matrix heatmap
        im1 = axes[0, 0].imshow(coordination_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Agent Coordination Matrix\nIntelligence: {intelligence_metric:.3f}')
        axes[0, 0].set_xlabel('Agent Index')
        axes[0, 0].set_ylabel('Agent Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Pattern formation
        im2 = axes[0, 1].imshow(pattern, cmap='plasma', aspect='auto')
        axes[0, 1].set_title('Pattern Formation')
        axes[0, 1].set_xlabel('Pattern Index')
        axes[0, 1].set_ylabel('Pattern Index')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot 3: Coordination strength distribution
        coordination_values = coordination_matrix.flatten()
        axes[1, 0].hist(coordination_values, bins=30, alpha=0.7, color='orange')
        axes[1, 0].set_title('Coordination Strength Distribution')
        axes[1, 0].set_xlabel('Coordination Strength')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Pattern complexity
        pattern_complexity = np.std(pattern)
        axes[1, 1].bar(['Pattern Complexity'], [pattern_complexity], 
                      color='purple', alpha=0.7)
        axes[1, 1].set_title('Pattern Formation Complexity')
        axes[1, 1].set_ylabel('Complexity (Std Dev)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_neuromorphic_dynamics(self, neuromorphic_results: Dict, save_path: str = None):
        """
        Visualize neuromorphic dynamics and spiking patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Neuromorphic Processor Dynamics', fontsize=16, fontweight='bold')
        
        # Extract data
        spike_history = neuromorphic_results['spike_history']
        voltage_history = neuromorphic_results['voltage_history']
        final_weights = neuromorphic_results['final_weights']
        spike_rate = neuromorphic_results['spike_rate']
        
        # Plot 1: Spike raster plot
        time_steps, n_neurons = spike_history.shape
        spike_times = []
        neuron_ids = []
        
        for t in range(time_steps):
            for n in range(n_neurons):
                if spike_history[t, n]:
                    spike_times.append(t)
                    neuron_ids.append(n)
        
        axes[0, 0].scatter(spike_times, neuron_ids, s=1, alpha=0.6, color='red')
        axes[0, 0].set_title(f'Spike Raster Plot\nSpike Rate: {spike_rate:.3f}')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Neuron ID')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Voltage traces
        for i in range(min(5, n_neurons)):  # Show first 5 neurons
            axes[0, 1].plot(voltage_history[:, i], alpha=0.7, linewidth=1)
        axes[0, 1].set_title('Voltage Traces (First 5 Neurons)')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Membrane Potential (mV)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Synaptic weight matrix
        im3 = axes[1, 0].imshow(final_weights, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('Final Synaptic Weight Matrix')
        axes[1, 0].set_xlabel('Post-synaptic Neuron')
        axes[1, 0].set_ylabel('Pre-synaptic Neuron')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Plot 4: Weight distribution
        weight_values = final_weights.flatten()
        axes[1, 1].hist(weight_values, bins=30, alpha=0.7, color='green')
        axes[1, 1].set_title('Synaptic Weight Distribution')
        axes[1, 1].set_xlabel('Weight Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_holographic_encoding(self, holographic_results: Dict, save_path: str = None):
        """
        Visualize holographic encoding and memory patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Holographic Data Engine Patterns', fontsize=16, fontweight='bold')
        
        # Extract data
        encoded_data = holographic_results['encoded_data']
        recalled_data = holographic_results['recalled_data']
        matching_patterns = holographic_results['matching_patterns']
        recall_accuracy = holographic_results['recall_accuracy']
        
        # Plot 1: Holographic encoding (magnitude)
        encoded_magnitude = np.abs(encoded_data)
        axes[0, 0].plot(encoded_magnitude, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title(f'Holographic Encoding Magnitude\nRecall Accuracy: {recall_accuracy:.3f}')
        axes[0, 0].set_xlabel('Data Index')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Phase encoding
        encoded_phase = np.angle(encoded_data)
        axes[0, 1].plot(encoded_phase, 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Holographic Phase Encoding')
        axes[0, 1].set_xlabel('Data Index')
        axes[0, 1].set_ylabel('Phase (radians)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Recall comparison
        x = np.arange(len(recalled_data))
        axes[1, 0].plot(x, np.real(recalled_data), 'g-', linewidth=2, alpha=0.8, label='Recalled')
        axes[1, 0].plot(x, np.real(encoded_data), 'b--', linewidth=1, alpha=0.6, label='Original')
        axes[1, 0].set_title('Memory Recall Comparison')
        axes[1, 0].set_xlabel('Data Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Matching patterns
        pattern_counts = len(matching_patterns)
        axes[1, 1].bar(['Matching Patterns'], [pattern_counts], 
                      color='purple', alpha=0.7)
        axes[1, 1].set_title('Associative Memory Matches')
        axes[1, 1].set_ylabel('Number of Matches')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_morphogenetic_patterns(self, morphogenetic_results: Dict, save_path: str = None):
        """
        Visualize morphogenetic patterns and field dynamics
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Morphogenetic System Patterns', fontsize=16, fontweight='bold')
        
        # Extract data
        morphogenetic_field = morphogenetic_results['morphogenetic_field']
        turing_pattern = morphogenetic_results['turing_pattern']
        pattern_completed = morphogenetic_results['pattern_completed']
        field_complexity = morphogenetic_results['field_complexity']
        
        # Plot 1: Morphogenetic field
        im1 = axes[0, 0].imshow(morphogenetic_field, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Morphogenetic Field\nComplexity: {field_complexity:.3f}')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Turing pattern
        im2 = axes[0, 1].imshow(turing_pattern, cmap='plasma', aspect='auto')
        axes[0, 1].set_title('Turing Pattern Dynamics')
        axes[0, 1].set_xlabel('X Position')
        axes[0, 1].set_ylabel('Y Position')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot 3: Field cross-section
        if len(morphogenetic_field.shape) > 1:
            mid_row = morphogenetic_field.shape[0] // 2
            axes[1, 0].plot(morphogenetic_field[mid_row, :], 'b-', linewidth=2, alpha=0.8)
        else:
            axes[1, 0].plot(morphogenetic_field, 'b-', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Field Cross-Section')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Field Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Pattern completion status
        completion_status = 1 if pattern_completed else 0
        axes[1, 1].bar(['Pattern Completed'], [completion_status], 
                      color='green' if pattern_completed else 'red', alpha=0.7)
        axes[1, 1].set_title('Pattern Completion Status')
        axes[1, 1].set_ylabel('Status (1=Complete, 0=Incomplete)')
        axes[1, 1].set_ylim(0, 1.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_quantum_cognitive_network(self, quantum_cog_results: Dict, save_path: str = None):
        """
        Visualize quantum cognitive network and entanglement
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Quantum Cognitive Processor Network', fontsize=16, fontweight='bold')
        
        # Extract data
        inference_state = quantum_cog_results['inference_state']
        processed_state = quantum_cog_results['processed_state']
        entanglement_network = quantum_cog_results['entanglement_network']
        quantum_coherence = quantum_cog_results['quantum_coherence']
        
        # Plot 1: Inference state
        state_magnitude = np.abs(inference_state)
        state_phase = np.angle(inference_state)
        axes[0, 0].scatter(state_magnitude, state_phase, c=state_phase, 
                          cmap='hsv', s=100, alpha=0.7)
        axes[0, 0].set_title(f'Quantum Inference State\nCoherence: {quantum_coherence:.3f}')
        axes[0, 0].set_xlabel('Magnitude')
        axes[0, 0].set_ylabel('Phase (radians)')
        
        # Plot 2: Processed state comparison
        x = np.arange(len(inference_state))
        axes[0, 1].plot(x, np.real(inference_state), 'b-', linewidth=2, alpha=0.8, label='Inference')
        axes[0, 1].plot(x, np.real(processed_state), 'r-', linewidth=2, alpha=0.8, label='Processed')
        axes[0, 1].set_title('State Processing Comparison')
        axes[0, 1].set_xlabel('State Index')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Entanglement network
        im3 = axes[1, 0].imshow(np.abs(entanglement_network), cmap='hot', aspect='auto')
        axes[1, 0].set_title('Quantum Entanglement Network')
        axes[1, 0].set_xlabel('Quantum State Index')
        axes[1, 0].set_ylabel('Quantum State Index')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Plot 4: Entanglement strength distribution
        entanglement_strengths = np.abs(entanglement_network.flatten())
        axes[1, 1].hist(entanglement_strengths, bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title('Entanglement Strength Distribution')
        axes[1, 1].set_xlabel('Entanglement Strength')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_emergence_metrics(self, emergence_metrics: Dict, save_path: str = None):
        """
        Visualize emergence metrics across all protocols
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Emergent Technology Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Extract metrics
        metrics = list(emergence_metrics.keys())
        values = list(emergence_metrics.values())
        
        # Plot 1: Metrics bar chart
        bars = axes[0, 0].bar(metrics, values, alpha=0.7, color='cyan')
        axes[0, 0].set_title('Emergence Metrics Overview')
        axes[0, 0].set_ylabel('Metric Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Metrics radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_radar = values + values[:1]  # Close the circle
        angles += angles[:1]
        
        axes[0, 1].plot(angles, values_radar, 'o-', linewidth=2, alpha=0.8)
        axes[0, 1].fill(angles, values_radar, alpha=0.25)
        axes[0, 1].set_xticks(angles[:-1])
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].set_title('Metrics Radar Chart')
        axes[0, 1].grid(True)
        
        # Plot 3: Metrics distribution
        axes[1, 0].hist(values, bins=10, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Metrics Value Distribution')
        axes[1, 0].set_xlabel('Metric Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: System health indicator
        overall_health = np.mean(values)
        health_color = 'green' if overall_health > 0.5 else 'orange' if overall_health > 0.3 else 'red'
        
        axes[1, 1].bar(['System Health'], [overall_health], color=health_color, alpha=0.7)
        axes[1, 1].set_title(f'Overall System Health\nScore: {overall_health:.3f}')
        axes[1, 1].set_ylabel('Health Score')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, results: Dict, save_path: str = None):
        """
        Create an interactive dashboard using Plotly
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Quantum Optimization', 'Swarm Intelligence', 
                          'Neuromorphic Dynamics', 'Holographic Encoding',
                          'Morphogenetic Patterns', 'Emergence Metrics'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Quantum optimization plot
        if 'quantum_optimization' in results:
            history = results['quantum_optimization']['optimization_history']
            iterations = [h['iteration'] for h in history]
            objectives = [h['objective'] for h in history]
            
            fig.add_trace(
                go.Scatter(x=iterations, y=objectives, mode='lines+markers',
                          name='Objective Evolution', line=dict(color='cyan')),
                row=1, col=1
            )
        
        # Swarm intelligence heatmap
        if 'swarm_cognitive' in results:
            coordination_matrix = results['swarm_cognitive']['coordination_matrix']
            
            fig.add_trace(
                go.Heatmap(z=coordination_matrix, colorscale='Viridis',
                          name='Coordination Matrix'),
                row=1, col=2
            )
        
        # Neuromorphic dynamics
        if 'neuromorphic' in results:
            spike_history = results['neuromorphic']['spike_history']
            spike_rate = results['neuromorphic']['spike_rate']
            
            # Create spike raster data
            spike_times = []
            neuron_ids = []
            for t in range(spike_history.shape[0]):
                for n in range(spike_history.shape[1]):
                    if spike_history[t, n]:
                        spike_times.append(t)
                        neuron_ids.append(n)
            
            fig.add_trace(
                go.Scatter(x=spike_times, y=neuron_ids, mode='markers',
                          marker=dict(size=2, color='red'), name=f'Spikes (Rate: {spike_rate:.3f})'),
                row=2, col=1
            )
        
        # Holographic encoding
        if 'holographic' in results:
            encoded_data = results['holographic']['encoded_data']
            x = np.arange(len(encoded_data))
            
            fig.add_trace(
                go.Scatter(x=x, y=np.abs(encoded_data), mode='lines',
                          name='Holographic Magnitude', line=dict(color='green')),
                row=2, col=2
            )
        
        # Morphogenetic patterns
        if 'morphogenetic' in results:
            morphogenetic_field = results['morphogenetic']['morphogenetic_field']
            
            fig.add_trace(
                go.Heatmap(z=morphogenetic_field, colorscale='Plasma',
                          name='Morphogenetic Field'),
                row=3, col=1
            )
        
        # Emergence metrics
        if 'emergence_metrics' in results:
            metrics = results['emergence_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name='Emergence Metrics',
                      marker_color='purple'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Emergent Cognitive Network Interactive Dashboard",
            height=1200,
            showlegend=True,
            template="plotly_dark"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def animate_quantum_evolution(self, quantum_results: Dict, save_path: str = None):
        """
        Create animation of quantum state evolution
        """
        history = quantum_results['optimization_history']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def animate(frame):
            ax.clear()
            
            if frame < len(history):
                state = history[frame]['state']
                state_magnitude = np.abs(state)
                state_phase = np.angle(state)
                
                ax.scatter(state_magnitude, state_phase, c=state_phase, 
                          cmap='hsv', s=100, alpha=0.7)
                ax.set_title(f'Quantum State Evolution - Iteration {frame}')
                ax.set_xlabel('Magnitude')
                ax.set_ylabel('Phase (radians)')
                ax.grid(True, alpha=0.3)
        
        anim = FuncAnimation(fig, animate, frames=len(history), 
                           interval=200, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
        
        plt.show()
        return anim


def demonstrate_emergent_network():
    """
    Demonstrate the emergent cognitive network with comprehensive visualization
    """
    print("Emergent Cognitive Network Demonstration")
    print("=" * 50)
    
    # Create sample input data
    input_data = np.random.uniform(-1, 1, 20)
    print(f"Input data shape: {input_data.shape}")
    
    # Execute emergent protocol
    print("\nExecuting Emergent Technology Orchestration Protocol...")
    results = execute_emergent_protocol(input_data)
    
    # Create visualization system
    viz = EmergentVisualization()
    
    print("\nGenerating visualizations...")
    
    # Generate all visualizations
    if 'quantum_optimization' in results:
        viz.plot_quantum_optimization_dynamics(results['quantum_optimization'])
    
    if 'swarm_cognitive' in results:
        viz.plot_swarm_intelligence_patterns(results['swarm_cognitive'])
    
    if 'neuromorphic' in results:
        viz.plot_neuromorphic_dynamics(results['neuromorphic'])
    
    if 'holographic' in results:
        viz.plot_holographic_encoding(results['holographic'])
    
    if 'morphogenetic' in results:
        viz.plot_morphogenetic_patterns(results['morphogenetic'])
    
    if 'quantum_cognitive' in results:
        viz.plot_quantum_cognitive_network(results['quantum_cognitive'])
    
    if 'emergence_metrics' in results:
        viz.plot_emergence_metrics(results['emergence_metrics'])
    
    # Create interactive dashboard
    print("\nCreating interactive dashboard...")
    viz.create_interactive_dashboard(results)
    
    # Print summary
    print("\nEmergent Technology Orchestration Complete!")
    print("=" * 50)
    
    if 'emergence_metrics' in results:
        print("Emergence Metrics Summary:")
        for metric, value in results['emergence_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nProtocol executed with {len(input_data)} input dimensions")
    print("All visualization components generated successfully!")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_emergent_network()