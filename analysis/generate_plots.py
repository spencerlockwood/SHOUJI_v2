"""
Generate plots matching the paper's figures
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class PlotGenerator:
    """Generate plots for paper comparison"""
    
    def __init__(self, results_dir: str = 'results', plots_dir: str = 'plots'):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
    
    def load_results(self, filename: str) -> Dict:
        """Load results from JSON file"""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_false_accept_rate(self, results: Dict):
        """
        Plot false accept rate across configurations (Figure 3 from paper)
        """
        # Organize data by sequence length
        data_by_length = {}
        
        for config_key, result in results.items():
            seq_length = result['seq_length']
            edit_threshold = result['edit_threshold']
            far = result['false_accept_rate']
            
            if seq_length not in data_by_length:
                data_by_length[seq_length] = {'E': [], 'FAR': []}
            
            data_by_length[seq_length]['E'].append(edit_threshold)
            data_by_length[seq_length]['FAR'].append(far)
        
        # Create subplots for each sequence length
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (seq_length, data) in enumerate(sorted(data_by_length.items())):
            ax = axes[idx]
            
            # Sort by edit threshold
            sorted_indices = np.argsort(data['E'])
            E_values = np.array(data['E'])[sorted_indices]
            FAR_values = np.array(data['FAR'])[sorted_indices]
            
            ax.plot(E_values, FAR_values, marker='o', linewidth=2, 
                   markersize=8, label='Shouji')
            
            ax.set_xlabel('Edit Distance Threshold (E)', fontsize=12)
            ax.set_ylabel('False Accept Rate (%)', fontsize=12)
            ax.set_title(f'Sequence Length = {seq_length} bp', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        output_file = os.path.join(self.plots_dir, 'false_accept_rate.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_false_reject_rate(self, results: Dict):
        """
        Plot false reject rate (should be 0% for Shouji)
        """
        seq_lengths = sorted(set(r['seq_length'] for r in results.values()))
        edit_thresholds = sorted(set(r['edit_threshold'] for r in results.values()))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for seq_length in seq_lengths:
            E_vals = []
            FRR_vals = []
            
            for E in edit_thresholds:
                config_key = f"L{seq_length}_E{E}"
                if config_key in results:
                    E_vals.append(E)
                    FRR_vals.append(results[config_key]['false_reject_rate'])
            
            ax.plot(E_vals, FRR_vals, marker='o', linewidth=2, 
                   markersize=8, label=f'Length={seq_length}')
        
        ax.set_xlabel('Edit Distance Threshold (E)', fontsize=12)
        ax.set_ylabel('False Reject Rate (%)', fontsize=12)
        ax.set_title('False Reject Rate (Should be 0%)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        output_file = os.path.join(self.plots_dir, 'false_reject_rate.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_execution_time_comparison(self, results: Dict):
        """
        Plot execution time comparison between Shouji and Edlib
        """
        configs = []
        shouji_times = []
        edlib_times = []
        
        for config_key, result in sorted(results.items()):
            label = f"L{result['seq_length']}\nE{result['edit_threshold']}"
            configs.append(label)
            shouji_times.append(result['avg_shouji_time_ms'])
            edlib_times.append(result['avg_edlib_time_ms'])
        
        x = np.arange(len(configs))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, shouji_times, width, label='Shouji', color='#2ecc71')
        bars2 = ax.bar(x + width/2, edlib_times, width, label='Edlib', color='#e74c3c')
        
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Average Execution Time (ms)', fontsize=12)
        ax.set_title('Execution Time Comparison: Shouji vs Edlib', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(self.plots_dir, 'execution_time_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_speedup(self, results: Dict):
        """
        Plot speedup factor of Shouji over Edlib
        """
        configs = []
        speedups = []
        
        for config_key, result in sorted(results.items()):
            label = f"L{result['seq_length']}, E{result['edit_threshold']}"
            configs.append(label)
            speedups.append(result['speedup'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(configs, speedups, color='#3498db', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title('Shouji Speedup Over Edlib', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = os.path.join(self.plots_dir, 'speedup.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_scalability(self, results: Dict):
        """
        Plot false accept rate vs edit threshold
        """
        E_values = []
        FAR_values = []
        FRR_values = []
        
        for config_key, result in sorted(results.items(), 
                                        key=lambda x: x[1]['edit_threshold']):
            E_values.append(result['edit_threshold'])
            FAR_values.append(result['false_accept_rate'])
            FRR_values.append(result['false_reject_rate'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(E_values, FAR_values, marker='o', linewidth=2, 
               markersize=8, label='False Accept Rate', color='#e74c3c')
        ax.plot(E_values, FRR_values, marker='s', linewidth=2, 
               markersize=8, label='False Reject Rate', color='#2ecc71')
        
        ax.set_xlabel('Edit Distance Threshold (E)', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Shouji Scalability Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        output_file = os.path.join(self.plots_dir, 'scalability.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_confusion_matrix(self, results: Dict, config_key: str):
        """
        Plot confusion matrix for a specific configuration
        """
        if config_key not in results:
            print(f"Configuration {config_key} not found")
            return
        
        result = results[config_key]
        
        # Create confusion matrix
        cm = np.array([
            [result['true_negatives'], result['false_positives']],
            [result['false_negatives'], result['true_positives']]
        ])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Dissimilar', 'Predicted Similar'],
                   yticklabels=['Actually Dissimilar', 'Actually Similar'],
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title(f'Confusion Matrix: {config_key}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_file = os.path.join(self.plots_dir, f'confusion_matrix_{config_key}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all plots"""
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60 + "\n")
        
        # Load results
        try:
            accuracy_results = self.load_results('accuracy_results.json')
            scalability_results = self.load_results('scalability_results.json')
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run 'python analysis/run_experiments.py' first")
            return
        
        # Generate plots
        print("Creating false accept rate plot...")
        self.plot_false_accept_rate(accuracy_results)
        
        print("Creating false reject rate plot...")
        self.plot_false_reject_rate(accuracy_results)
        
        print("Creating execution time comparison...")
        self.plot_execution_time_comparison(accuracy_results)
        
        print("Creating speedup plot...")
        self.plot_speedup(accuracy_results)
        
        print("Creating scalability plot...")
        self.plot_scalability(scalability_results)
        
        # Create confusion matrices for select configurations
        print("Creating confusion matrices...")
        for config in ['L100_E2', 'L100_E5', 'L250_E2']:
            if config in accuracy_results:
                self.plot_confusion_matrix(accuracy_results, config)
        
        print("\n" + "="*60)
        print("ALL PLOTS GENERATED")
        print("="*60)
        print(f"\nPlots saved in '{self.plots_dir}/' directory")


def main():
    """Generate all plots"""
    generator = PlotGenerator()
    generator.generate_all_plots()


if __name__ == '__main__':
    main()