"""
Attention Visualization Tools

This module provides tools for visualizing attention weights and patterns
to help understand how attention mechanisms work.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple

class AttentionVisualizer:
    """
    Visualization tools for attention mechanisms.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the attention visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
        
    def plot_attention_weights(self, attention_weights: np.ndarray, 
                              title: str = "Attention Weights",
                              labels: Optional[List[str]] = None,
                              figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weight matrix, shape (seq_len_q, seq_len_k)
            title: Title for the plot
            labels: Optional labels for sequence positions
            figsize: Figure size (uses default if None)
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(attention_weights, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   cbar=True,
                   ax=ax)
        
        # Set labels
        if labels is not None:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)
        
        ax.set_title(title)
        ax.set_xlabel('Key Positions')
        ax.set_ylabel('Query Positions')
        
        plt.tight_layout()
        plt.show()
    
    def plot_multi_head_attention(self, attention_weights: np.ndarray,
                                 title: str = "Multi-Head Attention",
                                 labels: Optional[List[str]] = None,
                                 figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot attention weights for multi-head attention.
        
        Args:
            attention_weights: Multi-head attention weights, shape (batch_size, num_heads, seq_len_q, seq_len_k)
            title: Title for the plot
            labels: Optional labels for sequence positions
            figsize: Figure size (uses default if None)
        """
        if figsize is None:
            figsize = (self.figsize[0] * 1.5, self.figsize[1] * 1.5)
            
        batch_size, num_heads, seq_len_q, seq_len_k = attention_weights.shape
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        # Plot each head
        for head_idx in range(min(num_heads, 8)):  # Show max 8 heads
            ax = axes[head_idx]
            
            # Get attention weights for this head (first batch)
            head_weights = attention_weights[0, head_idx]
            
            # Create heatmap
            sns.heatmap(head_weights, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='Blues',
                       cbar=False,
                       ax=ax)
            
            ax.set_title(f'Head {head_idx + 1}')
            
            # Set labels for first and last plots only to avoid clutter
            if head_idx == 0:
                if labels is not None:
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_yticklabels(labels, rotation=0)
                ax.set_xlabel('Key Positions')
                ax.set_ylabel('Query Positions')
        
        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_attention_comparison(self, attention_weights_list: List[np.ndarray],
                                 titles: List[str],
                                 labels: Optional[List[str]] = None,
                                 figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Compare attention weights from different models or layers.
        
        Args:
            attention_weights_list: List of attention weight matrices
            titles: List of titles for each subplot
            labels: Optional labels for sequence positions
            figsize: Figure size (uses default if None)
        """
        if figsize is None:
            figsize = (self.figsize[0] * 1.5, self.figsize[1])
            
        num_plots = len(attention_weights_list)
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        
        # Handle single subplot case
        if num_plots == 1:
            axes = [axes]
        
        for idx, (attention_weights, title) in enumerate(zip(attention_weights_list, titles)):
            ax = axes[idx]
            
            # Create heatmap
            sns.heatmap(attention_weights, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='Blues',
                       cbar=True,
                       ax=ax)
            
            ax.set_title(title)
            
            # Set labels for first plot only
            if idx == 0 and labels is not None:
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels, rotation=0)
                ax.set_xlabel('Key Positions')
                ax.set_ylabel('Query Positions')
        
        plt.tight_layout()
        plt.show()
    
    def plot_attention_evolution(self, attention_weights_sequence: List[np.ndarray],
                                step_labels: List[str],
                                title: str = "Attention Evolution",
                                labels: Optional[List[str]] = None,
                                figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot how attention weights evolve over time or steps.
        
        Args:
            attention_weights_sequence: List of attention weight matrices over time
            step_labels: Labels for each time step
            title: Title for the plot
            labels: Optional labels for sequence positions
            figsize: Figure size (uses default if None)
        """
        if figsize is None:
            figsize = (self.figsize[0] * 1.5, self.figsize[1])
            
        num_steps = len(attention_weights_sequence)
        fig, axes = plt.subplots(1, num_steps, figsize=figsize)
        
        # Handle single subplot case
        if num_steps == 1:
            axes = [axes]
        
        for idx, (attention_weights, step_label) in enumerate(zip(attention_weights_sequence, step_labels)):
            ax = axes[idx]
            
            # Create heatmap
            sns.heatmap(attention_weights, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='Blues',
                       cbar=True,
                       ax=ax)
            
            ax.set_title(f'{step_label}')
            
            # Set labels for first plot only
            if idx == 0 and labels is not None:
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels, rotation=0)
                ax.set_xlabel('Key Positions')
                ax.set_ylabel('Query Positions')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_sample_visualization(self) -> None:
        """
        Create a sample visualization to demonstrate the visualizer.
        """
        print("Creating sample attention visualization...")
        
        # Create sample attention weights
        seq_len = 5
        attention_weights = np.array([
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.1, 0.7, 0.15, 0.03, 0.02],
            [0.05, 0.1, 0.6, 0.2, 0.05],
            [0.02, 0.05, 0.15, 0.65, 0.13],
            [0.01, 0.02, 0.05, 0.12, 0.8]
        ])
        
        # Create sample labels
        labels = ['Word1', 'Word2', 'Word3', 'Word4', 'Word5']
        
        # Plot single attention weights
        self.plot_attention_weights(attention_weights, 
                                  "Sample Attention Weights",
                                  labels)
        
        # Create sample multi-head attention
        batch_size, num_heads = 1, 4
        multi_head_weights = np.random.rand(batch_size, num_heads, seq_len, seq_len)
        # Normalize to make it look like attention weights
        for b in range(batch_size):
            for h in range(num_heads):
                multi_head_weights[b, h] = self._softmax(multi_head_weights[b, h])
        
        # Plot multi-head attention
        self.plot_multi_head_attention(multi_head_weights,
                                     "Sample Multi-Head Attention",
                                     labels)
        
        print("✅ Sample visualizations created!")
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along specified axis."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Example usage and testing
if __name__ == "__main__":
    # Create visualizer
    visualizer = AttentionVisualizer()
    
    # Create sample visualization
    visualizer.create_sample_visualization()
    
    print("\n✅ Attention visualizer working correctly!")
