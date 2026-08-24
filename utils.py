"""
Helper class to visualize weights calculated by the antenna attention module.
"""

import mlflow
import torch

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class AttentionAnalyzer:
    def __init__(self, model):
        self.model = model

    def get_attention_weights(self, input_data):
        self.model.eval()

        # Forward pass first to get the attention weights
        with torch.no_grad():
            _ = self.model(input_data)

        # Get stored weights from attention module
        antenna_weights = None
        if hasattr(self.model.antenna_attention, 'current_weights'):
            antenna_weights = self.model.antenna_attention.current_weights

        # Convert to numpy arrays
        antenna_weights = antenna_weights.mean(dim=1)  # Average over heads
        antenna_weights = antenna_weights[0].cpu().numpy()  # Take first batch

        return antenna_weights

    def visualize(self, input_dataloader, save_path):
        # Get attention weights
        a_w = []
        for input_data, _ in input_dataloader:
            antenna_weights = self.get_attention_weights(input_data)
            a_w.append(antenna_weights)
        a_w = np.stack(a_w).T

        fig, ax = plt.subplots(1, 1, figsize=(30, 8))

        sns.heatmap(a_w, ax=ax, cmap='viridis')
        ax.set_title('Antenna Attention Pattern')
        ax.set_ylabel('Antenna')
        ax.set_xlabel('Sample')

        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), save_path + '.png')
