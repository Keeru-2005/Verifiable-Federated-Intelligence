import os
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Visualizer:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_class_imbalance(self):
        logging.info("Generating Class Imbalance Bar Plot...")
        df = pd.read_csv(self.data_path)
        plt.figure(figsize=(8, 6))
        
        counts = df['isMoneyLaundering'].value_counts()
        labels = ['Legitimate (0)', 'Laundering (1)']
        
        sns.barplot(x=labels, y=counts.values, hue=labels, palette=["#2ecc71", "#e74c3c"])
        plt.title("AML Dataset: Class Imbalance")
        plt.ylabel("Transaction Count")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "class_imbalance.png"), dpi=300)
        plt.close()
        logging.info("Class imbalance plot saved.")
        
    def plot_correlation_heatmap(self, features):
        logging.info("Generating Correlation Heatmap...")
        df = pd.read_csv(self.data_path)
        
        # Ensure we only correlate available numeric features
        existing_features = [f for f in features if f in df.columns]
        if not existing_features:
            logging.warning("No requested features found in dataset for correlation.")
            return
            
        corr = df[existing_features].corr(method='spearman')
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
        plt.title("Spearman Feature Correlation (Graph Embeddings)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"), dpi=300)
        plt.close()
        logging.info("Correlation heatmap saved.")
        
    def plot_network_topology(self, sample_size=200):
        logging.info("Generating Network Topology Sub-Graph...")
        df = pd.read_csv(self.data_path).head(sample_size)
        
        G = nx.from_pandas_edgelist(
            df, source='nameOrig', target='nameDest', 
            edge_attr=['amount', 'isMoneyLaundering'], create_using=nx.DiGraph()
        )
        
        # Identify illicit nodes
        illicit_edges = [(u, v) for u, v, d in G.edges(data=True) if d['isMoneyLaundering'] == 1]
        
        pos = nx.spring_layout(G, k=0.5, seed=42)
        plt.figure(figsize=(14, 14))
        
        # Draw base graph
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100, alpha=0.7)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2, arrows=True)
        
        # Draw illicit pathways prominently
        nx.draw_networkx_edges(G, pos, edgelist=illicit_edges, edge_color='red', width=2.0, arrows=True)
        
        plt.title(f"Transaction Network Sub-Graph (n={sample_size}) | Red = Illicit Edge")
        plt.axis("off")
        plt.savefig(os.path.join(self.output_dir, "network_topology.png"), dpi=300)
        plt.close()
        logging.info("Network topology plot saved.")

    def plot_fl_convergence(self, rounds_log_path):
        """Plots the learning curve from Federated FL rounds if available."""
        if not os.path.exists(rounds_log_path):
            logging.warning(f"FL logs not found at {rounds_log_path}")
            return
            
        logging.info("Generating FL Convergence Plot...")
        df = pd.read_csv(rounds_log_path)
        
        plt.figure(figsize=(8, 6))
        plt.plot(df['round'], df['global_auc'], marker='o', linestyle='-', color='#3498db', linewidth=2)
        plt.title("Federated Aggregation: Global Model AUC Convergence")
        plt.xlabel("Communication Round")
        plt.ylabel("Global AUC-ROC")
        plt.grid(True, alpha=0.4)
        plt.savefig(os.path.join(self.output_dir, "convergence_curve.png"), dpi=300)
        plt.close()
        logging.info("Convergence plot saved.")

if __name__ == "__main__":
    v = Visualizer(
        data_path="data/amlnet.csv", 
        output_dir="visualizations/"
    )
    # The try blocks prevent crashing if amlnet.csv isn't populated yet
    try:
        v.plot_class_imbalance()
        v.plot_correlation_heatmap(['amount', 'is_laundering', 'timestamp']) 
        v.plot_network_topology()
    except Exception as e:
        logging.error(f"Could not generate plots. Ensure data exists. {e}")
