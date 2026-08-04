import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GATLayer(nn.Module):
    """
    A Graph Attention Layer (GAT) implemented in pure PyTorch.
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        
        e = self.leakyrelu(Wh1 + Wh2.T)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        return h_prime, attention


class MultiHeadGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=4):
        super(MultiHeadGAT, self).__init__()
        self.dropout = dropout
        
        self.attentions = nn.ModuleList([GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        layer_out = []
        att_matrices = []
        for att in self.attentions:
            h, att_mat = att(x, adj)
            layer_out.append(h)
            att_matrices.append(att_mat)
            
        x = torch.cat(layer_out, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x, out_att_mat = self.out_att(x, adj)
        return F.elu(x), layer_out, att_matrices, out_att_mat


class GraphEncoder:
    """
    Graph Attention Network (GAT) encoder to process temporal transaction graphs.
    """
    def __init__(self, dimensions=16, heads=4, epochs=200, prune_threshold=0.1, lr=0.01):
        self.dimensions = dimensions
        self.heads = heads
        self.epochs = epochs
        self.prune_threshold = prune_threshold
        self.lr = lr
        
        self.node_attention = {}
        self.pruned_edges = []
        self.embeddings_dict = {}

    def fit_transform(self, G, df=None, node_labels=None):
        """
        Train the GAT model and extract node embeddings.
        Returns a dict of node_id -> embedding (numpy array of shape (dimensions,)).
        """
        if not G or len(G.nodes) == 0:
            logger.warning("Empty graph provided to GraphEncoder.")
            return {}

        self.nodes = list(G.nodes())
        N = len(self.nodes)

        feature_keys = [
            'pagerank', 'in_degree', 'out_degree', 'clustering_coefficient', 'betweenness_centrality',
            'cum_sent', 'cum_received', 'txn_count_out', 'txn_count_in', 'asymmetry', 'velocity', 
            'burst_score', 'fraud_exposure', 'delta_asymmetry', 'delta_velocity', 'delta_burst', 
            'delta_fraud_exposure', 'max_burst_score', 'time_to_peak_burst'
        ]
        
        features = np.zeros((N, len(feature_keys)))
        for i, n in enumerate(self.nodes):
            attr = G.nodes[n]
            for j, key in enumerate(feature_keys):
                features[i, j] = attr.get(key, 0.0)
                
        # Normalize features
        std = features.std(axis=0)
        std[std == 0] = 1.0
        features = (features - features.mean(axis=0)) / std
        
        adj = nx.adjacency_matrix(G, nodelist=self.nodes).toarray()
        np.fill_diagonal(adj, 1)

        features_tensor = torch.FloatTensor(features)
        adj_tensor = torch.FloatTensor(adj)

        labels = np.zeros(N, dtype=np.int64)
        mask = np.zeros(N, dtype=np.bool_)
        
        if node_labels:
            for i, n in enumerate(self.nodes):
                if n in node_labels:
                    labels[i] = node_labels[n]
                    mask[i] = True
                    
        labels_tensor = torch.LongTensor(labels)
        mask_tensor = torch.BoolTensor(mask)
        
        nfeat = features.shape[1]
        nhid = max(self.dimensions // self.heads, 1)
        
        model = MultiHeadGAT(nfeat=nfeat, nhid=nhid, nclass=2, nheads=self.heads)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out, _, _, _ = model(features_tensor, adj_tensor)
            
            if mask.sum() > 0:
                loss = criterion(out[mask_tensor], labels_tensor[mask_tensor])
                loss.backward()
                optimizer.step()
            else:
                break
                
        model.eval()
        with torch.no_grad():
            _, hidden_layers, att_matrices, _ = model(features_tensor, adj_tensor)
            embeddings = torch.cat(hidden_layers, dim=1).numpy()
            
            if len(att_matrices) > 0:
                avg_attention = torch.mean(torch.stack(att_matrices), dim=0).numpy()
            else:
                avg_attention = np.zeros((N, N))
            
        self.embeddings_dict = {str(self.nodes[i]): embeddings[i] for i in range(N)}
        
        # Calculate aggregated attention score per node (e.g. max incoming attention)
        node_att_scores = avg_attention.max(axis=0)
        self.node_attention = {str(self.nodes[i]): float(node_att_scores[i]) for i in range(N)}
        
        self.pruned_edges = []
        for i in range(N):
            for j in range(N):
                if G.has_edge(self.nodes[i], self.nodes[j]):
                    if avg_attention[i, j] >= self.prune_threshold:
                        self.pruned_edges.append((self.nodes[i], self.nodes[j]))

        return self.embeddings_dict

    def get_node_attention(self):
        """
        Returns a dict of {node_id: float} with aggregated attention scores.
        """
        return self.node_attention
        
    def get_pruned_edges(self):
        """
        Returns a list of (sender, receiver) tuples for edges that survived pruning.
        """
        return self.pruned_edges
