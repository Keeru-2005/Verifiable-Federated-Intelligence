import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _segment_softmax(scores, index, num_nodes):
    """Numerically-stable softmax of ``scores`` grouped by ``index``."""
    max_per_node = torch.full(
        (num_nodes,), float('-inf'), dtype=scores.dtype, device=scores.device
    )
    max_per_node = max_per_node.scatter_reduce(
        0, index, scores, reduce='amax', include_self=True
    )
    shifted = scores - max_per_node[index]
    exp_scores = torch.exp(shifted)
    denom = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    denom = denom.index_add(0, index, exp_scores)
    return exp_scores / denom[index].clamp_min(1e-12)


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

    def forward(self, h, edge_index, num_nodes=None):
        """Sparse graph attention.

        ``edge_index`` is a ``LongTensor`` of shape ``(2, E)`` where
        ``edge_index[0]`` holds destination (target) nodes and
        ``edge_index[1]`` holds source nodes.  Using an edge list instead of a
        dense ``N x N`` adjacency keeps memory proportional to ``E``; at the
        full-corpus scale (11k nodes but only ~350k edges) the dense matrix
        needed ~0.5 GB *per* intermediate tensor and exhausted system memory.
        """
        if num_nodes is None:
            num_nodes = h.size(0)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, num_edges).")

        dst, src = edge_index[0], edge_index[1]

        Wh = torch.mm(h, self.W)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        e = self.leakyrelu(Wh1[dst] + Wh2[src]).squeeze(-1)

        attention = _segment_softmax(e, dst, num_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.zeros(
            num_nodes, self.out_features, dtype=Wh.dtype, device=Wh.device
        )
        h_prime = h_prime.index_add(0, dst, attention.unsqueeze(-1) * Wh[src])
        return h_prime, attention


class MultiHeadGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=4):
        super(MultiHeadGAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, edge_index, num_nodes=None):
        if num_nodes is None:
            num_nodes = x.size(0)
        x = F.dropout(x, self.dropout, training=self.training)

        layer_out = []
        att_matrices = []
        for att in self.attentions:
            h, att_weights = att(x, edge_index, num_nodes)
            layer_out.append(h)
            att_matrices.append(att_weights)

        x = torch.cat(layer_out, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x, out_att = self.out_att(x, edge_index, num_nodes)
        return F.elu(x), layer_out, att_matrices, out_att


class GraphEncoder:
    """
    Graph Attention Network (GAT) encoder to process temporal transaction graphs.

    Leakage-safe usage
    ------------------
    ``fit_transform(G_train, node_labels=...)`` learns the feature normalisation
    statistics and GAT weights from training data only.  ``transform(G_holdout)``
    then applies those frozen artefacts to an unseen graph (for example the
    holdout split) without touching the holdout labels.

    The previous implementation only offered ``fit_transform`` and therefore had
    to be run once on the complete dataset before the train/test split.  That
    made graph features depend on holdout labels.  The explicit fit/transform
    split removes that dependency.
    """

    # Node attributes consumed by the GAT.  No label-derived attributes are
    # allowed in this list: every feature must be computable at inference time.
    FEATURE_KEYS = [
        'pagerank', 'in_degree', 'out_degree', 'clustering_coefficient',
        'betweenness_centrality',
        'cum_sent', 'cum_received', 'txn_count_out', 'txn_count_in',
        'asymmetry', 'velocity', 'burst_score',
        'delta_asymmetry', 'delta_velocity', 'delta_burst',
        'max_burst_score', 'time_to_peak_burst',
    ]

    def __init__(self, dimensions=16, heads=4, epochs=200, prune_threshold=0.1, lr=0.01):
        self.dimensions = dimensions
        self.heads = heads
        self.epochs = epochs
        self.prune_threshold = prune_threshold
        self.lr = lr

        self.node_attention = {}
        self.pruned_edges = []
        self.embeddings_dict = {}

        self.feature_keys = list(self.FEATURE_KEYS)
        self.feature_mean = None
        self.feature_std = None
        self.model = None
        self.fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_tensors(self, G, use_stored_stats):
        nodes = list(G.nodes())
        N = len(nodes)
        if N == 0:
            return nodes, None, None, None

        features = np.zeros((N, len(self.feature_keys)), dtype=np.float32)
        for i, n in enumerate(nodes):
            attr = G.nodes[n]
            for j, key in enumerate(self.feature_keys):
                features[i, j] = attr.get(key, 0.0)

        if use_stored_stats:
            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("GraphEncoder.transform() called before fit_transform().")
            mean, std = self.feature_mean, self.feature_std
        else:
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            std = np.where(std == 0, 1.0, std)
            self.feature_mean = mean
            self.feature_std = std

        features = (features - mean) / std

        # Keep the adjacency sparse: build a COO edge list directly instead of
        # materialising a dense N x N matrix.
        adj_sparse = nx.adjacency_matrix(G, nodelist=nodes, weight=None).tocoo()
        rows = torch.from_numpy(adj_sparse.row.astype(np.int64))
        cols = torch.from_numpy(adj_sparse.col.astype(np.int64))
        # adj[row, col] == 1 means edge row -> col.  In the attention layer,
        # dst consumes information from src, so dst=col and src=row.
        edge_index = torch.stack([cols, rows], dim=0)

        # Self-loops (matches the previous np.fill_diagonal(adj, 1)).
        loops = torch.arange(N, dtype=torch.long)
        edge_index = torch.cat([edge_index, torch.stack([loops, loops], dim=0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1).contiguous()

        return nodes, torch.FloatTensor(features), edge_index, N

    def _encode(self, G):
        """Run a forward pass and populate embeddings/attention/pruned edges."""
        nodes, features_tensor, edge_index, N = self._build_tensors(G, use_stored_stats=True)
        self.nodes = nodes
        if features_tensor is None:
            self.embeddings_dict = {}
            self.node_attention = {}
            self.pruned_edges = []
            return self.embeddings_dict

        self.model.eval()
        with torch.no_grad():
            _, hidden_layers, att_matrices, _ = self.model(features_tensor, edge_index, N)
            embeddings = torch.cat(hidden_layers, dim=1).numpy()

            if len(att_matrices) > 0:
                # Each entry is an (E,) vector of attention weights.  Average
                # over heads, then reduce to one score per node.
                avg_attention = torch.stack(att_matrices).mean(dim=0)
            else:
                avg_attention = torch.zeros(edge_index.size(1))

        src = edge_index[1]
        node_scores = torch.zeros(N, dtype=avg_attention.dtype)
        node_scores = node_scores.scatter_reduce(
            0, src, avg_attention, reduce='amax', include_self=True
        )

        self.embeddings_dict = {str(nodes[i]): embeddings[i] for i in range(N)}
        self.node_attention = {
            str(nodes[i]): float(node_scores[i]) for i in range(N)
        }

        keep = avg_attention >= self.prune_threshold
        kept_src = src[keep].tolist()
        kept_dst = edge_index[0][keep].tolist()
        self.pruned_edges = [
            (nodes[s], nodes[d]) for s, d in zip(kept_src, kept_dst)
        ]

        return self.embeddings_dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_transform(self, G, df=None, node_labels=None):
        """
        Train the GAT on ``G`` and return node embeddings.

        ``node_labels`` must come from the training split only.  The fitted
        feature statistics and model are retained so that ``transform`` can be
        applied to a disjoint holdout graph.
        """
        if G is None or len(G.nodes) == 0:
            logger.warning("Empty graph provided to GraphEncoder.")
            self.fitted = False
            return {}

        nodes, features_tensor, edge_index, N = self._build_tensors(G, use_stored_stats=False)
        self.nodes = nodes

        labels = np.zeros(N, dtype=np.int64)
        mask = np.zeros(N, dtype=bool)

        if node_labels:
            for i, n in enumerate(nodes):
                if n in node_labels:
                    labels[i] = node_labels[n]
                    mask[i] = True

        labels_tensor = torch.LongTensor(labels)
        mask_tensor = torch.BoolTensor(mask)

        nfeat = features_tensor.shape[1]
        nhid = max(self.dimensions // self.heads, 1)

        self.model = MultiHeadGAT(nfeat=nfeat, nhid=nhid, nclass=2, nheads=self.heads)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out, _, _, _ = self.model(features_tensor, edge_index, N)

            if mask.sum() > 0:
                loss = criterion(out[mask_tensor], labels_tensor[mask_tensor])
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == self.epochs:
                    logger.info(f"  [GAT Training] Epoch {epoch + 1}/{self.epochs} "
                                f"({((epoch + 1)/self.epochs)*100:.0f}%) | Loss: {loss.item():.4f}")
            else:
                break

        self.fitted = True
        return self._encode(G)

    def transform(self, G):
        """
        Apply the frozen training-time GAT to a new graph (e.g. the holdout set).

        No labels are accepted or used here, which is the whole point of the
        fit/transform split.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("GraphEncoder.transform() called before fit_transform().")
        if G is None or len(G.nodes) == 0:
            logger.warning("Empty graph provided to GraphEncoder.transform().")
            self.embeddings_dict = {}
            self.node_attention = {}
            self.pruned_edges = []
            return self.embeddings_dict
        return self._encode(G)

    def get_node_attention(self):
        """Returns a dict of {node_id: float} with aggregated attention scores."""
        return self.node_attention

    def get_pruned_edges(self):
        """Returns a list of (sender, receiver) tuples for edges that survived pruning."""
        return self.pruned_edges
