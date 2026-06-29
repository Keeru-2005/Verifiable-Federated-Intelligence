import networkx as nx
import numpy as np
import pandas as pd
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GraphEncoder:

    def __init__(self, dimensions=16, heads=4, epochs=300,
                 prune_threshold=0.1, lr=0.005, dropout=0.3,
                 seed=42, n_folds=3):
        self.dimensions       = dimensions
        self.heads            = heads
        self.epochs           = epochs
        self.prune_threshold  = prune_threshold
        self.lr               = lr
        self.dropout          = dropout
        self.seed             = seed
        self.n_folds          = n_folds
        self.attention_scores = {}
        self.pruned_edge_list = []
        self.node_attention   = {}
        self.gat_model        = None

    def _build_node_features(self, G, df):
        structural_keys = [
            'pagerank', 'in_degree', 'out_degree',
            'clustering_coefficient', 'betweenness_centrality'
        ]

        if 'amount_log' not in df.columns:
            df['amount_log'] = df['amount'].apply(lambda x: math.log1p(x) if x > 0 else 0)

        sender_stats = df.groupby('sender').agg(
            total_sent    = ('amount', 'sum'),
            avg_sent      = ('amount', 'mean'),
            std_sent      = ('amount', 'std'),
            txn_count_out = ('amount', 'count'),
            first_ts      = ('timestamp', 'min'),
            last_ts       = ('timestamp', 'max'),
        ).fillna(0)

        receiver_stats = df.groupby('receiver').agg(
            total_received = ('amount', 'sum'),
            avg_received   = ('amount', 'mean'),
            txn_count_in   = ('amount', 'count'),
        ).fillna(0)

        sender_stats['time_window'] = (sender_stats['last_ts'] - sender_stats['first_ts']).clip(lower=1)
        sender_stats['velocity']    = sender_stats['total_sent'] / sender_stats['time_window']

        combined   = sender_stats[['total_sent']].join(receiver_stats[['total_received']], how='outer').fillna(0)
        total_flow = (combined['total_sent'] + combined['total_received']).clip(lower=1)
        combined['asymmetry'] = (combined['total_sent'] - combined['total_received']).abs() / total_flow

        global_mean = sender_stats['avg_sent'].mean()
        global_std  = sender_stats['avg_sent'].std() + 1e-8
        sender_stats['behavioral_anomaly'] = ((sender_stats['avg_sent'] - global_mean) / global_std).abs()

        node_feat_dict = {}
        for node in G.nodes():
            attrs      = G.nodes[node]
            structural = [float(attrs.get(k, 0.0)) for k in structural_keys]
            s = sender_stats.loc[node]   if node in sender_stats.index   else None
            r = receiver_stats.loc[node] if node in receiver_stats.index else None
            c = combined.loc[node]       if node in combined.index       else None
            behavioral = [
                float(s['total_sent'])        if s is not None else 0.0,
                float(s['avg_sent'])           if s is not None else 0.0,
                float(s['velocity'])           if s is not None else 0.0,
                float(s['behavioral_anomaly']) if s is not None else 0.0,
                float(s['txn_count_out'])      if s is not None else 0.0,
                float(r['total_received'])     if r is not None else 0.0,
                float(r['txn_count_in'])       if r is not None else 0.0,
                float(c['asymmetry'])          if c is not None else 0.0,
            ]
            node_feat_dict[node] = structural + behavioral

        n_feats = len(structural_keys) + 8
        logging.info(f"Node feature vector size: {n_feats} per account")
        return node_feat_dict, n_feats

    def _augment(self, x, edge_index, drop_rate=0.1, noise_std=0.01):
        import torch
        # randomly drop edges
        n_edges    = edge_index.shape[1]
        keep_mask  = torch.rand(n_edges) > drop_rate
        edge_index = edge_index[:, keep_mask]
        # add small gaussian noise to node features
        x = x + torch.randn_like(x) * noise_std
        return x, edge_index

    def _train_single_fold(self, data, n_feats, train_mask, val_mask, torch, F, GATConv):
        import torch as t
        hidden = max(16, self.dimensions)

        class _GAT(t.nn.Module):
            def __init__(self, in_ch, hidden_ch, out_ch, heads, dropout):
                super().__init__()
                self.conv1      = GATConv(in_ch, hidden_ch, heads=heads, dropout=dropout, concat=True)
                self.conv2      = GATConv(hidden_ch * heads, out_ch, heads=1, dropout=dropout, concat=False)
                self.classifier = t.nn.Linear(out_ch, 2)
                self.bn1        = t.nn.BatchNorm1d(hidden_ch * heads)
                self.dropout    = dropout

            def forward(self, x, edge_index):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
                x = self.bn1(x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
                return x, F.log_softmax(self.classifier(F.relu(x)), dim=1), attn2

        n_legit      = max((data.y == 0).sum().item(), 1)
        n_fraud      = max((data.y == 1).sum().item(), 1)
        class_weight = t.tensor([n_fraud / n_legit, 1.0], dtype=t.float)

        model     = _GAT(in_ch=n_feats, hidden_ch=hidden, out_ch=self.dimensions,
                         heads=self.heads, dropout=self.dropout)
        optimizer = t.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-5
        )

        best_val_loss  = float('inf')
        best_state     = None
        patience_count = 0
        early_stop     = 30

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            # augment each epoch — prevents memorization
            x_aug, ei_aug = self._augment(data.x.clone(), data.edge_index.clone())
            _, logits, _  = model(x_aug, ei_aug)
            train_loss    = F.nll_loss(logits[train_mask], data.y[train_mask], weight=class_weight)
            train_loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # validation loss on clean data
            model.eval()
            with t.no_grad():
                _, val_logits, _ = model(data.x, data.edge_index)
                val_loss = F.nll_loss(val_logits[val_mask], data.y[val_mask], weight=class_weight)

            scheduler.step(val_loss)

            if val_loss.item() < best_val_loss:
                best_val_loss  = val_loss.item()
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if epoch % 30 == 0:
                logging.info(f"    Epoch {epoch:03d} | Train: {train_loss.item():.4f} | Val: {val_loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if patience_count >= early_stop:
                logging.info(f"    Early stopping at epoch {epoch} | Best val loss: {best_val_loss:.4f}")
                break

        model.load_state_dict(best_state)
        return model

    def fit_transform(self, G, df=None, node_labels=None):
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.data import Data
            from torch_geometric.nn import GATConv
        except ImportError:
            logging.error("torch-geometric not installed. Run: pip install torch torch-geometric")
            return {str(n): np.zeros(self.dimensions) for n in G.nodes()}

        torch.manual_seed(self.seed)
        logging.info(f"GAT encoder started — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        nodes       = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx_to_node = {i: n for n, i in node_to_idx.items()}
        n_nodes     = len(nodes)

        edges      = list(G.edges())
        src        = [node_to_idx[u] for u, v in edges]
        dst        = [node_to_idx[v] for u, v in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        if df is not None:
            node_feat_dict, n_feats = self._build_node_features(G, df)
        else:
            structural_keys = ['pagerank', 'in_degree', 'out_degree',
                               'clustering_coefficient', 'betweenness_centrality']
            node_feat_dict  = {n: [float(G.nodes[n].get(k, 0.0)) for k in structural_keys] for n in nodes}
            n_feats         = len(structural_keys)

        feat_matrix = np.array([node_feat_dict.get(n, [0.0] * n_feats) for n in nodes])
        feat_matrix = (feat_matrix - feat_matrix.mean(axis=0)) / (feat_matrix.std(axis=0) + 1e-8)
        x           = torch.tensor(feat_matrix, dtype=torch.float)

        if node_labels is not None:
            y = torch.tensor([int(node_labels.get(n, 0)) for n in nodes], dtype=torch.long)
            logging.info(f"Supervised mode — fraud nodes: {y.sum().item()}/{n_nodes}")
        else:
            degrees = torch.tensor([G.degree(n) for n in nodes], dtype=torch.float)
            y       = (degrees > degrees.mean() + degrees.std()).long()
            logging.info("Unsupervised mode — high-degree heuristic.")

        data = Data(x=x, edge_index=edge_index, y=y)

        # cross-validation — n_folds different train/val/test splits
        logging.info(f"Cross-validation: {self.n_folds} folds...")
        fold_attention_scores = []
        fold_embeddings       = []

        indices = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(self.seed))
        fold_size = n_nodes // self.n_folds

        for fold in range(self.n_folds):
            logging.info(f"  Fold {fold + 1}/{self.n_folds}")

            val_start  = fold * fold_size
            val_end    = val_start + fold_size
            val_idx    = indices[val_start:val_end]
            train_idx  = torch.cat([indices[:val_start], indices[val_end:]])

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx]     = True

            model = self._train_single_fold(data, n_feats, train_mask, val_mask, torch, F, GATConv)

            model.eval()
            with torch.no_grad():
                node_emb, _, attn = model(data.x, data.edge_index)

            node_emb    = node_emb.numpy()
            edge_idx_np = attn[0].numpy()
            attn_scores = attn[1].squeeze().numpy()

            fold_attn = {}
            for i, score in enumerate(attn_scores):
                s_node = idx_to_node[edge_idx_np[0][i]]
                d_node = idx_to_node[edge_idx_np[1][i]]
                key    = (s_node, d_node)
                if key not in fold_attn or score > fold_attn[key]:
                    fold_attn[key] = float(score)

            fold_attention_scores.append(fold_attn)
            fold_embeddings.append(node_emb)

            self.gat_model = model

        # average attention scores across all folds
        logging.info("Averaging attention scores across folds...")
        all_keys = set()
        for fold_attn in fold_attention_scores:
            all_keys.update(fold_attn.keys())

        self.attention_scores = {}
        for key in all_keys:
            scores = [fold_attn[key] for fold_attn in fold_attention_scores if key in fold_attn]
            self.attention_scores[key] = float(np.mean(scores))

        # average embeddings across folds
        avg_embeddings = np.mean(fold_embeddings, axis=0)

        # edge pruning on averaged scores
        before = len(self.attention_scores)
        self.pruned_edge_list = [
            edge for edge, score in self.attention_scores.items()
            if score >= self.prune_threshold
        ]
        after   = len(self.pruned_edge_list)
        removed = before - after
        logging.info(f"Edge pruning complete: {before} → {after} edges ({removed} removed, {removed/max(before,1)*100:.1f}% graph reduction)")

        node_attn_sum   = {}
        node_attn_count = {}
        for (s, _), score in self.attention_scores.items():
            node_attn_sum[s]   = node_attn_sum.get(s, 0) + score
            node_attn_count[s] = node_attn_count.get(s, 0) + 1
        self.node_attention = {
            n: node_attn_sum[n] / node_attn_count[n]
            for n in node_attn_sum
        }

        embeddings = {str(nodes[i]): avg_embeddings[i] for i in range(n_nodes)}
        logging.info(f"GAT encoding complete — embedding dim: {self.dimensions}")
        return embeddings

    def get_attention_scores(self):
        return self.attention_scores

    def get_pruned_edges(self):
        return self.pruned_edge_list

    def get_node_attention(self):
        return self.node_attention

    def save_model(self, path):
        if self.gat_model:
            import torch
            torch.save(self.gat_model.state_dict(), path)
            logging.info(f"Model saved to {path}")