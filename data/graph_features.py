"""
Graph feature extraction with an explicit, leakage-safe fit/transform contract.

Rules learned from data (edge-pruning thresholds, feature normalisation, GAT
weights) are fitted on the **training split only**.  ``transform`` can then be
applied to a disjoint holdout split without ever reading its labels.

This replaces the original design in which ``extract_features`` was called on the
full dataset before the train/test split.  That earlier design used
``is_laundering`` to preserve fraudulent edges and to train the GAT, so holdout
labels influenced the features that were later evaluated.

The graph construction below is deliberately **label-free**:

* edge pruning only uses amount / velocity / flow-asymmetry statistics;
* temporal features never aggregate ``is_laundering``;
* the GAT is the only supervised component and receives labels from the
  training split only.  Its weights are frozen before ``transform`` is called.
"""

import logging
import math
import os
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _safe_fill_zero(df):
    return df.fillna(0)


class GraphFeatureExtractor:
    """Fit graph-pruning rules + GAT on train, then transform holdout rows."""

    def __init__(
        self,
        use_gat=True,
        gat_dimensions=16,
        gat_heads=4,
        gat_epochs=200,
        gat_prune_threshold=0.1,
        gat_lr=0.01,
        apply_gat_pruning=True,
    ):
        self.use_gat = use_gat
        self.gat_dimensions = gat_dimensions
        self.gat_heads = gat_heads
        self.gat_epochs = gat_epochs
        self.gat_prune_threshold = gat_prune_threshold
        self.gat_lr = gat_lr
        self.apply_gat_pruning = apply_gat_pruning

        self.thresholds = None
        self.encoder = None
        self.feature_columns = None
        self.fitted = False

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_pairs(df):
        work = df.sort_values(['sender', 'timestamp']).copy()
        work['temporal_delta'] = work.groupby('sender')['timestamp'].diff().fillna(0)
        work['amount_log'] = work['amount'].apply(lambda x: math.log1p(x) if x > 0 else 0)

        pair_stats = work.groupby(['sender', 'receiver']).agg(
            total_amount=('amount', 'sum'),
            avg_amount=('amount', 'mean'),
            max_amount=('amount', 'max'),
            min_amount=('amount', 'min'),
            txn_count=('amount', 'count'),
            first_ts=('timestamp', 'min'),
            last_ts=('timestamp', 'max'),
            std_amount=('amount', 'std'),
        ).reset_index().fillna(0)

        pair_stats['time_span'] = (pair_stats['last_ts'] - pair_stats['first_ts']).clip(lower=1)
        pair_stats['velocity'] = pair_stats['total_amount'] / pair_stats['time_span']
        return work, pair_stats

    @staticmethod
    def _asymmetry_scores(df):
        out_flow = df.groupby('sender')['amount'].sum()
        in_flow = df.groupby('receiver')['amount'].sum()
        all_accounts = set(out_flow.index) | set(in_flow.index)
        asymmetry = {}
        for acc in all_accounts:
            out_val = float(out_flow.get(acc, 0))
            in_val = float(in_flow.get(acc, 0))
            total = out_val + in_val
            asymmetry[acc] = abs(out_val - in_val) / max(total, 1)
        return asymmetry

    def _fit_thresholds(self, pair_stats, df):
        amount_threshold = float(pair_stats['total_amount'].quantile(0.25))

        stage1 = pair_stats[pair_stats['total_amount'] >= amount_threshold]
        if stage1.empty:
            stage1 = pair_stats

        avg_txn_count = float(stage1['txn_count'].mean())
        median_time_span = float(stage1['time_span'].median())

        # If every surviving pair is a single transaction there is no support for
        # a "repeat interaction" rule.  Disable it consistently for train and
        # holdout instead of silently falling back to a different rule set.
        use_burst_filter = avg_txn_count > 1.0
        if not use_burst_filter:
            logging.info("  Burst filter disabled: training pairs are all single transactions.")

        asym = self._asymmetry_scores(df)
        if asym:
            asym_threshold = float(np.percentile(list(asym.values()), 60))
        else:
            asym_threshold = 0.0

        return {
            'amount_threshold': amount_threshold,
            'avg_txn_count': avg_txn_count,
            'median_time_span': median_time_span,
            'use_burst_filter': bool(use_burst_filter),
            'asym_threshold': asym_threshold,
        }

    def _apply_pruning_rules(self, pair_stats, df):
        """Apply train-fitted, label-free pruning rules to a pair table."""
        t = self.thresholds

        # Rule 1: amount threshold
        filtered = pair_stats[pair_stats['total_amount'] >= t['amount_threshold']].copy()
        logging.info(f"  Amount filter: {len(pair_stats)} → {len(filtered)} edges")

        # Rule 2: temporal burst / repeated interaction (only when supported)
        if t.get('use_burst_filter', True):
            before = len(filtered)
            filtered = filtered[
                (filtered['txn_count'] >= t['avg_txn_count']) |
                ((filtered['txn_count'] > 1) & (filtered['time_span'] <= t['median_time_span']))
            ].copy()
            logging.info(f"  Temporal burst filter: {before} → {len(filtered)} edges")
        else:
            logging.info("  Temporal burst filter: skipped (no training support)")

        # Rule 3: flow asymmetry (computed from the incoming rows, never labels)
        asym = self._asymmetry_scores(df)
        high_asym = {acc for acc, score in asym.items() if score >= t['asym_threshold']}
        before = len(filtered)
        filtered = filtered[
            filtered['sender'].isin(high_asym) | filtered['receiver'].isin(high_asym)
        ].copy()
        logging.info(f"  Flow asymmetry filter: {before} → {len(filtered)} edges")

        if filtered.empty:
            logging.warning("  All edges pruned; falling back to the amount-filtered pairs to keep the graph usable.")
            filtered = pair_stats[pair_stats['total_amount'] >= t['amount_threshold']].copy()
        if filtered.empty:
            filtered = pair_stats.copy()

        return filtered

    def _build_graph(self, df, fit):
        work, pair_stats = self._prepare_pairs(df)
        logging.info(f"Unique pairs before filtering: {len(pair_stats)}")

        if fit:
            self.thresholds = self._fit_thresholds(pair_stats, df)
        elif self.thresholds is None:
            raise RuntimeError("GraphFeatureExtractor.transform() called before fit().")

        kept_pairs = self._apply_pruning_rules(pair_stats, df)

        G = nx.from_pandas_edgelist(
            kept_pairs,
            source='sender',
            target='receiver',
            edge_attr=['total_amount', 'avg_amount', 'max_amount', 'txn_count', 'velocity'],
            create_using=nx.DiGraph(),
        )

        logging.info(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        # NOTE: pruned pairs only shape the *graph* used to compute node features.
        # Transactions are never dropped here, so the holdout keeps every naturally
        # distributed row and evaluation remains on the untouched population.
        return G, work, kept_pairs

    # ------------------------------------------------------------------
    # Node feature computation
    # ------------------------------------------------------------------
    @staticmethod
    def _structural_features(G):
        if G.number_of_nodes() == 0:
            return pd.DataFrame(columns=[
                'account', 'pagerank', 'in_degree', 'out_degree',
                'clustering_coefficient', 'betweenness_centrality',
            ])

        logging.info("Calculating PageRank (damping=0.85)...")
        pagerank_dict = nx.pagerank(G, alpha=0.85, weight='total_amount')
        logging.info("Calculating In/Out Degrees...")
        in_degree_dict = dict(G.in_degree(weight=None))
        out_degree_dict = dict(G.out_degree(weight=None))
        logging.info("Calculating Clustering Coefficient...")
        clustering_dict = nx.clustering(nx.DiGraph(G))
        logging.info("Calculating Betweenness Centrality...")
        k = min(100, G.number_of_nodes())
        betweenness_dict = nx.betweenness_centrality(G, k=k, weight='total_amount', seed=42)

        for node in G.nodes():
            G.nodes[node]['pagerank'] = pagerank_dict.get(node, 0)
            G.nodes[node]['in_degree'] = in_degree_dict.get(node, 0)
            G.nodes[node]['out_degree'] = out_degree_dict.get(node, 0)
            G.nodes[node]['clustering_coefficient'] = clustering_dict.get(node, 0)
            G.nodes[node]['betweenness_centrality'] = betweenness_dict.get(node, 0)

        return pd.DataFrame({
            'account': list(G.nodes()),
            'pagerank': [pagerank_dict.get(n, 0) for n in G.nodes()],
            'in_degree': [in_degree_dict.get(n, 0) for n in G.nodes()],
            'out_degree': [out_degree_dict.get(n, 0) for n in G.nodes()],
            'clustering_coefficient': [clustering_dict.get(n, 0) for n in G.nodes()],
            'betweenness_centrality': [betweenness_dict.get(n, 0) for n in G.nodes()],
        })

    @staticmethod
    def _temporal_features(df):
        data_path = os.path.dirname(os.path.abspath(__file__))
        if data_path not in sys.path:
            sys.path.insert(0, data_path)

        from temporal_bipartite_graph import (
            build_temporal_bipartite_graph,
            compute_snapshot_features,
            build_final_graph_for_gat,
        )

        snapshots, df_with_time = build_temporal_bipartite_graph(df)
        snapshots, account_features_over_time = compute_snapshot_features(snapshots, df_with_time)
        G_temporal = build_final_graph_for_gat(snapshots, account_features_over_time)

        temporal_feature_keys = [
            'cum_sent', 'cum_received', 'txn_count_out', 'txn_count_in',
            'asymmetry', 'velocity', 'burst_score',
            'delta_asymmetry', 'delta_velocity', 'delta_burst',
            'max_burst_score', 'time_to_peak_burst',
        ]

        rows = []
        for node in G_temporal.nodes():
            acc = node.replace('acc_', '')
            row = {'account': acc}
            for feat in temporal_feature_keys:
                row[feat] = G_temporal.nodes[node].get(feat, 0.0)
            rows.append(row)

        temporal_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['account'] + temporal_feature_keys)
        return temporal_df, G_temporal, df_with_time, temporal_feature_keys

    def _add_gat_features(self, G, node_features, df, transactions, fit, train_labels=None):
        if not self.use_gat:
            node_features['gat_attention_score'] = 0.0
            return transactions, node_features

        models_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
        )
        if models_path not in sys.path:
            sys.path.insert(0, models_path)

        from graph_encoder import GraphEncoder

        if fit:
            self.encoder = GraphEncoder(
                dimensions=self.gat_dimensions,
                heads=self.gat_heads,
                epochs=self.gat_epochs,
                prune_threshold=self.gat_prune_threshold,
                lr=self.gat_lr,
            )
            node_labels = {}
            if train_labels is not None:
                fraud_senders = set(train_labels)
                node_labels = {n: (1 if n in fraud_senders else 0) for n in G.nodes()}
            embeddings = self.encoder.fit_transform(G, df=df, node_labels=node_labels)
        else:
            embeddings = self.encoder.transform(G)

        node_attn = self.encoder.get_node_attention()

        node_features['gat_attention_score'] = node_features['account'].map(
            lambda acc: node_attn.get(acc, 0.0)
        )
        for dim in range(self.encoder.dimensions):
            node_features[f'gat_emb_{dim}'] = node_features['account'].map(
                lambda acc, d=dim: float(
                    embeddings.get(str(acc), np.zeros(self.encoder.dimensions))[d]
                )
            )

        # GAT edge pruning is intentionally NOT used to drop evaluation rows.
        # Dropping holdout transactions based on a learned edge mask would change
        # the evaluation population.  Attention remains available as a feature.
        return transactions, node_features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df):
        """Fit pruning thresholds + GAT using training rows (labels allowed here)."""
        if 'is_laundering' not in df.columns:
            raise ValueError("fit() requires the training label column 'is_laundering'.")

        train_labels = set(df.loc[df['is_laundering'] == 1, 'sender'].unique())
        self._fit_transform_internal(df, fit=True, train_labels=train_labels)
        self.fitted = True
        return self

    def transform(self, df):
        """Apply frozen training artefacts to holdout rows.  Labels are ignored."""
        if not self.fitted:
            raise RuntimeError("GraphFeatureExtractor.transform() called before fit().")
        return self._fit_transform_internal(df, fit=False, train_labels=None)

    def fit_transform(self, df):
        if 'is_laundering' not in df.columns:
            raise ValueError("fit_transform() requires the training label column 'is_laundering'.")

        train_labels = set(df.loc[df['is_laundering'] == 1, 'sender'].unique())
        out = self._fit_transform_internal(df, fit=True, train_labels=train_labels)
        self.fitted = True
        return out

    def _fit_transform_internal(self, df, fit, train_labels):
        G, transactions, kept_pairs = self._build_graph(df, fit=fit)

        node_features = self._structural_features(G)

        # Temporal features
        try:
            temporal_df, G_temporal, work_with_time, temporal_keys = self._temporal_features(transactions)
            node_features = node_features.merge(
                temporal_df, on='account', how='left'
            ).fillna(0)
            logging.info(f"Temporal features added: {len(temporal_keys)} new features per account")

            for node in G_temporal.nodes():
                acc = node.replace('acc_', '')
                if G.has_node(acc):
                    for feat in temporal_keys:
                        G.nodes[acc][feat] = G_temporal.nodes[node].get(feat, 0.0)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.warning(f"Temporal bipartite graph failed: {exc} — continuing without temporal features.")

        # GAT encoder (train-only supervision; frozen for transform)
        try:
            transactions, node_features = self._add_gat_features(
                G, node_features, df, transactions, fit=fit, train_labels=train_labels
            )
        except Exception as exc:
            logging.warning(f"GraphEncoder GAT execution failed or not found ({exc}) — skipping GAT.")
            if 'gat_attention_score' not in node_features.columns:
                node_features['gat_attention_score'] = 0.0

        logging.info("Merging node features to transactions...")
        out = transactions.merge(
            node_features.add_suffix('_sender'),
            left_on='sender', right_on='account_sender', how='left'
        ).drop(columns=['account_sender'])

        out = out.merge(
            node_features.add_suffix('_receiver'),
            left_on='receiver', right_on='account_receiver', how='left'
        ).drop(columns=['account_receiver'])

        out = _safe_fill_zero(out)
        self.feature_columns = [
            c for c in out.columns
            if c not in ('is_laundering', 'txn_id', 'timestamp', 'sender', 'receiver')
        ]
        logging.info("Feature extraction complete.")
        return out


# ----------------------------------------------------------------------
# Backwards-compatible helper (one-shot, NOT leakage-safe by itself)
# ----------------------------------------------------------------------
def extract_features(df):
    """
    Backward-compatible one-shot helper using causal transaction features.

    For train/validation/test experiments, use one extractor instance and call
    ``fit_transform(train)``, ``transform(validation)``, then
    ``transform(test)`` so state flows forward in time without labels.
    """
    logging.warning("One-shot feature extraction treats the full input as historical context.")
    return CausalTransactionFeatureExtractor().fit_transform(df)


class CausalTransactionFeatureExtractor:
    """Create transaction graph features from strictly prior transactions.

    Calls to ``fit_transform`` and ``transform`` share an append-only history.
    Rows with the same timestamp are featurized before any row in that timestamp
    is added to history, so a transaction cannot use itself or a simultaneous
    transaction as evidence.  Call ``transform`` on validation and then test data
    in chronological order to model a forward-looking deployment.

    This deliberately avoids the previous full-split graph snapshot and
    supervised GAT embeddings: those representations allowed later transactions
    (and, for the GAT, training labels for the same account) to shape features of
    earlier transactions.
    """

    FEATURE_NAMES = (
        "amount_log", "temporal_delta",
        "prior_txn_count_sender", "prior_txn_count_receiver",
        "prior_amount_sent_sender", "prior_amount_received_sender",
        "prior_amount_sent_receiver", "prior_amount_received_receiver",
        "prior_out_degree_sender", "prior_in_degree_sender",
        "prior_out_degree_receiver", "prior_in_degree_receiver",
        "prior_sender_imbalance", "prior_receiver_imbalance",
        "prior_sender_time_gap", "prior_pair_txn_count", "prior_pair_amount",
        "prior_sender_avg_sent", "prior_receiver_avg_received",
        "prior_pair_avg_amount", "prior_pair_time_gap",
        "prior_sender_velocity", "prior_receiver_velocity",
        "prior_sender_counterparty_ratio", "prior_receiver_counterparty_ratio",
    )

    def __init__(self):
        self.fitted = False
        self._accounts = {}
        self._pairs = {}
        self._last_timestamp = None

    @staticmethod
    def _account_state():
        return {
            "sent": 0.0, "received": 0.0, "out_count": 0, "in_count": 0,
            "out_neighbors": set(), "in_neighbors": set(), "last_time": None,
            "first_time": None,
        }

    def fit_transform(self, df):
        if "timestamp" not in df.columns or "sender" not in df.columns or "receiver" not in df.columns:
            raise ValueError("Causal graph features require timestamp, sender, and receiver columns.")
        if self.fitted:
            raise RuntimeError("This extractor is already fitted; create a new instance for another run.")
        self.fitted = True
        return self._transform_rows(df)

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Call fit_transform() on the training period before transform().")
        return self._transform_rows(df)

    def _transform_rows(self, df):
        if df.empty:
            out = df.copy()
            for name in self.FEATURE_NAMES:
                out[name] = pd.Series(dtype="float64")
            return out

        ordered = df.sort_values(["timestamp", "txn_id"], kind="mergesort").reset_index(drop=True) if "txn_id" in df.columns else df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        if self._last_timestamp is not None and ordered["timestamp"].min() < self._last_timestamp:
            raise ValueError("Causal feature data must be transformed in nondecreasing timestamp order.")

        generated = {name: np.zeros(len(ordered), dtype=np.float32) for name in self.FEATURE_NAMES}
        for timestamp, batch in ordered.groupby("timestamp", sort=True):
            batch_updates = []
            for position, row in enumerate(batch.itertuples(index=False)):
                sender = getattr(row, "sender")
                receiver = getattr(row, "receiver")
                amount = float(getattr(row, "amount"))
                s = self._accounts.get(sender, self._account_state())
                r = self._accounts.get(receiver, self._account_state())
                pair = self._pairs.get((sender, receiver), (0, 0.0, None))
                s_total = s["sent"] + s["received"]
                r_total = r["sent"] + r["received"]
                pair_last_time = pair[2]
                feature = {
                    "amount_log": math.log1p(max(amount, 0.0)),
                    "temporal_delta": 0.0 if s["last_time"] is None else float(timestamp - s["last_time"]),
                    "prior_txn_count_sender": float(s["out_count"]),
                    "prior_txn_count_receiver": float(r["in_count"]),
                    "prior_amount_sent_sender": s["sent"],
                    "prior_amount_received_sender": s["received"],
                    "prior_amount_sent_receiver": r["sent"],
                    "prior_amount_received_receiver": r["received"],
                    "prior_out_degree_sender": float(len(s["out_neighbors"])),
                    "prior_in_degree_sender": float(len(s["in_neighbors"])),
                    "prior_out_degree_receiver": float(len(r["out_neighbors"])),
                    "prior_in_degree_receiver": float(len(r["in_neighbors"])),
                    "prior_sender_imbalance": abs(s["sent"] - s["received"]) / max(s_total, 1.0),
                    "prior_receiver_imbalance": abs(r["sent"] - r["received"]) / max(r_total, 1.0),
                    "prior_sender_time_gap": 0.0 if s["last_time"] is None else float(timestamp - s["last_time"]),
                    "prior_pair_txn_count": float(pair[0]),
                    "prior_pair_amount": float(pair[1]),
                    "prior_sender_avg_sent": s["sent"] / max(s["out_count"], 1),
                    "prior_receiver_avg_received": r["received"] / max(r["in_count"], 1),
                    "prior_pair_avg_amount": float(pair[1]) / max(pair[0], 1),
                    "prior_pair_time_gap": 0.0 if pair_last_time is None else float(timestamp - pair_last_time),
                    "prior_sender_velocity": s["sent"] / max(float(timestamp - s["first_time"]), 1.0)
                    if s["first_time"] is not None else 0.0,
                    "prior_receiver_velocity": r["received"] / max(float(timestamp - r["first_time"]), 1.0)
                    if r["first_time"] is not None else 0.0,
                    "prior_sender_counterparty_ratio": len(s["out_neighbors"]) / max(s["out_count"], 1),
                    "prior_receiver_counterparty_ratio": len(r["in_neighbors"]) / max(r["in_count"], 1),
                }
                row_index = batch.index[position]
                for name, value in feature.items():
                    generated[name][row_index] = value
                batch_updates.append((sender, receiver, amount))

            # Update history only after all rows at this timestamp were featurized.
            for sender, receiver, amount in batch_updates:
                s = self._accounts.setdefault(sender, self._account_state())
                r = self._accounts.setdefault(receiver, self._account_state())
                s["sent"] += amount
                s["out_count"] += 1
                s["out_neighbors"].add(receiver)
                s["last_time"] = timestamp
                if s["first_time"] is None:
                    s["first_time"] = timestamp
                r["received"] += amount
                r["in_count"] += 1
                r["in_neighbors"].add(sender)
                if r["first_time"] is None:
                    r["first_time"] = timestamp
                pair = self._pairs.setdefault((sender, receiver), [0, 0.0, None])
                pair[0] += 1
                pair[1] += amount
                pair[2] = timestamp
            self._last_timestamp = timestamp

        feature_df = pd.DataFrame(generated, index=ordered.index)
        return pd.concat([ordered, feature_df], axis=1)
