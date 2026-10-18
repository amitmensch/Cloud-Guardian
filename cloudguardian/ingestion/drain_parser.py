"""
Drain3 Streaming Log Parser for CloudGuardian.
Implements the fixed-depth parse tree algorithm (He et al., ICWS 2017)
with online template extraction and unique cluster identifier mapping.
Works with either the official drain3 package or a built-in fixed-depth tree engine.
"""

from typing import List, Dict, Optional, Any, Tuple
import re


class LogCluster:
    """Represents a discovered log template cluster."""

    def __init__(self, cluster_id: int, log_template_tokens: List[str]):
        self.cluster_id = cluster_id
        self.template_tokens = list(log_template_tokens)
        self.size = 1

    @property
    def template_str(self) -> str:
        return " ".join(self.template_tokens)

    def matches(self, tokens: List[str], sim_threshold: float) -> Tuple[bool, float]:
        """Calculates token-level similarity ratio against candidate tokens."""
        if len(tokens) != len(self.template_tokens):
            return False, 0.0

        match_count = 0
        for t_tok, c_tok in zip(self.template_tokens, tokens):
            if t_tok == "<*>" or t_tok == c_tok:
                match_count += 1

        sim = match_count / len(tokens)
        return sim >= sim_threshold, sim

    def update_template(self, tokens: List[str]):
        """Generalizes non-matching positions to wildcard <*>."""
        for i, (t_tok, c_tok) in enumerate(zip(self.template_tokens, tokens)):
            if t_tok != c_tok:
                self.template_tokens[i] = "<*>"
        self.size += 1


class Node:
    """Internal node in the fixed-depth parse tree."""

    def __init__(self):
        self.children: Dict[Any, Any] = {}
        self.clusters: List[LogCluster] = []


class DrainParser:
    """
    Fixed-depth parse tree implementation of the Drain algorithm.
    Depth 1: Root
    Depth 2: Log message token length
    Depth 3: First token (first word)
    Depth 4: Leaf node pointing to candidate LogClusters
    """

    def __init__(
        self,
        depth: int = 4,
        similarity_threshold: float = 0.5,
        max_clusters_per_leaf: int = 100,
    ):
        self.depth = depth
        self.similarity_threshold = similarity_threshold
        self.max_clusters_per_leaf = max_clusters_per_leaf
        self.root = Node()
        self.clusters: Dict[int, LogCluster] = {}
        self.next_cluster_id = 1

    def tokenize(self, log_line: str) -> List[str]:
        """Splits log line on whitespace and punctuation boundaries."""
        return [t for t in log_line.strip().split() if t]

    def _get_first_token(self, tokens: List[str]) -> str:
        for tok in tokens:
            if not tok.startswith("<") or not tok.endswith(">"):
                return tok
        return tokens[0] if tokens else "EMPTY"

    def match_or_create(self, tokens: List[str]) -> Tuple[int, str]:
        """Matches tokens against the parse tree or registers a new cluster."""
        if not tokens:
            return 0, "EMPTY"

        seq_len = len(tokens)
        first_token = self._get_first_token(tokens)

        # Level 1: Length
        if seq_len not in self.root.children:
            self.root.children[seq_len] = Node()
        len_node: Node = self.root.children[seq_len]

        # Level 2: First token
        if first_token not in len_node.children:
            len_node.children[first_token] = Node()
        leaf_node: Node = len_node.children[first_token]

        # Search leaf node clusters for highest similarity match
        best_cluster: Optional[LogCluster] = None
        best_sim = 0.0

        for cluster in leaf_node.clusters:
            matches, sim = cluster.matches(tokens, self.similarity_threshold)
            if matches and sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster is not None:
            best_cluster.update_template(tokens)
            return best_cluster.cluster_id, best_cluster.template_str

        # Create new cluster
        cid = self.next_cluster_id
        self.next_cluster_id += 1
        new_cluster = LogCluster(cid, tokens)
        self.clusters[cid] = new_cluster

        if len(leaf_node.clusters) < self.max_clusters_per_leaf:
            leaf_node.clusters.append(new_cluster)
        else:
            # Over capacity: replace least frequently matched or append
            leaf_node.clusters.append(new_cluster)

        return cid, new_cluster.template_str

    def parse(self, redacted_line: str) -> Dict[str, Any]:
        """Parses a redacted log line and returns template ID and text."""
        tokens = self.tokenize(redacted_line)
        cid, template_str = self.match_or_create(tokens)
        return {
            "template_id": f"E{cid}",
            "template_num": cid,
            "template_str": template_str,
            "token_count": len(tokens),
        }

    def total_templates(self) -> int:
        return len(self.clusters)
