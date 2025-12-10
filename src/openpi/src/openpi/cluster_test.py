"""Tests for clustering functionality with t-SNE visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.manifold import TSNE

from openpi.analysis.clustering import (
    ClusterConfig,
    ProjectionRecord,
    cluster_projections,
    load_projections,
    rank_clusters_by_keywords,
    save_clusters,
)


# Path to the test FFN projections file
TEST_PROJECTIONS_PATH = Path("test_outputs/test_ffn_projections.json")


def test_load_projections():
    """Test loading FFN projections from JSON file."""
    if not TEST_PROJECTIONS_PATH.exists():
        pytest.skip(f"Test projections file not found: {TEST_PROJECTIONS_PATH}")
    
    records = load_projections(TEST_PROJECTIONS_PATH)
    assert len(records) > 0, "Should load at least some projections"
    
    # Check that records have the expected structure
    first_record = records[0]
    assert hasattr(first_record, "layer_index")
    assert hasattr(first_record, "neuron_index")
    assert hasattr(first_record, "value_vector")
    assert hasattr(first_record, "top_tokens")
    assert isinstance(first_record.value_vector, np.ndarray)
    assert len(first_record.value_vector) > 0
    
    print(f"✓ Loaded {len(records)} projection records")
    print(f"  - Value vector dimension: {first_record.value_vector.shape}")
    print(f"  - Top tokens: {first_record.top_tokens}")
    
    return records


def test_cluster_raw_weight_space():
    """Test clustering in raw weight space (default behavior - no projection)."""
    if not TEST_PROJECTIONS_PATH.exists():
        pytest.skip(f"Test projections file not found: {TEST_PROJECTIONS_PATH}")
    
    records = load_projections(TEST_PROJECTIONS_PATH)
    
    # Cluster in raw weight space (default - no semantic projection)
    config = ClusterConfig(
        num_clusters=10,
        max_iterations=100,
        normalize=True,
        random_seed=42,
        use_semantic_space=False,  # Default: cluster raw value vectors
    )
    
    print(f"\n📊 Clustering {len(records)} projections in RAW WEIGHT SPACE...")
    print(f"   Algorithm: K-means with Euclidean distance")
    print(f"   Number of clusters: {config.num_clusters}")
    print(f"   Normalize vectors: {config.normalize}")
    
    clusters = cluster_projections(records, config)
    
    assert len(clusters) > 0, "Should create at least some clusters"
    assert len(clusters) <= config.num_clusters, "Should not exceed requested clusters"
    
    # Check cluster structure
    total_members = sum(len(cluster.members) for cluster in clusters)
    assert total_members == len(records), "All records should be assigned to clusters"
    
    print(f"✓ Created {len(clusters)} clusters")
    for cluster in clusters[:3]:  # Show first 3 clusters
        print(f"  Cluster {cluster.cluster_id}: {len(cluster.members)} neurons")
        top_tokens = ", ".join(token for token, _ in cluster.token_summary[:5])
        print(f"    Top tokens: [{top_tokens}]")
    
    return clusters, records


def test_cluster_semantic_space():
    """Test clustering in semantic token space (requires model)."""
    if not TEST_PROJECTIONS_PATH.exists():
        pytest.skip(f"Test projections file not found: {TEST_PROJECTIONS_PATH}")
    
    # Check if PyTorch is available
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch not available, skipping semantic space clustering test")
    
    records = load_projections(TEST_PROJECTIONS_PATH)
    
    # For semantic clustering, we would need model components
    # This test is skipped unless model checkpoint is provided
    pytest.skip(
        "Semantic space clustering requires model checkpoint. "
        "Run this test manually with model components loaded."
    )


def test_save_and_load_clusters():
    """Test saving and loading clusters."""
    if not TEST_PROJECTIONS_PATH.exists():
        pytest.skip(f"Test projections file not found: {TEST_PROJECTIONS_PATH}")
    
    records = load_projections(TEST_PROJECTIONS_PATH)
    config = ClusterConfig(num_clusters=5, random_seed=42)
    clusters = cluster_projections(records, config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_clusters.json"
        save_clusters(clusters, output_path)
        
        assert output_path.exists(), "Cluster file should be created"
        
        # Verify JSON structure
        import json
        loaded = json.loads(output_path.read_text())
        assert len(loaded) == len(clusters)
        assert "cluster_id" in loaded[0]
        assert "members" in loaded[0]
        assert "token_summary" in loaded[0]
        
        print(f"✓ Saved and verified {len(loaded)} clusters to {output_path}")


def test_rank_clusters_by_keywords():
    """Test ranking clusters by keyword matching."""
    if not TEST_PROJECTIONS_PATH.exists():
        pytest.skip(f"Test projections file not found: {TEST_PROJECTIONS_PATH}")
    
    records = load_projections(TEST_PROJECTIONS_PATH)
    config = ClusterConfig(num_clusters=10, random_seed=42)
    clusters = cluster_projections(records, config)
    
    keywords = ["up", "down", "move", "action"]
    scored = rank_clusters_by_keywords(clusters, keywords)
    
    assert len(scored) == len(clusters)
    assert all(isinstance(score, float) for _, score in scored)
    
    print(f"\n🔍 Ranking clusters by keywords: {keywords}")
    for cluster, score in scored[:3]:
        tokens = ", ".join(token for token, _ in cluster.token_summary[:5])
        print(f"  Cluster {cluster.cluster_id}: score={score:.3f}, tokens=[{tokens}]")
    
    # Top clusters should have higher scores
    assert scored[0][1] >= scored[-1][1], "Clusters should be sorted by score (descending)"


def test_tsne_visualization():
    """Create t-SNE visualization of clusters (optional visual test)."""
    if not TEST_PROJECTIONS_PATH.exists():
        pytest.skip(f"Test projections file not found: {TEST_PROJECTIONS_PATH}")
    
    records = load_projections(TEST_PROJECTIONS_PATH)
    
    # Limit to reasonable size for visualization
    if len(records) > 1000:
        records = records[:1000]
        print(f"⚠ Limiting to first 1000 records for visualization")
    
    config = ClusterConfig(num_clusters=10, random_seed=42, normalize=True)
    clusters = cluster_projections(records, config)
    
    # Get value vectors and cluster assignments
    value_vectors = np.stack([rec.value_vector for rec in records], axis=0)
    
    # Create assignment map
    cluster_assignments = {}
    for cluster in clusters:
        for layer_idx, neuron_idx in cluster.members:
            # Find index in records
            for idx, rec in enumerate(records):
                if rec.layer_index == layer_idx and rec.neuron_index == neuron_idx:
                    cluster_assignments[idx] = cluster.cluster_id
                    break
    
    assignments = np.array([cluster_assignments.get(i, -1) for i in range(len(records))])
    
    # Normalize for t-SNE
    if config.normalize:
        norms = np.linalg.norm(value_vectors, axis=1, keepdims=True) + 1e-8
        value_vectors = value_vectors / norms
    
    print(f"\n📉 Computing t-SNE visualization for {len(records)} projections...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(records) - 1))
    embeddings_2d = tsne.fit_transform(value_vectors)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=assignments,
        cmap="tab10",
        alpha=0.6,
        s=20,
    )
    
    ax.set_title(f"t-SNE Visualization of FFN Value Vectors\n{len(clusters)} Clusters in Raw Weight Space")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    
    output_path = Path("test_outputs/cluster_tsne_visualization.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved t-SNE visualization to {output_path}")
    plt.close()


def test_clustering_summary():
    """Print a comprehensive summary of the clustering process."""
    if not TEST_PROJECTIONS_PATH.exists():
        pytest.skip(f"Test projections file not found: {TEST_PROJECTIONS_PATH}")
    
    print("\n" + "=" * 80)
    print("CLUSTERING SUMMARY")
    print("=" * 80)
    
    records = load_projections(TEST_PROJECTIONS_PATH)
    print(f"\n📁 Loaded {len(records)} FFN projections from {TEST_PROJECTIONS_PATH}")
    
    # Analyze layer distribution
    layers = [rec.layer_index for rec in records]
    layer_counts = {layer: layers.count(layer) for layer in set(layers)}
    print(f"\n📊 Layer distribution:")
    for layer in sorted(layer_counts.keys()):
        print(f"   Layer {layer}: {layer_counts[layer]} neurons")
    
    # Clustering configuration
    config = ClusterConfig(
        num_clusters=10,
        max_iterations=100,
        normalize=True,
        random_seed=42,
        use_semantic_space=False,  # RAW WEIGHT SPACE (no projection)
    )
    
    print(f"\n⚙️  Clustering Configuration:")
    print(f"   Algorithm: K-means (Euclidean distance)")
    print(f"   Space: {'Semantic Token Space' if config.use_semantic_space else 'Raw Weight Space'}")
    print(f"   Number of clusters: {config.num_clusters}")
    print(f"   Normalize: {config.normalize}")
    print(f"   Max iterations: {config.max_iterations}")
    
    if not config.use_semantic_space:
        print(f"\n   ⚠️  Note: Clustering in RAW WEIGHT SPACE (value vectors NOT projected to token space)")
        print(f"      To use semantic clustering, set use_semantic_space=True and provide model components")
    
    # Perform clustering
    print(f"\n🔄 Running K-means clustering...")
    clusters = cluster_projections(records, config)
    
    print(f"\n✅ Clustering Results:")
    print(f"   Created {len(clusters)} clusters")
    
    # Cluster sizes
    cluster_sizes = [len(cluster.members) for cluster in clusters]
    print(f"\n📈 Cluster sizes:")
    print(f"   Min: {min(cluster_sizes)} neurons")
    print(f"   Max: {max(cluster_sizes)} neurons")
    print(f"   Mean: {np.mean(cluster_sizes):.1f} neurons")
    print(f"   Median: {np.median(cluster_sizes):.1f} neurons")
    
    # Show all clusters
    print(f"\n🔍 Cluster Details:")
    for cluster in clusters:
        print(f"\n   Cluster {cluster.cluster_id} ({len(cluster.members)} neurons):")
        top_tokens = [token for token, _ in cluster.token_summary[:5]]
        print(f"      Top tokens: {', '.join(top_tokens)}")
        print(f"      Sample neurons: {cluster.members[:5]}...")
    
    print("\n" + "=" * 80)
    
    return clusters, records


if __name__ == "__main__":
    # Run tests interactively
    print("Running clustering tests...")
    
    # Load projections
    records = test_load_projections()
    
    # Test ranking
    test_rank_clusters_by_keywords()
    
    # Save clusters
    test_save_and_load_clusters()
    
    # Generate summary
    test_clustering_summary()
    
    # Optional: Generate t-SNE visualization
    # test_tsne_visualization()
    
    print("\n✅ All tests completed!")