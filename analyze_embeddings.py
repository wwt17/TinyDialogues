import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from hierarchy import dendrogram, linkage, set_link_color_palette
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nltk
from nltk.corpus import wordnet
from nltk.corpus import brown
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/brown')
except LookupError:
    print("Downloading Brown corpus...")
    nltk.download('brown', quiet=True)
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading POS tagger...")
    nltk.download('averaged_perceptron_tagger', quiet=True)


def load_model_and_tokenizer(path):
    """
    Load a causal language model and tokenizer from the given path.

    Args:
        path (str): Path to the model directory or Hugging Face model ID

    Returns:
        model: The loaded causal language model
        tokenizer: The loaded tokenizer
    """
    print(f"Loading model and tokenizer from: {path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
    )

    return model, tokenizer


def get_pos_filtered_embeddings(
    model,
    tokenizer,
    n_tokens=100,
    pos_types={'noun', 'verb'},
    filter_prefix=' ',
    selection_method='first',
    embedding_type='input',
    subtract_mean=True,
):
    """
    Extract input or output embeddings for an even distribution of specified POS types.

    Args:
        model: The causal language model
        tokenizer: The tokenizer
        n_tokens (int): Total number of tokens to select
        pos_types (set): Set of POS types to include {'noun', 'verb', 'adjective', 'adverb'}
        filter_prefix (str): Prefix for tokens (e.g., ' ' for whole words)
        selection_method (str): 'first' or 'random' for token selection
        embedding_type (str): 'input' or 'output' to specify which embeddings to use

    Returns:
        embeddings (np.ndarray): Embedding matrix
        tokens (list): List of token strings with POS tags
        pos_labels (list): List of POS types for each token
    """

    # Map POS types to their codes and tags
    pos_mapping = {
        'noun': ('n', 'NN', '[N]'),
        'verb': ('v', 'VB', '[V]'),
        'adjective': ('a', 'JJ', '[ADJ]'),
        'adverb': ('r', 'RB', '[ADV]')
    }

    vocab_size = len(tokenizer)
    tokens_per_pos = n_tokens // len(pos_types)

    # Initialize collections for each POS type
    pos_collections = {pos: {'indices': [], 'tokens': []} for pos in pos_types}

    print(f"Searching for {tokens_per_pos} tokens for each POS: {pos_types}")
    
    # Get common words from Brown corpus for better POS detection
    try:
        from collections import Counter
        tagged_words = brown.tagged_words()
        word_pos_freq = Counter()

        for word, tag in tagged_words:
            for pos_type, (_, brown_prefix, _) in pos_mapping.items():
                if pos_type in pos_types and tag.startswith(brown_prefix):
                    word_pos_freq[(word.lower(), pos_type)] += 1

        # Create dictionaries of known words for each POS
        known_words_by_pos = {}
        for pos_type in pos_types:
            known_words_by_pos[pos_type] = {
                word for (word, pos), _ in word_pos_freq.most_common(10000) 
                if pos == pos_type
            }
    except:
        print("Using WordNet only for POS detection...")
        known_words_by_pos = {pos: set() for pos in pos_types}

    # Collect all candidate tokens
    candidates_by_pos = {pos: [] for pos in pos_types}

    for i in range(vocab_size):
        try:
            # Decode the token
            token_str = tokenizer.decode([i])

            # Apply prefix filter
            if filter_prefix and not token_str.startswith(filter_prefix):
                continue

            # Clean the token for POS checking
            word = token_str.strip().lower()

            # Skip if too short or contains non-alphabetic characters
            if len(word) <= 2 or not word.isalpha():
                continue

            # Determine POS using multiple methods
            detected_pos = None

            # Method 1: Check against known words from corpus
            for pos_type in pos_types:
                if word in known_words_by_pos[pos_type]:
                    detected_pos = pos_type
                    break

            # Method 2: Use WordNet if not found in corpus
            if detected_pos is None:
                synsets = wordnet.synsets(word)
                if synsets:
                    # Count occurrences of each POS
                    pos_counts = {pos: 0 for pos in pos_types}
                    for synset in synsets:
                        synset_pos = synset.pos()
                        for pos_type, (wordnet_code, _, _) in pos_mapping.items():
                            if pos_type in pos_types and synset_pos == wordnet_code:
                                pos_counts[pos_type] += 1

                    # Assign to the most common POS
                    if any(pos_counts.values()):
                        detected_pos = max(pos_counts, key=pos_counts.get)

            # Add to candidates if POS was detected
            if detected_pos is not None:
                display_token = token_str.replace('\n', '\\n').replace('\t', '\\t')
                if len(display_token) > 15:
                    display_token = display_token[:12] + "..."

                candidates_by_pos[detected_pos].append({
                    'index': i,
                    'token': display_token,
                    'tag': pos_mapping[detected_pos][2]
                })

        except:
            continue

    # Select tokens based on method
    selected_indices = []
    selected_tokens = []
    selected_pos_labels = []

    for pos_type in pos_types:
        candidates = candidates_by_pos[pos_type]

        if len(candidates) == 0:
            print(f"Warning: No {pos_type}s found in vocabulary")
            continue

        # Select tokens
        if selection_method == 'random':
            n_select = min(tokens_per_pos, len(candidates))
            selected = np.random.choice(candidates, size=n_select, replace=False)
        else:  # 'first'
            selected = candidates[:tokens_per_pos]

        # Add to final lists
        for item in selected:
            selected_indices.append(item['index'])
            selected_tokens.append(f"{item['index']}: {item['token']}")
            selected_pos_labels.append(pos_type)

        print(f"Selected {len(selected)}/{len(candidates)} {pos_type}s")

    if len(selected_indices) == 0:
        raise ValueError("No tokens found for specified POS types")

 
    # Get the appropriate embedding layer
    if embedding_type == "input":
        embedding_layer = model.get_input_embeddings()
    elif embedding_type == "output":
        embedding_layer = model.get_output_embeddings()
    else:
        raise ValueError(f"Unknown embedding_type {embedding_type}")

    embedding_weight = embedding_layer.weight

   # Extract embeddings
    with torch.no_grad():
        if subtract_mean:
            embedding_weight = embedding_weight - embedding_weight.mean(dim=0, keepdim=True)
        embedding_matrix = embedding_weight[selected_indices].cpu().numpy()

    return embedding_matrix, selected_tokens, selected_pos_labels


def plot_dendrograms(
    embeddings,
    tokens,
    pos_labels=None,
    method='average',
    metric='cosine',
    figsize=(15, 8),
    save_path=None,
):
    """
    Plot dendrograms of the embeddings using hierarchical clustering with POS-based coloring.
    
    Args:
        embeddings (np.ndarray): Embedding matrix
        tokens (list): List of token strings
        pos_labels (list): List of POS types for each token
        method (str): Linkage method ('ward', 'complete', 'average', 'single')
        metric (str): Distance metric ('euclidean', 'cosine', 'correlation')
        figsize (tuple): Figure size
        max_display (int): Maximum number of tokens to display in the dendrogram
        save_path (str): Path to save the plot as PDF (optional)
    """
    n_tokens = len(tokens)

    # Define colors for each POS type
    pos_colors = {
        'noun': '#2E86AB',      # Blue
        'verb': '#C73E1D',      # Red
        'adjective': '#F18F01', # Orange
        'adverb': '#A23B72',    # Purple
    }

    # Perform hierarchical clustering
    linkage_matrix = linkage(embeddings, method=method, metric=metric)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    if pos_labels:
        pos_sets = [{pos} for pos in pos_labels]
        for link in linkage_matrix:
            pos_set = pos_sets[int(link[0])] | pos_sets[int(link[1])]
            pos_sets.append(pos_set)

    def link_color_func(k):
        pos_set = pos_sets[k]
        if len(pos_set) == 1:
            pos = next(iter(pos_set))
            return pos_colors[pos]
        else:
            return 'black'

    # Plot dendrogram
    dend = dendrogram(
        linkage_matrix,
        labels=tokens,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        link_color_func=link_color_func,
        color_branch=False,
    )

    # Color the labels based on POS
    if pos_labels:
        xlbls = ax.get_xmajorticklabels()
        for lbl, original_idx in zip(xlbls, dend['leaves']):
            lbl.set_color(pos_colors.get(pos_labels[original_idx], 'black'))

    # Create legend
    if pos_labels:
        unique_pos = list(set(pos_labels))
        legend_patches = [
            mpatches.Patch(color=pos_colors.get(pos, 'black'), label=pos.capitalize())
            for pos in unique_pos
        ]
        ax.legend(handles=legend_patches, loc='upper right', frameon=True, fancybox=True)

    ax.set_title(f'Dendrogram of Embeddings\n(Method: {method}, Metric: {metric})')
    ax.set_xlabel('Token')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to PDF if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()

    return linkage_matrix


def analyze_embeddings(embeddings, tokens, pos_labels=None):
    """
    Provide basic statistics about the embeddings.
    
    Args:
        embeddings (np.ndarray): Embedding matrix
        tokens (list): List of token strings
        pos_labels (list): List of POS types for each token
    """
    print("\n" + "="*50)
    print("EMBEDDING STATISTICS")
    print("="*50)
    print(f"Number of tokens: {len(tokens)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
    print(f"Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.4f}")
    print(f"Min norm: {np.min(np.linalg.norm(embeddings, axis=1)):.4f}")
    print(f"Max norm: {np.max(np.linalg.norm(embeddings, axis=1)):.4f}")

    # POS distribution
    if pos_labels:
        from collections import Counter
        pos_counts = Counter(pos_labels)
        print(f"\nPOS Distribution:")
        for pos, count in pos_counts.items():
            print(f"  {pos.capitalize()}: {count}")

    # Check for special tokens (usually have very different norms)
    norms = np.linalg.norm(embeddings, axis=1)
    outlier_threshold = np.mean(norms) + 2 * np.std(norms)
    outliers = np.where(norms > outlier_threshold)[0]

    if len(outliers) > 0:
        print(f"\nPotential outliers (high norm):")
        for idx in outliers[:5]:  # Show first 5 outliers
            print(f"  {tokens[idx]} (norm: {norms[idx]:.4f})")


def main(
        path,
        n_tokens=100,
        pos_types={'noun', 'verb'},
        selection_method='first',
        embedding_type='input',
        method='average',
        metric='cosine',
        save_path=None,
):
    """
    Main function to load model, extract embeddings, and plot dendrograms.

    Args:
        path (str): Path to the model
        n_tokens (int): Total number of tokens to analyze
        pos_types (set): Set of POS types to include {'noun', 'verb', 'adjective', 'adverb'}
        selection_method (str): 'first' or 'random' for token selection
        method (str): Linkage method for hierarchical clustering
        metric (str): Distance metric
        save_path (str): Path to save the plot as PDF (optional)
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(path)

    # Extract embeddings
    pos_str = ", ".join([f"{pos}s" for pos in pos_types])
    print(f"\nExtracting embeddings for {pos_str} ({selection_method} selection)...")

    embeddings, tokens, pos_labels = get_pos_filtered_embeddings(
        model, tokenizer, n_tokens, pos_types,
        selection_method=selection_method,
        embedding_type=embedding_type,
    )

    # Analyze embeddings
    analyze_embeddings(embeddings, tokens, pos_labels)

    # Plot dendrograms with POS-based coloring
    print("\nCreating color-coded dendrogram...")
    linkage_matrix = plot_dendrograms(
        embeddings,
        tokens,
        pos_labels,
        method=method,
        metric=metric,
        save_path=save_path,
    )

    return embeddings, tokens, linkage_matrix, pos_labels


if __name__ == "__main__":
    paths = [
        "trained_GPT2_models/GPT2-small_CHILDES_ordered_1e-04_bs32",
        "gpt2",
    ]
    selection_methods = ['first', 'random']
    embedding_type = 'output'
    pos_types_ = {
        "nv": {'noun', 'verb'},
        "all": {'noun', 'verb', 'adjective', 'adverb'},
    }

    for path in paths:
        model_name = path.split("/")[-1]
        for selection_method in selection_methods:
            for pos_types_name, pos_types in pos_types_.items():
                embeddings, tokens, linkage_matrix, pos_labels = main(
                    path=path,
                    n_tokens=120,
                    pos_types=pos_types,
                    selection_method=selection_method,
                    embedding_type=embedding_type,
                    save_path=f"{model_name}_{selection_method}_{pos_types_name}.png"
                )