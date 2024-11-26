CLIP_CONFIG = {
    "fixed_names": [
        "clip.class_embedding",
        "clip.positional_embedding",
        "clip.conv1.weight",
        "clip.ln_pre.weight",
        "clip.ln_pre.bias",
    ],
    "fixed_layers": ["clip.conv1", "embedding", "clip.ln_pre"],
    "common_layers": ["clip.ln_post", "clip.proj", "classifier"],
}

ROBERTA_BASE_CONFIG = {
    "fixed_names": [
        "roberta.embeddings.word_embeddings.weight",
        "roberta.embeddings.position_embeddings.weight",
        "roberta.embeddings.token_type_embeddings.weight",
        "roberta.embeddings.LayerNorm.weight",
        "roberta.embeddings.LayerNorm.bias",
    ],
    "fixed_layers": ["roberta.embeddings"],
    "common_layers": ["classifier"],
}
