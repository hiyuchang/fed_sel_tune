import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import datasets
datasets.utils.logging.set_verbosity_error()
datasets.utils.disable_progress_bar()
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from path import Path
MAIN_DIR = Path(__file__).parent.parent.parent.parent.abspath()
DATASETS_DIR = MAIN_DIR / "datasets"
MODEL_DIR = MAIN_DIR / "models"
MODEL_PATH = {
    "xlm-roberta-large": MODEL_DIR / "xlm-roberta-large",
    "roberta-large": MODEL_DIR / "roberta-large",
    "xlm-roberta-base": MODEL_DIR / "xlm-roberta-base",
    "roberta-base": MODEL_DIR / "roberta-base",
}
LANGS = ["en", "de", "es", "fr", "ru"]


def partition_multi_lang_data(
    model: str, n_clients: int = 0, max_length: int = 128, only_size: bool = False
):
    """Training set and test set are split as (0.8, 0.2)"""
    raw_dataset = load_dataset(DATASETS_DIR / "xglue/xglue.py", "nc")
    # print("successfully load raw dataset")

    # use valiadation data as training data
    multi_train_sets = []
    for lang in LANGS:
        train = raw_dataset["validation.{}".format(lang)]
        train = train.map(
            lambda example: {
                "label": example["news_category"],
                "text": example["news_title"] + " " + example["news_body"],
            },
            remove_columns=["news_title", "news_body", "news_category"],
        )
        multi_train_sets.append(train)

    # allocate datasets to clients
    net_dataidx_map, net_lang_indices = {}, {}
    n_clients_per_language = max(1, n_clients // len(LANGS))

    for i, lang in enumerate(LANGS):
        lang_dataset = multi_train_sets[i]
        lang_indices = [idx for idx in range(len(lang_dataset))]
        np.random.shuffle(lang_indices)
        lang_indices_split = np.array_split(lang_indices, n_clients_per_language)

        # allocate each chunk to a client
        for client_id, indices_chunk in enumerate(lang_indices_split):
            actual_client_id = client_id + i * n_clients_per_language
            net_dataidx_map[actual_client_id] = indices_chunk.tolist()
            net_lang_indices[actual_client_id] = i

    if only_size:
        return net_dataidx_map

    # the tokenizer converts the raw text into input IDs and attention masks
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model], max_length=max_length)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding=True, truncation=True, max_length=max_length
        )

    # trainset: client_id
    trainset_dict = {}
    for client_id in net_dataidx_map.keys():
        lang = net_lang_indices[client_id]
        pre_trainset = multi_train_sets[lang].select(net_dataidx_map[client_id])
        tokenized_trainset = pre_trainset.map(tokenize_function, batched=True)
        trainset = tokenized_trainset.remove_columns(["text"])
        trainset.set_format("torch")
        trainset_dict[client_id] = trainset

    # testset: unique
    multi_test_sets = []
    for lang in LANGS:
        test = raw_dataset["test.{}".format(lang)]
        test = test.map(
            lambda example: {
                "label": example["news_category"],
                "text": example["news_title"] + " " + example["news_body"],
            },
            remove_columns=["news_title", "news_body", "news_category"],
        )
        multi_test_sets.append(test)
    multi_test_sets = concatenate_datasets(multi_test_sets)
    tokenized_testset = multi_test_sets.map(tokenize_function, batched=True)
    testset = tokenized_testset.remove_columns(["text"])
    testset.set_format("torch")  # each lang has 2000 samples

    return trainset_dict, testset, net_dataidx_map
