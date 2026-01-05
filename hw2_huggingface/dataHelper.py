# hw2_huggingface/dataHelper.py

import os
import json
import random
import csv
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

## * You can add more helper functions or modify function arguments if needed
def _process_ABSA_data(data, sep_token: str, polarity_to_label: dict):
    texts = []
    labels = []
    for item in data.values():
        term = item['term']
        sentence = item['sentence']
        polarity = item['polarity']
        text = f"{term}{sep_token}{sentence}"
        texts.append(text)
        labels.append(polarity_to_label[polarity])
    return texts, labels

def restaurant(sep_token: str):
    '''Load the ABSA restaurant dataset.'''
    
    # Load train and test JSON files
    train_path = os.path.join("data", "restaurant_sup", "train.json")
    test_path = os.path.join("data", "restaurant_sup", "test.json")
    
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    polarity_to_label = {'negative': 0, 'neutral': 1, 'positive': 2}

    train_texts, train_labels = _process_ABSA_data(train_data, sep_token, polarity_to_label)
    test_texts, test_labels = _process_ABSA_data(test_data, sep_token, polarity_to_label)
    
    return DatasetDict({
        'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}),
        'test': Dataset.from_dict({'text': test_texts, 'label': test_labels})
    })


def laptop(sep_token: str):
    '''Load the ABSA laptop dataset.'''
    
    # Load train and test JSON files
    train_path = os.path.join("data", "SemEval14-laptop", "train.json")
    test_path = os.path.join("data", "SemEval14-laptop", "test.json")
    
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    polarity_to_label = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    train_texts, train_labels = _process_ABSA_data(train_data, sep_token, polarity_to_label)
    test_texts, test_labels = _process_ABSA_data(test_data, sep_token, polarity_to_label)
    
    return DatasetDict({
        'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}),
        'test': Dataset.from_dict({'text': test_texts, 'label': test_labels})
    })


def acl(sep_token: str):
    '''Load the ACL-ARC dataset.'''
    
    # Load train and test JSONL files
    train_path = os.path.join("data", "acl_sup", "train.jsonl")
    test_path = os.path.join("data", "acl_sup", "test.jsonl")
    
    # Map label strings to integers
    label_to_int = {
        'Background': 0,
        'CompareOrContrast': 1,
        'Extends': 2,
        'Future': 3,
        'Motivation': 4,
        'Uses': 5
    }
    
    train_texts = []
    train_labels = []
    with open(train_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            train_texts.append(item['text'])
            train_labels.append(label_to_int[item['label']])
    
    test_texts = []
    test_labels = []
    with open(test_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            test_texts.append(item['text'])
            test_labels.append(label_to_int[item['label']])
    
    return DatasetDict({
        'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}),
        'test': Dataset.from_dict({'text': test_texts, 'label': test_labels})
    })


def agnews(sep_token: str):
    '''Load the AGNews dataset (test set only).'''
    
    # Load CSV file
    csv_path = os.path.join("data", "agnews_sup", "agnews_sup.csv")
    
    texts = []
    labels = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0]) - 1
            description = row[2]
            texts.append(description)
            labels.append(label)
    
    full_dataset = Dataset.from_dict({'text': texts, 'label': labels})
    
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=2025)
    
    return DatasetDict({
        'train': split_dataset['train'],
        'test': split_dataset['test']
    })


_DATASET_MAPPING = {
    'restaurant': restaurant,
    'laptop': laptop,
    'acl': acl,
    'agnews': agnews
}


def get_fs(dataset_name: str, sep_token: str, sample_size: int):
    '''
    Get few-shot dataset. Call this function inside `get_dataset` if needed.
    dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    
    # Strip _fs suffix if present
    base_name = dataset_name.replace('_fs', '')
    
    # Get the base dataset
    if base_name not in _DATASET_MAPPING:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    base_dataset = _DATASET_MAPPING[base_name](sep_token)
    
    # Sample few-shot data
    train_dataset = base_dataset['train']
    test_dataset = base_dataset['test']
    
    train_indices = list(range(len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    
    sampled_train_indices = train_indices[:sample_size]
    sampled_test_indices = test_indices[:sample_size]
    
    train_texts = [train_dataset[i]['text'] for i in sampled_train_indices]
    train_labels = [train_dataset[i]['label'] for i in sampled_train_indices]
    
    test_texts = [test_dataset[i]['text'] for i in sampled_test_indices]
    test_labels = [test_dataset[i]['label'] for i in sampled_test_indices]
    
    return DatasetDict({
        'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}),
        'test': Dataset.from_dict({'text': test_texts, 'label': test_labels})
    })

## ! DO NOT change the function name or arguments
def get_dataset(dataset_name: str | list[str], sep_token: str) -> DatasetDict:
    '''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
    dataset = None

    # parse dataset name
    if ',' in dataset_name:
        dataset_name = dataset_name.replace('[', '').replace(']', '').split(',')

    if isinstance(dataset_name, str):
        # Handle few-shot datasets
        if dataset_name.endswith('_fs'):
            base_name = dataset_name.replace('_fs', '')
            if base_name not in _DATASET_MAPPING:
                raise ValueError(f"Unknown few-shot dataset: {dataset_name}")
            dataset = get_fs(base_name, sep_token, sample_size=32)
        # Handle regular datasets
        elif dataset_name.endswith('_sup'):
            base_name = dataset_name.replace('_sup', '')
            if base_name not in _DATASET_MAPPING:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            dataset = _DATASET_MAPPING[base_name](sep_token)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_name}")

    elif isinstance(dataset_name, list):
        # Handle aggregation
        datasets = []
        label_offset = 0
        
        for name in dataset_name:
            # Load each dataset
            if name.endswith('_fs'):
                base_name = name.replace('_fs', '')
                ds = get_fs(base_name, sep_token, sample_size=32)
            elif name.endswith('_sup'):
                base_name = name.replace('_sup', '')
                if base_name not in _DATASET_MAPPING:
                    raise ValueError(f"Unknown dataset: {name}")
                ds = _DATASET_MAPPING[base_name](sep_token)
            else:
                raise ValueError(f"Unsupported dataset format: {name}")
            
            # Get number of unique labels in this dataset
            num_labels = len(set(ds['train']['label']))
            
            # Re-label: add offset to labels
            def relabel_labels(examples):
                return {'label': [l + label_offset for l in examples['label']]}
            
            ds_train = ds['train'].map(relabel_labels, batched=True)
            ds_test = ds['test'].map(relabel_labels, batched=True)
            
            datasets.append({
                'train': ds_train,
                'test': ds_test
            })
            
            # Update offset for next dataset
            label_offset += num_labels

        train_datasets = [d['train'] for d in datasets]
        test_datasets = [d['test'] for d in datasets]
        
        combined_train = concatenate_datasets(train_datasets)
        combined_test = concatenate_datasets(test_datasets)
        
        dataset = DatasetDict({
            'train': combined_train,
            'test': combined_test
        })

    else:
        raise ValueError("Unsupported dataset!")

    return dataset