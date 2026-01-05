# hw2_huggingface/eval.py

import torch
import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score

def inference(model, tokenizer, inputs, max_length=64, max_batch_size=16, verbose=False):
    '''Generate predictions from the model given inputs.'''
    model.eval()
    device = next(model.parameters()).device
    results = []
    with torch.no_grad():
        iterator = range(0, len(inputs), max_batch_size) if not verbose else tqdm.tqdm(range(0, len(inputs), max_batch_size))
        for i in iterator:
            batch_inputs = inputs[i:min(i + max_batch_size, len(inputs))]
            # Tokenize inputs
            encoded_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = encoded_inputs["input_ids"].to(device)
            attention_mask = encoded_inputs["attention_mask"].to(device)
            
            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                num_beams=2,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            
            # Extract only the newly generated tokens (excluding input)
            input_length = input_ids.shape[1]
            generated_ids = generated_ids[:, input_length:]
            
            # Decode predictions
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend(predictions)
    return results


def compute_bleu(references, candidates):
    ## TODO: Compute average BLEU score for reference and candidate pairs.
    bleu_scores = []
    for ref, cand in zip(references, candidates):
        # Tokenize reference and candidate
        ref_tokens = ref.split()
        cand_tokens = cand.split()
        # Compute BLEU score without smoothing
        bleu = sentence_bleu([ref_tokens], cand_tokens)
        bleu_scores.append(bleu)
    bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    return bleu_score


def compute_rouge(references, candidates):
    ## TODO: Compute average ROUGE-L score for reference and candidate pairs.
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    rouge_l_score = scores['rouge-l']['f']
    return rouge_l_score


def compute_bertscore(references, candidates, model="facebook/bart-large"):
    ## TODO: Compute average BERTScore for reference and candidate pairs.
    P, R, F1 = bert_score(candidates, references, model_type=model, verbose=False)
    bertscore = F1.mean().item()
    return bertscore


def evaluate_model(model_name, dataset_name, split="test", verbose=False):
    '''Evaluate the model on the given dataset and split.'''

    ## TODO: Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Move model to appropriate device
    device = "cuda"
    model = model.to(device)
    model.eval()

    ## TODO: Load dataset
    dataset = load_dataset(dataset_name, split=split)
    if "summary" in dataset.column_names:
        references = dataset["summary"]
    elif "target" in dataset.column_names:
        references = dataset["target"]
    elif "label" in dataset.column_names:
        # For classification tasks, labels might be the references
        references = [str(label) for label in dataset["label"]]
    else:
        raise ValueError(f"Dataset must have 'summary', 'target', or 'label' field. Available fields: {dataset.column_names}")

    inputs = dataset["text"] if "text" in dataset.column_names else dataset["dialogue"]
    candidates = inference(model, tokenizer, list(inputs), verbose=verbose)

    valid_references = []
    valid_candidates = []
    for ref, cand in zip(references, candidates):
        if cand != "":
            valid_references.append(ref)
            valid_candidates.append(cand)

    # Compute evaluation metrics
    bleu = compute_bleu(valid_references, valid_candidates)
    print("BLEU:", bleu)
    rouge_l = compute_rouge(valid_references, valid_candidates)
    print("ROUGE-L:", rouge_l)
    bertscore = compute_bertscore(valid_references, valid_candidates)
    print("BERTScore:", bertscore)
 
    return {"BLEU": bleu, "ROUGE-L": rouge_l, "BERTScore": bertscore}


if __name__ == "__main__":
    model_name = "LLaMA-Factory/output/qwen1_5_lora"
    dataset_name = "knkarthick/samsum"

    results = evaluate_model(model_name, dataset_name, verbose=True)
    print("Evaluation Results:", results)
