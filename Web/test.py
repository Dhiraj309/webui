import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from keybert import KeyBERT
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import torch

# Paths for models
classification_model_dir = "Model/Text/Classification/"
summarization_model_dir = "Model/Text/Summary-Model/T5-Summary/"

# Load classification model & tokenizer
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_dir)
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_dir, from_tf = True)

# Load summarization model & tokenizer
summarization_tokenizer = T5Tokenizer.from_pretrained(summarization_model_dir)
summarization_model = T5ForConditionalGeneration.from_pretrained(summarization_model_dir, from_tf = True)

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model.to(device)
summarization_model.to(device)

# Classification label map
label_map = {
    0: 'Science', 1: 'Environment', 2: 'Art & Culture', 3: 'Technology', 4: 'Lifestyle',
    5: 'Entertainment', 6: 'Religion & Spirituality', 7: 'Business', 8: 'Finance',
    9: 'Food & Cooking', 10: 'Gaming & Esports', 11: 'Health', 12: 'History',
    13: 'News & Current Events', 14: 'Law & Governance', 15: 'Education', 16: 'Politics',
    17: 'Social Media & Digital Culture', 18: 'Sports', 19: 'Travel'
}

def predict_category(input_text):
    classification_inputs = classification_tokenizer(
        input_text, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)

    classification_model.eval()
    with torch.no_grad():
        logits = classification_model(**classification_inputs).logits
        predicted_label = torch.argmax(logits, dim=1).cpu().item()

    return label_map.get(predicted_label, "Unknown")


# def predict_summary(input_text):
#     summarization_inputs = summarization_tokenizer(
#         input_text, max_length=512, truncation=True, return_tensors="pt"
#     ).to(device)

#     summarization_model.eval()
#     with torch.no_grad():
#         summary_ids = summarization_model.generate(
#             summarization_inputs["input_ids"], 
#             attention_mask=summarization_inputs["attention_mask"],
#             max_length=512, 
#             num_beams=4, 
#             length_penalty=0.8, 
#             early_stopping=True
#         )

#     return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

import torch

def chunk_text(text, tokenizer, chunk_size=512):
    """Splits text into chunks of size <= chunk_size based on tokenization."""
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)  
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

def summarize_chunk(chunk_tokens, tokenizer, model, device, max_summary_length=150):
    """Generates a summary for a single chunk of text tokens."""
    input_tensor = torch.tensor([chunk_tokens]).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_tensor, 
            max_length=max_summary_length,  
            min_length=50,  
            num_beams=5, 
            length_penalty=0.8,  # Adjusted for more concise output
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def clean_summary(summary):
    """Removes repeated phrases and ensures coherence."""
    sentences = summary.split(". ")
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    return ". ".join(unique_sentences) + "."

def predict_summary(input_text):
    summarization_model.eval()

    # Step 1: Tokenize and check length
    tokenized_text = summarization_tokenizer.encode(input_text, truncation=False, add_special_tokens=False)

    # Step 2: Summarize directly if within limit
    if len(tokenized_text) <= 512:
        summary_ids = summarization_model.generate(
            torch.tensor([tokenized_text]).to(device), 
            max_length=200,
            min_length=50,
            num_beams=5,
            length_penalty=0.8,
            early_stopping=True
        )
        return clean_summary(summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    # Step 3: Chunking for longer texts
    text_chunks = chunk_text(input_text, summarization_tokenizer, chunk_size=512)
    chunk_summaries = [summarize_chunk(chunk, summarization_tokenizer, summarization_model, device) for chunk in text_chunks]

    # Step 4: Merge chunk summaries and summarize again
    combined_summary_text = " ".join(chunk_summaries)

    if len(summarization_tokenizer.tokenize(combined_summary_text)) > 512:
        return predict_summary(combined_summary_text)  # Recursive summarization

    return clean_summary(combined_summary_text)