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

# # Classification label map
# label_map = {
#     0: 'Science', 1: 'Environment', 2: 'Art & Culture', 3: 'Technology', 4: 'Lifestyle',
#     5: 'Entertainment', 6: 'Religion & Spirituality', 7: 'Business', 8: 'Finance',
#     9: 'Food & Cooking', 10: 'Gaming & Esports', 11: 'Health', 12: 'History',
#     13: 'News & Current Events', 14: 'Law & Governance', 15: 'Education', 16: 'Politics',
#     17: 'Social Media & Digital Culture', 18: 'Sports', 19: 'Travel'
# }

# def predict_category(input_text):
#     classification_inputs = classification_tokenizer(
#         input_text, truncation=True, max_length=512, return_tensors="pt"
#     ).to(device)

#     classification_model.eval()
#     with torch.no_grad():
#         logits = classification_model(**classification_inputs).logits
#         predicted_label = torch.argmax(logits, dim=1).cpu().item()

#     return label_map.get(predicted_label, "Unknown")


def predict_summary(input_text):
    summarization_inputs = summarization_tokenizer(
        input_text, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)

    summarization_model.eval()
    with torch.no_grad():
        summary_ids = summarization_model.generate(
            summarization_inputs["input_ids"], 
            attention_mask=summarization_inputs["attention_mask"],
            max_length=512, 
            num_beams=4, 
            length_penalty=0.8, 
            early_stopping=True
        )

    return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Load models
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Fix UMAP issue with small datasets
umap_model = UMAP(n_neighbors=5, min_dist=0.1, metric='cosine')

# Fix HDBSCAN issue by tuning minimum cluster size
hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')

# Initialize BERTopic with tuned models
topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model)

def generate_labels_with_keybert(text):
    """Generate topic labels using KeyBERT."""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
    return keywords[0][0] if keywords else "Unknown"

def generate_labels_with_bertopic(text):
    """Generate topic labels using BERTopic. Requires multiple texts."""
    example_texts = [
        "Machine learning algorithms improve automation.",
        "Blockchain technology is disrupting finance.",
        "Climate change is a global crisis.",
        "Electric vehicles are becoming more popular.",
        text  # The actual input text
    ]
    
    # Generate embeddings
    embeddings = embedding_model.encode(example_texts, show_progress_bar=False)

    # Fit BERTopic
    topics, _ = topic_model.fit_transform(example_texts, embeddings)

    # Get the topic for the last input text
    topic_id = topics[-1]
    if topic_id == -1:  # If BERTopic fails to detect a topic
        return "Unknown"
    
    return topic_model.get_topic(topic_id)[0][0]

def generate_labels(text):
    """Generate labels using BERTopic, with KeyBERT as a backup."""
    label = generate_labels_with_bertopic(text)
    if not label or label == "Unknown":
        label = generate_labels_with_keybert(text)
    return label.title()