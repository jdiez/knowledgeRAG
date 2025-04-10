import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Get token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Sentences for embedding
sentences = ["This is an example sentence", "Each sentence is converted"]

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Get token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Apply mean pooling to get sentence embeddings
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

# Normalize the embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Output the embeddings
print(sentence_embeddings)
