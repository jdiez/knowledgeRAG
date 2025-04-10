from transformers import AutoModel, AutoTokenizer

our_hf_tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
our_hf_model = AutoModel.from_pretrained("thenlper/gte-base")


def embedding_text(input_text, hf_tokenizer=our_hf_tokenizer, hf_model=our_hf_model) -> list[float]:
    batch_dict = hf_tokenizer([input_text], max_length=512, padding=True, truncation=True, return_tensors="pt")
    outputs = hf_model(**batch_dict)
    # do a masked mean over the dimension(average_pool)
    last_hidden = outputs.last_hidden_state.masked_fill(~batch_dict["attention_mask"][..., None].bool(), 0.0)
    torch_embeddings_list = last_hidden.sum(dim=1) / batch_dict["attention_mask"].sum(dim=1)[..., None]
    # return only the first element of the batch (since we only passed one sentence to the model)
    # and transform embedding numbers in pytorch into a simple float list
    return torch_embeddings_list[0].tolist()  # dimension of  768
