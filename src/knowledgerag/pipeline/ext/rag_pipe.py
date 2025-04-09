from haystack.components.builders import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.utils import Secret

prompt_template = """
    Given these documents, answer the question.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Question: {{query}}
    Answer:
    """

rag_pipe = Pipeline()
rag_pipe.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(model=EMBED_MODEL_ID),
)
rag_pipe.add_component(
    "retriever",
    MilvusEmbeddingRetriever(document_store=document_store, top_k=TOP_K),
)
rag_pipe.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipe.add_component(
    "llm",
    HuggingFaceAPIGenerator(
        api_type="serverless_inference_api",
        api_params={"model": GENERATION_MODEL_ID},
        token=Secret.from_token(HF_TOKEN) if HF_TOKEN else None,
    ),
)
rag_pipe.add_component("answer_builder", AnswerBuilder())
rag_pipe.connect("embedder.embedding", "retriever")
rag_pipe.connect("retriever", "prompt_builder.documents")
rag_pipe.connect("prompt_builder", "llm")
rag_pipe.connect("llm.replies", "answer_builder.replies")
rag_pipe.connect("llm.meta", "answer_builder.meta")
rag_pipe.connect("retriever", "answer_builder.documents")
rag_res = rag_pipe.run(
    {
        "embedder": {"text": QUESTION},
        "prompt_builder": {"query": QUESTION},
        "answer_builder": {"query": QUESTION},
    }
)