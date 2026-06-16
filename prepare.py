
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import os
from dotenv import load_dotenv
from netfree_unstrict_ssl import unstrict_ssl
unstrict_ssl()
load_dotenv()

# Loading
reader = SimpleDirectoryReader(input_dir="docs")
documents = reader.load_data()

print(len(documents))


#Chunking
node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)

nodes = node_parser.get_nodes_from_documents(
    documents=documents, show_progress=True
)

print(len(nodes))


#Embedding

COHERE_API_KEY=os.getenv("COHERE_API_KEY")

from llama_index.embeddings.cohere import CohereEmbedding

embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_document",
)

# texts = [node.get_content() for node in nodes]
# embeddings = embed_model.get_text_embedding_batch(texts)

# print(len(embeddings))
# print(embeddings[:5])


#Indexing and Saving

#pinecone instance - Connection
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("kiro")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index,namespace="kiro-steering")

#Storage Context 
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)

#Index - VectorStoreIndex

