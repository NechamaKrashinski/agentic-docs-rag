
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

import os
import gradio as gr
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

# ייבוא המודל של OpenAI
from llama_index.llms.openai import OpenAI

# ==========================================
# 1. הגדרת המודלים (Cohere לוקטורים, OpenAI לתשובות)
# ==========================================

# נגדיר את מודל השפה שינסח את התשובות (GPT-4o-mini הוא מהיר וזול)
llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# חשוב: נגדיר ל-LlamaIndex את המודלים כברירת מחדל לכל המערכת
# embed_model - המשתנה שכבר יצרת קודם עם Cohere
Settings.embed_model = embed_model 
Settings.llm = llm

# ==========================================
# 2. בניית מנוע התשאול (Retriever -> Postprocessor -> Synthesizer)
# ==========================================

# א. שולף המידע (Retriever) - מביא את 5 הפסקאות הרלוונטיות ביותר מ-Pinecone
retriever = VectorIndexRetriever(
    index=index, # משתנה האינדקס שיצרת קודם לכן
    similarity_top_k=5,
)

# ב. מסנן התוצאות (Postprocessor) - זורק תוצאות שרמת ההתאמה שלהן נמוכה
node_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)

# ג. המלחים (Synthesizer) - לוקח את התוצאות ושולח ל-OpenAI
response_synthesizer = get_response_synthesizer(
    response_mode="compact" 
)

# חיבור כל החלקים למנוע אחד שיודע לקבל שאלה ולהחזיר תשובה
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[node_postprocessor],
    response_synthesizer=response_synthesizer,
)

# ==========================================
# 3. יצירת ממשק המשתמש עם Gradio
# ==========================================

# פונקציה שעוטפת את מנוע השאילתות עבור ממשק הצ'אט
def chat_with_rag(message, history):
    response = query_engine.query(message)
    return str(response)

# בניית חלון הצ'אט
demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="🤖 Agentic Docs RAG (Powered by OpenAI)",
    description="שאלי אותי כל שאלה על החלטות הפיתוח וחוקי הקוד של הפרויקט!",
)

# הפעלת האפליקציה המקומית
if __name__ == "__main__":
    demo.launch()