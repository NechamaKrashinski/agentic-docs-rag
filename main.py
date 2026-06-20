import os
import gradio as gr
from dotenv import load_dotenv
from netfree_unstrict_ssl import unstrict_ssl

from llama_index.core import Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 1. הגדרות ראשוניות
unstrict_ssl()
load_dotenv()

# ==========================================
# 2. הגדרת המודלים (Cohere לוקטורים, OpenAI לתשובות)
# ==========================================
embed_model = CohereEmbedding(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-english-v3.0",
    input_type="search_document",
)

llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# הגדרת LlamaIndex להשתמש במודלים שלנו
Settings.embed_model = embed_model
Settings.llm = llm

# ==========================================
# 3. חיבור לאינדקס הקיים ב-Pinecone (בלי לקרוא קבצים מחדש!)
# ==========================================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("kiro")

# מתחברים למאגר הוקטורים הקיים
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="kiro-steering")

# מתחברים לאינדקס מבלי לקרוא מסמכים מחדש
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# ==========================================
# 4. בניית מנוע התשאול 
# ==========================================
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

node_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.2)

response_synthesizer = get_response_synthesizer(
    response_mode="compact" 
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[node_postprocessor],
    response_synthesizer=response_synthesizer,
)

# ==========================================
# 5. ממשק Gradio
# ==========================================
def chat_with_rag(message, history):
    response = query_engine.query(message)
    return str(response)

# בניית הממשק (ללא פרמטר העיצוב שגרם לשגיאה)
demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="🤖 Agentic Docs RAG",
    description="שאלי אותי כל שאלה על החלטות הפיתוח, חוקי הקוד והארכיטקטורה של הפרויקט!"
)

if __name__ == "__main__":
    demo.launch()