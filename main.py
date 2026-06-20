import os
import gradio as gr
from dotenv import load_dotenv
from netfree_unstrict_ssl import unstrict_ssl

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

# מייבאים את ה-Workflow ה-Event-Driven שבנינו בקובץ workflow.py
from workflow import AgenticRAGWorkflow

# מייבאים את כלי הציור הוויזואלי של LlamaIndex
from llama_index.utils.workflow import draw_all_possible_flows

# 1. הגדרות ראשוניות
unstrict_ssl()
load_dotenv()

# ==========================================
# 2. הגדרת המודלים
# ==========================================
embed_model = CohereEmbedding(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-english-v3.0",
    input_type="search_document",
)

# הערה: אם חזרת ל-Cohere במקום OpenAI (בגלל התקציב קודם), פשוט תשני פה ל-Cohere
llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

Settings.embed_model = embed_model
Settings.llm = llm

# ==========================================
# 3. חיבור לאינדקס הקיים ב-Pinecone 
# ==========================================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("kiro")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="kiro-steering")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# ==========================================
# 4. אתחול מנוע ה-Workflow (פס הייצור)
# ==========================================
# מקימים את האובייקט ומעבירים לו את האינדקס כדי שיוכל לחפש
app_workflow = AgenticRAGWorkflow(index=index, timeout=60, verbose=True)

# 🎨 יצירת מפת ה-HTML של ה-Event Driven!
try:
    # מצייר את כל המסלולים והאירועים האפשריים מהמחלקה שכתבנו
    draw_all_possible_flows(AgenticRAGWorkflow, filename="workflow_map.html")
    print("✅ מפת ה-Workflow צוירה בהצלחה! נוצר קובץ בשם 'workflow_map.html'.")
except Exception as e:
    print(f"⚠️ שגיאה ביצירת מפת ה-HTML: {e}")

# ==========================================
# 5. ממשק Gradio
# ==========================================
# שימי לב: הפונקציה עכשיו היא אסינכרונית (async) כי Workflow עובד בצורה אסינכרונית
async def chat_with_rag(message, history):
    # במקום query_engine, אנחנו משגרות StartEvent לתוך ה-Workflow שלנו!
    response = await app_workflow.run(query=message)
    return str(response)

demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="🤖 Agentic Docs RAG (Event-Driven)",
    description="שאלי אותי כל שאלה! (המערכת מבוססת כעת על Workflow שמבצע ולידציות חכמות)"
)

if __name__ == "__main__":
    demo.launch()