from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Event, 
    Workflow, 
    step, 
    StartEvent, 
    StopEvent, 
    Context
)

# ==========================================
# 1. הגדרת האירועים (Events)
# ==========================================

class RetrieveEvent(Event):
    """אירוע לביצוע חיפוש - נשלח אחרי שהשאלה נמצאה תקינה"""
    query: str

class EvaluateEvent(Event):
    """אירוע להערכת תוצאות - נשלח אחרי שה-Retriever חזר עם מידע"""
    query: str
    nodes: List[NodeWithScore] # רשימת הפסקאות (Nodes) והניקוד שלהן

class SynthesizeEvent(Event):
    """אירוע ליצירת תשובה - נשלח אחרי שאישרנו שהתוצאות רלוונטיות"""
    query: str
    nodes: List[NodeWithScore]


# ==========================================
# 2. מעטפת ה-Workflow (פס הייצור שלנו)
# ==========================================

class AgenticRAGWorkflow(Workflow):
    
    @step
    async def validate_input(self, ctx: Context, ev: StartEvent) -> RetrieveEvent | StopEvent:
        """
        תחנה 1: קבלת השאלה, ולידציה והכנת הזיכרון הגלובלי.
        """
        query = ev.get("query")
        
        if not query:
            return StopEvent(result="נראה שלא הוקלדה שאלה. אנא נסי שוב.")
            
        if len(query.strip()) < 3:
            return StopEvent(result="השאלה קצרה מדי, אשמח אם תפרטי קצת יותר כדי שאוכל למצוא תשובה מדויקת.")
            
        # ==================================
        # תיקון ניהול מצב - הגרסה העדכנית עם ctx.store
        # ==================================
        await ctx.store.set("original_query", query)
        await ctx.store.set("retries", 0)
        
        print(f"[תחנה 1] הולידציה עברה בהצלחה. השאלה: '{query}'")
        return RetrieveEvent(query=query)

    def __init__(self, index, *args, **kwargs):
        """
        פונקציית אתחול: מופעלת פעם אחת כשהמערכת עולה.
        אנחנו מקבלים את האינדקס של Pinecone ומגדירים ממנו את שולף המידע.
        """
        super().__init__(*args, **kwargs)
        # ניצור Retriever שמביא את 5 התוצאות הכי קרובות לשאלה
        self.retriever = index.as_retriever(similarity_top_k=5)

    @step
    async def retrieve_data(self, ctx: Context, ev: RetrieveEvent) -> EvaluateEvent | StopEvent:
        """
        תחנה 2: חיפוש וקטורי במסד הנתונים.
        """
        query = ev.query
        print(f"[תחנה 2] מחפשת ב-Pinecone מידע עבור: '{query}'...")
        
        # שליפת הנתונים בפועל מול Pinecone
        # אנחנו משתמשים ב-aretrieve שזו הגרסה האסינכרונית (המהירה) של הפעולה
        nodes = await self.retriever.aretrieve(query)
        
        # ולידציה: האם מצאנו משהו בכלל?
        if not nodes:
            # אם הרשימה ריקה, אין טעם להמשיך ל-LLM. עוצרים כאן.
            return StopEvent(result="חיפשתי במסמכים, אבל לא מצאתי מידע שרלוונטי לשאלה שלך.")
            
        print(f"[תחנה 2] מעולה! מצאתי {len(nodes)} פסקאות מידע. מעבירה לבדיקת איכות...")
        
        # אורזים את השאלה ואת התוצאות במעטפת ומשגרים לתחנה 3
        return EvaluateEvent(query=query, nodes=nodes)

    @step
    async def evaluate_results(self, ctx: Context, ev: EvaluateEvent) -> SynthesizeEvent | RetrieveEvent | StopEvent:
        """
        תחנה 3: הערכת התוצאות (Evaluation).
        """
        query = ev.query
        nodes = ev.nodes

        best_score = nodes[0].score if nodes else 0.0
        print(f"[תחנה 3] בודקת איכות. ציון ההתאמה הגבוה ביותר הוא: {best_score:.2f}")

        if best_score < 0.3:
            # ==================================
            # תיקון קריאה מה-State עם ctx.store
            # ==================================
            current_retries = await ctx.store.get("retries")
            
            # למקרה שהמשתנה עוד לא קיים
            if current_retries is None:
                current_retries = 0
            
            if current_retries < 1: 
                next_retry_count = current_retries + 1
                await ctx.store.set("retries", next_retry_count)
                print(f"[תחנה 3] איכות נמוכה. מנסה לחפש שוב... (ניסיון {next_retry_count})")
                
                modified_query = query + " פרטים נוספים"
                return RetrieveEvent(query=modified_query)
            else:
                print("[תחנה 3] מיצינו את הניסיונות. עוצרים.")
                return StopEvent(result="מצטערת, חיפשתי שוב ושוב אך לא מצאתי תשובה ברמת ודאות גבוהה מספיק במסמכים שלנו.")

        print("[תחנה 3] התוצאות איכותיות. עוברים לניסוח תשובה...")
        return SynthesizeEvent(query=query, nodes=nodes)
    
    
    @step
    async def synthesize_response(self, ctx: Context, ev: SynthesizeEvent) -> StopEvent:
        """
        תחנה 4: המלחים (Synthesizer).
        ה-LLM של Cohere מקבל את השאלה והפסקאות הרלוונטיות, ומנסח תשובה.
        """
        from llama_index.core import get_response_synthesizer
        
        print("[תחנה 4] מנסחת תשובה סופית בעזרת ה-LLM...")
        
        # אנחנו קוראות לפונקציה המובנית של LlamaIndex שלוקחת את ה-LLM שהגדרנו במערכת
        response_synthesizer = get_response_synthesizer(response_mode="compact")
        
        # שולחים הכל ל-LLM. שימי לב לשימוש ב-asynthesize (אסינכרוני)
        response = await response_synthesizer.asynthesize(
            query=ev.query,
            nodes=ev.nodes,
        )
        
        print("[תחנה 4] התשובה מוכנה! מחזירה למשתמש.")
        
        # משגרות את אירוע הסיום עם התשובה המוצלחת שתגיע עד ל-Gradio
        return StopEvent(result=str(response))