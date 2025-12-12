from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
import os

from db import get_chat_history, add_message_to_firestore, get_chat_history_as_dict, get_user_chat_list, create_new_chat_doc
from graph import app_graph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EASyariah Chat API")

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
API_KEY_NAME = "X-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
REAL_API_KEY = "EASyariahRDIBCakshon2025!" 

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Fungsi validasi API Key"""
    if api_key_header == REAL_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials (API Key Salah/Tidak Ada)"
        )

# --- 2. MODEL DATA ---
class ChatRequest(BaseModel):
    userId: str
    newMessage: str

class MessageItem(BaseModel):
    role: str
    content: str
    timestamp: str | None = None # Opsional

@app.get("/")
async def root():
    return {"status":"Success", "message": "EA Syariah Chatbot API is running."}

@app.get("/api/v1/chats/{chat_id}", response_model=List[MessageItem])
async def get_history_endpoint(chat_id: str, api_key: str = Depends(get_api_key)):
    history = get_chat_history_as_dict(chat_id)
    return history

# B. Endpoint Kirim Pesan (POST)
@app.post("/api/v1/chats/{chat_id}/message")
async def chat_endpoint(chat_id: str, request: ChatRequest, api_key: str = Depends(get_api_key)):
    try:
        # LOGIKA SAMA SEPERTI SEBELUMNYA (SLIDING WINDOW)
        MAX_HISTORY_LIMIT = 10 
        history_raw = get_chat_history(chat_id) # Ini yang return tuple (role, content)

        if len(history_raw) > MAX_HISTORY_LIMIT:
            recent_history = history_raw[-MAX_HISTORY_LIMIT:]
        else:
            recent_history = history_raw
            
        langchain_messages = []
        langchain_messages.append(SystemMessage(content="Jawab singkat dan padat."))

        for role, content in recent_history:
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
        
        langchain_messages.append(HumanMessage(content=request.newMessage))

        # Simpan User msg
        add_message_to_firestore(chat_id, "user", request.newMessage)

        # Proses AI
        result = app_graph.invoke({"messages": langchain_messages})
        ai_response_content = result["messages"][-1].content

        # Simpan AI msg
        add_message_to_firestore(chat_id, "assistant", ai_response_content)

        return {
            "status": "success",
            "chatId": chat_id,
            "reply": ai_response_content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/users/{user_id}/chats")
async def get_sidebar_list(user_id: str, api_key: str = Depends(get_api_key)):
    """API untuk mengisi list di sidebar"""
    try:
        chat_list = get_user_chat_list(user_id)
        return chat_list
    except Exception as e:
        # Note: Di production, firebase butuh composite index untuk query where+orderBy
        # Cek log error firebase nanti ada link untuk buat index otomatis.
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chats/new")
async def create_chat_endpoint(request: ChatRequest, api_key: str = Depends(get_api_key)):
    """
        Endpoint khusus buat chat pertama/chat baru.
    """
    try:
        # 1. Buat Dokumen Chat Baru di Firestore
        new_chat_id = create_new_chat_doc(request.userId, request.newMessage)
        
        # 2. Setup LangChain Message (Hanya pesan ini saja karena baru)
        langchain_messages = [
            SystemMessage(content="Jawab singkat."),
            HumanMessage(content=request.newMessage)
        ]

        # 3. Simpan Pesan User ke Firestore (di ID yang baru dibuat)
        add_message_to_firestore(new_chat_id, "user", request.newMessage)

        # 4. Proses AI
        result = app_graph.invoke({"messages": langchain_messages})
        ai_response_content = result["messages"][-1].content

        # 5. Simpan Respon AI
        add_message_to_firestore(new_chat_id, "assistant", ai_response_content)

        return {
            "status": "success",
            "chatId": new_chat_id, # KEMBALIKAN ID BARU KE FRONTEND
            "title": request.newMessage[:30], # Kembalikan judul juga biar sidebar update
            "reply": ai_response_content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)