import uuid
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, firestore
import os

cred_path = "serviceAccountKey.json"
if not os.path.exists(cred_path):
    print(f"ERROR: File {cred_path} tidak ditemukan! Pastikan lokasinya benar.")
else:
    print(f"SUCCESS: File {cred_path} ditemukan.")

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

def get_chat_history(chat_id: str):
    """Mengambil history chat dan memformatnya untuk LangGraph"""
    doc_ref = db.collection("chats").document(chat_id)
    doc = doc_ref.get()
    
    messages = []
    if doc.exists:
        data = doc.to_dict()
        # Mengambil array 'messages' dari Firestore
        firebase_msgs = data.get("messages", [])
        for msg in firebase_msgs:
            # LangGraph butuh format (role, content)
            messages.append((msg['role'], msg['content']))
            
    return messages

def get_chat_history_as_dict(chat_id: str):
    """Mengambil history lengkap dalam bentuk Dictionary untuk Frontend"""
    doc_ref = db.collection("chats").document(chat_id)
    doc = doc_ref.get()
    
    if doc.exists:
        data = doc.to_dict()
        # Mengembalikan list of object: [{'role': 'user', 'content': '...'}, ...]
        return data.get("messages", [])
    return []

def add_message_to_firestore(chat_id: str, role: str, content: str):
    """Menyimpan pesan baru ke array di Firestore"""
    doc_ref = db.collection("chats").document(chat_id)
    
    new_message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    }
    
    # Jika dokumen belum ada, buat baru. Jika ada, update array-nya.
    if not doc_ref.get().exists:
        doc_ref.set({
            "userId": "user_default", # Bisa diganti nanti
            "createdAt": datetime.now(),
            "messages": [new_message]
        })
    else:
        # ArrayUnion menambahkan item ke array yang sudah ada
        doc_ref.update({
            "messages": firestore.ArrayUnion([new_message]),
            "updatedAt": datetime.now()
        })
        
def create_new_chat_doc(user_id: str, initial_message: str):
    """Membuat dokumen chat baru dan mengembalikan ID-nya"""
    new_chat_id = str(uuid.uuid4()) # Generate ID unik
    
    doc_ref = db.collection("chats").document(new_chat_id)
    
    # Gunakan 30 karakter pertama pesan user sebagai Judul (Title)
    title_preview = initial_message[:30] + "..." if len(initial_message) > 30 else initial_message
    
    doc_ref.set({
        "userId": user_id,
        "title": title_preview, # Field ini untuk ditampilkan di Sidebar
        "createdAt": datetime.now(),
        "updatedAt": datetime.now(),
        "messages": [] # Masih kosong, nanti diisi flow selanjutnya
    })
    
    return new_chat_id

def get_user_chat_list(user_id: str):
    """Mengambil daftar chat untuk sidebar (Hanya ID dan Title)"""
    # Query: Ambil semua chat milik user_id, urutkan dari yang terbaru
    chats_ref = db.collection("chats")
    query = chats_ref.where("userId", "==", user_id).order_by("updatedAt", direction=firestore.Query.DESCENDING)
    
    results = []
    for doc in query.stream():
        data = doc.to_dict()
        results.append({
            "chatId": doc.id,
            "title": data.get("title", "Percakapan Baru"),
            "updatedAt": data.get("updatedAt")
        })
    return results