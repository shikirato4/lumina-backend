from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="Lumina Backend", version="1.0")

# Almacenamiento en memoria temporal del servidor (puedes conectarlo a SQLite o Supabase después)
memory_store = []

class MemoryItem(BaseModel):
    user_id: str
    content: str

@app.get("/")
def health_check():
    return {"status": "online", "bot": "Lumina Backend Active"}

@app.post("/memory/save")
def save_memory(item: MemoryItem):
    memory_store.append({"user_id": item.user_id, "content": item.content})
    # Mantener un límite razonable
    if len(memory_store) > 100:
        memory_store.pop(0)
    return {"success": True, "total_memories": len(memory_store)}

@app.get("/memory/get")
def get_memories(user_id: str):
    user_memories = [m["content"] for m in memory_store if m["user_id"] == user_id]
    return {"user_id": user_id, "memories": user_memories}

@app.delete("/memory/clear")
def clear_memories(user_id: str):
    global memory_store
    memory_store = [m for m in memory_store if m["user_id"] != user_id]
    return {"success": True, "message": "Memorias borradas para el usuario."}