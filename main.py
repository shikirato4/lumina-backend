from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import base64

app = FastAPI(title="Lumina Backend", version="1.0")

memory_store = []
TELEGRAM_BOT_TOKEN = "8882482341:AAFA1_ueRX8mmwFDBwySPPW8i5xmff-rYTU"

class MemoryItem(BaseModel):
    user_id: str
    content: str

class VisionItem(BaseModel):
    user_id: str
    chat_id: str
    file_id: str
    gemini_key: str
    caption: str = "Analiza este archivo con detalle."
    mime_type: str = "image/jpeg"

@app.get("/")
def health_check():
    return {"status": "online", "bot": "Lumina Backend Active"}

@app.post("/memory/save")
def save_memory(item: MemoryItem):
    memory_store.append({"user_id": item.user_id, "content": item.content})
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

@app.post("/vision")
def analyze_vision(item: VisionItem):
    try:
        # 1. Obtener la ruta del archivo en Telegram
        tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={item.file_id}"
        r = requests.get(tg_url)
        res_json = r.json()
        if not res_json.get("ok"):
            return {"success": False, "error": "No se pudo obtener el archivo de Telegram"}
        
        file_path = res_json["result"]["file_path"]
        
        # 2. Descargar el archivo desde Telegram
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        file_response = requests.get(file_url)
        if file_response.status_code != 200:
            return {"success": False, "error": "No se pudo descargar el archivo"}
        
        # 3. Codificar en base64
        encoded_data = base64.b64encode(file_response.content).decode("utf-8")
        
        # 4. Llamar a Gemini 1.5 Flash (Soporta imágenes, videos y documentos)
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": item.gemini_key
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": item.caption},
                        {
                            "inline_data": {
                                "mime_type": item.mime_type,
                                "data": encoded_data
                            }
                        }
                    ]
                }
            ]
        }
        
        gem_resp = requests.post(gemini_url, headers=headers, json=payload)
        if gem_resp.status_code != 200:
            return {"success": False, "error": f"Error en Gemini API: {gem_resp.text}"}
            
        gem_data = gem_resp.json()
        answer = gem_data["candidates"][0]["content"]["parts"][0]["text"]
        return {"success": True, "analysis": answer}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
