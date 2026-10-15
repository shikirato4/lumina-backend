import base64
import os
import sqlite3
import threading
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import google.generativeai as genai
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Lumina Backend", version="3.5-fixed")

# ---------------- CONFIGURACIÓN ----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
ADMIN_ID = "6363412200"

CHAT_MODEL = "gemma-4-31b-it"
VISION_AND_PROMPT_MODEL = "gemini-3.8-flash"

MAX_STORED_MEMORIES_PER_USER = 100

DB_PATH = os.getenv(
    "LUMINA_DB_PATH",
    os.path.join(os.path.dirname(__file__), "lumina.db")
)

DEFAULT_USERS = {
    "6363412200": "shikirato",
    "8389931379": "regina",
    "1057724787": "josho",
}

SESSION = requests.Session()

chat_context: dict[str, list[dict[str, str]]] = {}
chat_lock = threading.RLock()
_db_lock = threading.RLock()
_job_lock = threading.RLock()

jobs: dict[str, dict[str, Any]] = {}
job_request_index: dict[str, str] = {}

executor = ThreadPoolExecutor(
    max_workers=3,
    thread_name_prefix="lumina-job"
)


# ---------------- BASE DE DATOS SQLITE ----------------
def init_db():
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            for uid, name in DEFAULT_USERS.items():
                cursor.execute("INSERT OR REPLACE INTO users (user_id, name) VALUES (?, ?)", (uid, name))
            conn.commit()
        conn.close()

init_db()


# ---------------- MODELOS ----------------
class MemoryItem(BaseModel):
    user_id: str
    content: str
    request_id: str = ""


class ChatItem(BaseModel):
    user_id: str
    chat_id: str
    message: str
    gemini_keys: list[str] = Field(default_factory=list)
    model: str = CHAT_MODEL


class VisionItem(BaseModel):
    user_id: str
    chat_id: str
    file_id: str
    gemini_keys: list[str] = Field(default_factory=list)
    caption: str = "Analiza este archivo con detalle."
    mime_type: str = "image/jpeg"
    request_id: str = ""


class ImageItem(BaseModel):
    chat_id: str
    prompt: str
    gemini_keys: list[str] = Field(default_factory=list)
    request_id: str = ""


@app.on_event("startup")
def startup_event():
    if TELEGRAM_BOT_TOKEN:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            SESSION.post(
                url,
                json={
                    "chat_id": ADMIN_ID,
                    "text": "🟢 ¡Backend Lumina 3.5 en línea y listo!",
                },
                timeout=10
            )
        except Exception as e:
            print(f"Error startup notification: {e}")


@app.get("/")
def health_check():
    return {"status": "online", "bot": "Lumina Backend Active"}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    with _job_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job no encontrado")
        return job


# ---------------- ENDPOINT IMAGEN (ASÍNCRONO) ----------------
@app.post("/image")
def create_image_job(item: ImageItem):
    with _job_lock:
        if item.request_id and item.request_id in job_request_index:
            existing_id = job_request_index[item.request_id]
            return {"success": True, "job_id": existing_id}

        job_id = uuid.uuid4().hex
        if item.request_id:
            job_request_index[item.request_id] = job_id

        jobs[job_id] = {
            "state": "processing",
            "success": False,
            "error": None
        }

    def run_image_task():
        try:
            enhanced_prompt = item.prompt
            keys_to_try = item.gemini_keys if item.gemini_keys else []

            # Optimización de prompt con gemini-3.8-flash
            for key in keys_to_try:
                if key and len(key) > 10:
                    try:
                        genai.configure(api_key=key)
                        model = genai.GenerativeModel(VISION_AND_PROMPT_MODEL)
                        opt_res = model.generate_content(
                            f"Translate and optimize this prompt into English for FLUX AI to generate an ultra-realistic, highly detailed, cinematic masterpiece image, ensuring any requested text inside quotes remains exactly as specified: {item.prompt}"
                        )
                        if opt_res and hasattr(opt_res, "text") and opt_res.text:
                            enhanced_prompt = opt_res.text.strip()
                        break
                    except Exception:
                        continue

            image_bytes = None
            engine_used = ""

            # 1. Cloudflare Workers AI
            if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:
                try:
                    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell"
                    cf_headers = {
                        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                        "Content-Type": "application/json",
                    }
                    cf_payload = {"prompt": enhanced_prompt}
                    cf_resp = SESSION.post(cf_url, headers=cf_headers, json=cf_payload, timeout=30)
                    
                    print(f"CF Status Code: {cf_resp.status_code}")
                    print(f"CF Response Text: {cf_resp.text}")

                    if cf_resp.status_code == 200:
                        res_json = cf_resp.json()
                        if "result" in res_json and res_json["result"] and "image" in res_json["result"]:
                            image_bytes = base64.b64decode(res_json["result"]["image"])
                            engine_used = "Cloudflare FLUX.1"
                except Exception as e:
                    print(f"Cloudflare exception: {e}")

            # 2. Respaldo Pollinations.ai
            if not image_bytes:
                try:
                    encoded_prompt = urllib.parse.quote(enhanced_prompt)
                    image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&nologo=true&model=flux"
                    img_resp = SESSION.get(image_url, timeout=55)
                    if img_resp.status_code == 200:
                        image_bytes = img_resp.content
                        engine_used = "Pollinations FLUX"
                except Exception as e:
                    print(f"Pollinations error: {e}")

            if not image_bytes:
                with _job_lock:
                    jobs[job_id] = {
                        "state": "completed",
                        "success": False,
                        "error": "Servidor lleno o generador saturado, espera un minuto."
                    }
                return

            # Enviar a Telegram
            tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {"photo": ("image.png", image_bytes, "image/png")}
            data = {
                "chat_id": item.chat_id,
                "caption": f"✨ *Prompt:* {item.prompt}\n⚡ *Motor:* {engine_used}",
                "parse_mode": "Markdown",
            }
            tg_resp = SESSION.post(tg_url, data=data, files=files, timeout=45)

            if tg_resp.status_code == 200:
                with _job_lock:
                    jobs[job_id] = {
                        "state": "completed",
                        "success": True,
                        "error": None
                    }
            else:
                with _job_lock:
                    jobs[job_id] = {
                        "state": "completed",
                        "success": False,
                        "error": "Error al enviar la imagen generada a Telegram."
                    }
        except Exception as e:
            with _job_lock:
                jobs[job_id] = {
                    "state": "completed",
                    "success": False,
                    "error": f"Excepción interna: {str(e)}"
                }

    executor.submit(run_image_task)
    return {"success": True, "job_id": job_id}


# ---------------- ENDPOINT VISIÓN (ASÍNCRONO) ----------------
@app.post("/vision")
def create_vision_job(item: VisionItem):
    with _job_lock:
        if item.request_id and item.request_id in job_request_index:
            existing_id = job_request_index[item.request_id]
            return {"success": True, "job_id": existing_id}

        job_id = uuid.uuid4().hex
        if item.request_id:
            job_request_index[item.request_id] = job_id

        jobs[job_id] = {
            "state": "processing",
            "success": False,
            "error": None,
            "analysis": None
        }

    def run_vision_task():
        try:
            tg_file_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={item.file_id}"
            r = SESSION.get(tg_file_url, timeout=15)
            res_json = r.json()
            if not res_json.get("ok"):
                with _job_lock:
                    jobs[job_id] = {
                        "state": "completed",
                        "success": False,
                        "error": f"Telegram error: {res_json.get('description', 'Desconocido')}"
                    }
                return

            file_path = res_json["result"]["file_path"]
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
            file_resp = SESSION.get(file_url, timeout=30)
            if file_resp.status_code != 200:
                with _job_lock:
                    jobs[job_id] = {
                        "state": "completed",
                        "success": False,
                        "error": "No se pudo descargar el archivo multimedia de Telegram."
                    }
                return

            encoded_data = base64.b64encode(file_resp.content).decode("utf-8")
            session_id = item.chat_id

            with chat_lock:
                if session_id not in chat_context:
                    chat_context[session_id] = []
                raw_history = chat_context[session_id][-10:]
                while raw_history and raw_history[0]["role"] != "user":
                    raw_history.pop(0)

            contents_payload = []
            expected_role = "user"
            with chat_lock:
                for msg in raw_history:
                    if msg["role"] == expected_role:
                        contents_payload.append({"role": msg["role"], "parts": [{"text": msg["text"]}]})
                        expected_role = "model" if expected_role == "user" else "user"

            contents_payload.append({
                "role": "user",
                "parts": [
                    {"text": item.caption},
                    {"inline_data": {"mime_type": item.mime_type, "data": encoded_data}},
                ],
            })

            keys_to_try = item.gemini_keys if item.gemini_keys else []
            if not keys_to_try:
                with _job_lock:
                    jobs[job_id] = {
                        "state": "completed",
                        "success": False,
                        "error": "No se proporcionaron llaves de Gemini para visión."
                    }
                return

            headers = {"Content-Type": "application/json"}
            payload = {"contents": contents_payload}
            last_error = ""
            analysis_text = ""

            for key in keys_to_try:
                if not key or len(key) < 10:
                    continue
                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{VISION_AND_PROMPT_MODEL}:generateContent?key={key}"
                gem_resp = SESSION.post(gemini_url, headers=headers, json=payload, timeout=60)
                if gem_resp.status_code == 200:
                    gem_data = gem_resp.json()
                    if "candidates" in gem_data and len(gem_data["candidates"]) > 0:
                        analysis_text = gem_data["candidates"][0]["content"]["parts"][0].get("text", "").strip()
                        if analysis_text:
                            break
                else:
                    last_error = f"HTTP {gem_resp.status_code}: {gem_resp.text}"

            if analysis_text:
                with chat_lock:
                    chat_context[session_id].append({"role": "user", "text": item.caption})
                    chat_context[session_id].append({"role": "model", "text": analysis_text})
                    if len(chat_context[session_id]) > 20:
                        chat_context[session_id] = chat_context[session_id][-20:]

                with _job_lock:
                    jobs[job_id] = {
                        "state": "completed",
                        "success": True,
                        "analysis": analysis_text,
                        "error": None
                    }
            else:
                with _job_lock:
                    jobs[job_id] = {
                        "state": "completed",
                        "success": False,
                        "error": f"Todas las llaves de visión fallaron. {last_error}"
                    }
        except Exception as e:
            with _job_lock:
                jobs[job_id] = {
                    "state": "completed",
                    "success": False,
                    "error": f"Excepción en visión: {str(e)}"
                }

    executor.submit(run_vision_task)
    return {"success": True, "job_id": job_id}


# ---------------- ENDPOINT CHAT ----------------
@app.post("/chat")
def handle_chat_backend(item: ChatItem):
    try:
        session_id = item.chat_id
        is_new_or_reset = False

        with chat_lock:
            if session_id not in chat_context:
                chat_context[session_id] = []
                is_new_or_reset = True
            elif len(chat_context[session_id]) == 0:
                is_new_or_reset = True

            raw_history = chat_context[session_id][-10:]
            while raw_history and raw_history[0]["role"] != "user":
                raw_history.pop(0)

            contents_payload = []
            expected_role = "user"
            for msg in raw_history:
                if msg["role"] == expected_role:
                    contents_payload.append({"role": msg["role"], "parts": [{"text": msg["text"]}]})
                    expected_role = "model" if expected_role == "user" else "user"

            contents_payload.append({"role": "user", "parts": [{"text": item.message}]})

        keys_to_try = item.gemini_keys if item.gemini_keys else []
        if not keys_to_try:
            return {"success": False, "error": "No se proporcionaron llaves de Gemini."}

        headers = {"Content-Type": "application/json"}
        payload = {"contents": contents_payload}
        target_model = CHAT_MODEL
        last_error = ""

        for key in keys_to_try:
            if not key or len(key) < 10:
                continue
            gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={key}"
            gem_resp = SESSION.post(gemini_url, headers=headers, json=payload, timeout=45)

            if gem_resp.status_code == 200:
                gem_data = gem_resp.json()
                if "candidates" in gem_data and len(gem_data["candidates"]) > 0:
                    parts = gem_data["candidates"][0]["content"]["parts"]
                    answer = ""
                    for part in parts:
                        if part.get("thought", False):
                            continue
                        if "text" in part:
                            answer += part["text"]
                    answer = answer.strip()

                    if answer:
                        with chat_lock:
                            chat_context[session_id].append({"role": "user", "text": item.message})
                            chat_context[session_id].append({"role": "model", "text": answer})
                            if len(chat_context[session_id]) > 20:
                                chat_context[session_id] = chat_context[session_id][-20:]

                        final_reply = answer
                        if is_new_or_reset:
                            final_reply = "✨ *¡Chat iniciado / abierto con éxito!* 🚀\n\n" + answer

                        return {"success": True, "reply": final_reply}
            else:
                last_error = f"HTTP {gem_resp.status_code}: {gem_resp.text}"

        return {"success": False, "error": f"Todas las llaves fallaron. {last_error}"}
    except Exception as e:
        return {"success": False, "error": f"Excepción en chat: {str(e)}"}


# ---------------- ENDPOINTS MEMORIA ----------------
@app.post("/memory/save")
def save_memory(item: MemoryItem):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memories (user_id, content, created_at) VALUES (?, ?, ?)",
            (item.user_id, item.content, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (item.user_id,))
        count = cursor.fetchone()[0]
        if count > MAX_STORED_MEMORIES_PER_USER:
            cursor.execute("""
                DELETE FROM memories WHERE id IN (
                    SELECT id FROM memories WHERE user_id = ? ORDER BY id ASC LIMIT ?
                )
            """, (item.user_id, count - MAX_STORED_MEMORIES_PER_USER))
            conn.commit()
        conn.close()
    return {"success": True, "total_memories": count}


@app.get("/memory/get")
def get_memories(user_id: str):
    memories = []
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM memories WHERE user_id = ? ORDER BY id ASC", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        memories = [row[0] for row in rows]
    return {"user_id": user_id, "memories": memories}


@app.delete("/memory/clear")
def clear_memories(user_id: str):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
    return {"success": True, "message": "Memorias borradas para el usuario."}
