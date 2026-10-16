import base64
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import google.generativeai as genai
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Lumina Backend", version="3.6-jarvis")

# ---------------- CONFIGURACIÓN ----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
ADMIN_ID = "6363412200"

CHAT_MODEL = "gemma-4-31b-it"
VISION_AND_PROMPT_MODEL = "gemini-3.8-flash"
CONCEPTUAL_MODEL = "gemma-4-31b-it"
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-dev"

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
    max_workers=4,
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                task TEXT NOT NULL,
                trigger_at TEXT NOT NULL,
                interval_minutes INTEGER DEFAULT 0
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


class MapItem(BaseModel):
    chat_id: str
    theme: str
    request_id: str = ""


# ---------------- HILO SEGUNDO PLANO (RECORDATORIOS) ----------------
def reminder_background_worker():
    while True:
        try:
            now_str = datetime.now(timezone.utc).isoformat()
            with _db_lock:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, chat_id, task, trigger_at, interval_minutes FROM reminders WHERE trigger_at <= ?",
                    (now_str,)
                )
                due_reminders = cursor.fetchall()

                for rem_id, chat_id, task, trigger_at, interval_min in due_reminders:
                    msg_text = f"⏰ *¡Recordatorio Jarvis!*\n\n{task}"
                    try:
                        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                        SESSION.post(
                            url,
                            json={"chat_id": chat_id, "text": msg_text, "parse_mode": "Markdown"},
                            timeout=10
                        )
                    except Exception as e:
                        print(f"Error enviando recordatorio a Telegram: {e}")

                    if interval_min and interval_min > 0:
                        next_time = datetime.fromisoformat(trigger_at) + timedelta(minutes=interval_min)
                        if next_time < datetime.now(timezone.utc):
                            next_time = datetime.now(timezone.utc) + timedelta(minutes=interval_min)
                        cursor.execute(
                            "UPDATE reminders SET trigger_at = ? WHERE id = ?",
                            (next_time.isoformat(), rem_id)
                        )
                    else:
                        cursor.execute("DELETE FROM reminders WHERE id = ?", (rem_id,))

                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Error en worker de recordatorios: {e}")
        time.sleep(60)


@app.on_event("startup")
def startup_event():
    # Arrancar el hilo de recordatorios autónomos
    threading.Thread(target=reminder_background_worker, daemon=True).start()

    if TELEGRAM_BOT_TOKEN:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            SESSION.post(
                url,
                json={
                    "chat_id": ADMIN_ID,
                    "text": "🟢 ¡Lumina 3.6 Jarvis en línea! (FLUX-dev + Visión + Mapas + Recordatorios + Memoria)",
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


# ---------------- ENDPOINT IMAGEN (ASÍNCRONO CON FLUX-DEV) ----------------
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

            # Optimización de prompt con estilo fotográfico crudo y natural (sin CGI ni plásticos)
            for key in keys_to_try:
                if key and len(key) > 10:
                    try:
                        genai.configure(api_key=key)
                        model = genai.GenerativeModel(VISION_AND_PROMPT_MODEL)
                        opt_res = model.generate_content(
                            f"Translate and expand this prompt into a natural, raw, casual, unedited amateur smartphone photo style with realistic lighting, natural skin textures, imperfections, and optical artifacts, avoiding any CGI or cinematic render look: {item.prompt}"
                        )
                        if opt_res and hasattr(opt_res, "text") and opt_res.text:
                            enhanced_prompt = opt_res.text.strip()
                        break
                    except Exception:
                        continue

            image_bytes = None
            engine_used = ""

            # 1. Cloudflare Workers AI con FLUX-DEV
            if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:
                try:
                    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
                    cf_headers = {
                        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                        "Content-Type": "application/json",
                    }
                    cf_payload = {"prompt": enhanced_prompt}
                    cf_resp = SESSION.post(cf_url, headers=cf_headers, json=cf_payload, timeout=40)
                    
                    print(f"CF Status Code: {cf_resp.status_code}")
                    print(f"CF Response Text: {cf_resp.text}")

                    if cf_resp.status_code == 200:
                        res_json = cf_resp.json()
                        if "result" in res_json and res_json["result"] and "image" in res_json["result"]:
                            image_bytes = base64.b64decode(res_json["result"]["image"])
                            engine_used = "Cloudflare FLUX.1-dev"
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


# ---------------- ENDPOINT CONCEPTUAL MAP (MERMAID CLI) ----------------
@app.post("/conceptual_map")
def create_map_job(item: MapItem):
    with _job_lock:
        if item.request_id and item.request_id in job_request_index:
            existing_id = job_request_index[item.request_id]
            return {"success": True, "job_id": existing_id}

        job_id = uuid.uuid4().hex
        if item.request_id:
            job_request_index[item.request_id] = job_id

        jobs[job_id] = {"state": "processing", "success": False, "error": None}

    def run_map_task():
        try:
            keys_to_try = [os.getenv("GEMINI_KEY", "")]
            mermaid_code = ""

            for key in keys_to_try:
                if key and len(key) > 10:
                    try:
                        genai.configure(api_key=key)
                        model = genai.GenerativeModel(CONCEPTUAL_MODEL)
                        system_instr = "You are an AI specialized in conceptual diagrams. Create a logical, structured Mermaid.js conceptual map or flowchart of the requested theme. Provide ONLY the Mermaid.js code, with legible labels in Spanish, without markdown formatting or code blocks. Use a 'graph TD' (top-down) structure."
                        
                        opt_res = model.generate_content(
                            contents=[{"role": "user", "parts": [{"text": f"Tema: {item.theme}"}]}],
                            generation_config={"temperature": 0.2},
                            system_instruction={"parts": [{"text": system_instr}]}
                        )
                        if opt_res and hasattr(opt_res, "text") and opt_res.text:
                            mermaid_code = opt_res.text.strip()
                        break
                    except Exception:
                        continue

            if not mermaid_code or len(mermaid_code) < 20:
                with _job_lock:
                    jobs[job_id] = {"state": "completed", "success": False, "error": "No se pudo generar el código del mapa mental."}
                return

            output_filename = f"diagram-{job_id}.png"
            command = ["mmdc", "-i", "-", "-o", output_filename, "-b", "white", "-t", "neutral"]
            
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=mermaid_code, timeout=35)

            if process.returncode != 0 or not os.path.exists(output_filename):
                with _job_lock:
                    jobs[job_id] = {"state": "completed", "success": False, "error": f"Error renderizando diagrama: {stderr}"}
                return

            with open(output_filename, "rb") as image_file:
                map_bytes = image_file.read()
            os.remove(output_filename)

            tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {"photo": ("map.png", map_bytes, "image/png")}
            data = {
                "chat_id": item.chat_id,
                "caption": f"💡 *Mapa Conceptual:* {item.theme}\n🛠️ *Generado con:* Gemma + Mermaid",
                "parse_mode": "Markdown",
            }
            tg_resp = SESSION.post(tg_url, data=data, files=files, timeout=45)

            with _job_lock:
                jobs[job_id] = {"state": "completed", "success": tg_resp.status_code == 200, "error": None}
        except Exception as e:
            with _job_lock:
                jobs[job_id] = {"state": "completed", "success": False, "error": f"Excepción en mapa: {str(e)}"}

    executor.submit(run_map_task)
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


# ---------------- ENDPOINT CHAT + RECORDATORIOS INTELIGENTES ----------------
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

        # Instrucción de sistema para detector de recordatorios en lenguaje natural + Jarvis
        now_iso = datetime.now(timezone.utc).isoformat()
        system_instruction = f"""You are Lumina (Jarvis), an advanced personal AI assistant. 
Current UTC Time: {now_iso}.
Analyze the user message. If the user wants to set a reminder (e.g., "recuérdame tal en 10 minutos", "recuérdame cada 2 horas"), you MUST output a JSON block at the very beginning of your response in this exact format:
{{
  "is_reminder": true,
  "task": "descripción de la tarea",
  "trigger_at": "ISO8601 UTC timestamp when it should fire",
  "interval_minutes": 0 or minutes if it's recurring
}}
If it is NOT a reminder, output:
{{
  "is_reminder": false
}}
Followed by your natural conversational response to the user in Spanish."""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": contents_payload,
            "system_instruction": {"parts": [{"text": system_instruction}]}
        }
        
        target_model = CHAT_MODEL
        answer = ""
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
                    for part in parts:
                        if part.get("thought", False):
                            continue
                        if "text" in part:
                            answer += part["text"]
                    answer = answer.strip()
                    if answer:
                        break
            else:
                last_error = f"HTTP {gem_resp.status_code}: {gem_resp.text}"

        if not answer:
            return {"success": False, "error": f"Todas las llaves fallaron. {last_error}"}

        reply_to_user = answer
        if "is_reminder" in answer:
            try:
                import json
                first_brace = answer.find("{")
                last_brace = answer.rfind("}")
                if first_brace != -1 and last_brace != -1:
                    json_str = answer[first_brace:last_brace+1]
                    rem_data = json.loads(json_str)
                    
                    if rem_data.get("is_reminder"):
                        task_desc = rem_data.get("task", "Recordatorio general")
                        trigger_at = rem_data.get("trigger_at", datetime.now(timezone.utc).isoformat())
                        interval_min = rem_data.get("interval_minutes", 0)

                        with _db_lock:
                            conn = sqlite3.connect(DB_PATH)
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT INTO reminders (chat_id, task, trigger_at, interval_minutes) VALUES (?, ?, ?, ?)",
                                (session_id, task_desc, trigger_at, interval_min)
                            )
                            conn.commit()
                            conn.close()

                        reply_to_user = answer[last_brace+1:].strip()
                        if not reply_to_user:
                            reply_to_user = f"✅ ¡Entendido! He programado tu recordatorio: *{task_desc}*."
            except Exception as e:
                print(f"Error procesando JSON de recordatorio: {e}")

        with chat_lock:
            chat_context[session_id].append({"role": "user", "text": item.message})
            chat_context[session_id].append({"role": "model", "text": reply_to_user})
            if len(chat_context[session_id]) > 20:
                chat_context[session_id] = chat_context[session_id][-20:]

        final_reply = reply_to_user
        if is_new_or_reset and not "is_reminder" in answer:
            final_reply = "✨ *¡Chat iniciado / abierto con éxito!* 🚀\n\n" + reply_to_user

        return {"success": True, "reply": final_reply}

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
