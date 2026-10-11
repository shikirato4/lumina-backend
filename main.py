import base64
import os
import threading
import time
import urllib.parse
from fastapi import FastAPI, HTTPException
import google.generativeai as genai
from pydantic import BaseModel
import requests

app = FastAPI(title="Lumina Backend", version="2.5")

memory_store = []
chat_context = {}
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Whitelist de IDs autorizados
allowed_users = {"6363412200", "8389931379", "1057724787"}

GEMINI_MODEL = "gemini-1.5-flash"


class MemoryItem(BaseModel):
  user_id: str
  content: str


class ChatItem(BaseModel):
  user_id: str
  chat_id: str
  message: str
  gemini_keys: list[str] = []
  model: str = "gemma-4-31b-it"


class VisionItem(BaseModel):
  user_id: str
  chat_id: str
  file_id: str
  gemini_keys: list[str] = []
  caption: str = "Analiza este archivo con detalle."
  mime_type: str = "image/jpeg"


class ImageItem(BaseModel):
  chat_id: str
  prompt: str
  gemini_keys: list[str] = []


# Evento de inicio en Render: Notifica al admin que el servidor encendió correctamente
@app.on_event("startup")
def startup_event():
  if TELEGRAM_BOT_TOKEN:
    try:
      url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
      requests.post(
          url,
          json={
              "chat_id": "6363412200",
              "text": (
                  "🟢 ¡Reinicio completado! Lumina Backend está en línea y"
                  " lista."
              ),
          },
      )
    except Exception as e:
      print(f"Error al enviar mensaje de inicio: {e}")


@app.get("/")
def health_check():
  return {"status": "online", "bot": "Lumina Backend Active"}


# --- WEBHOOK PARA COMANDOS DE TELEGRAM ---
@app.post("/webhook")
def telegram_webhook(update: dict):
  try:
    if "message" not in update:
      return {"ok": True}

    msg = update["message"]
    chat_id = msg["chat"]["id"]
    user_id = str(msg["from"]["id"])
    text = msg.get("text", "")

    # Validar Whitelist para mensajes generales (excepto /start)
    if user_id not in allowed_users and not text.startswith("/start"):
      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
          json={
              "chat_id": chat_id,
              "text": (
                  "❌ No estás autorizado para usar este bot. Solicita acceso"
                  " al administrador."
              ),
          },
      )
      return {"ok": True}

    # Comando /addnewuser (Solo permitido para el administrador principal)
    if text.startswith("/addnewuser"):
      if user_id != "6363412200":
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "❌ No tienes permisos para usar este comando.",
            },
        )
        return {"ok": True}

      parts = text.split()
      if len(parts) > 1:
        new_id = parts[1].strip()
        allowed_users.add(new_id)
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": (
                    f"✅ Usuario con ID `{new_id}` agregado exitosamente a la"
                    " whitelist."
                ),
            },
        )
      else:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "⚠️ Uso correcto: `/addnewuser <ID_DEL_USUARIO>`",
            },
        )
      return {"ok": True}

    # Comando /reset (Estrictamente restringido al administrador principal)
    if text == "/reset":
      if user_id != "6363412200":
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "❌ No tienes permisos para usar este comando.",
            },
        )
        return {"ok": True}

      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
          json={"chat_id": chat_id, "text": "🔄 Reiniciando sistema..."},
      )

      def delayed_restart():
        time.sleep(1)
        os._exit(0)

      threading.Thread(target=delayed_restart).start()
      return {"ok": True}

    return {"ok": True}
  except Exception as e:
    print(f"Error en webhook: {e}")
    return {"ok": True}


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


@app.delete("/chat/clear")
def clear_chat(user_id: str):
  if user_id in chat_context:
    chat_context[user_id] = []

  welcome_msg = (
      "✨ *¡Chat reiniciado / abierto con éxito!* \nHola, soy Lumina. ¿En qué"
      " te puedo ayudar hoy?"
  )
  return {
      "success": True,
      "message": "Contexto de conversación reiniciado.",
      "welcome_message": welcome_msg,
  }


@app.post("/chat")
def handle_chat_backend(item: ChatItem):
  try:
    user_id = item.user_id
    is_new_or_reset = False

    if user_id not in chat_context:
      chat_context[user_id] = []
      is_new_or_reset = True
    elif len(chat_context[user_id]) == 0:
      is_new_or_reset = True

    history = chat_context[user_id][-10:]
    contents_payload = []
    for msg in history:
      contents_payload.append(
          {"role": msg["role"], "parts": [{"text": msg["text"]}]}
      )

    contents_payload.append({"role": "user", "parts": [{"text": item.message}]})

    keys_to_try = item.gemini_keys if item.gemini_keys else []
    if not keys_to_try:
      return {
          "success": False,
          "error": "No se proporcionaron llaves de Gemini.",
      }

    headers = {"Content-Type": "application/json"}
    payload = {"contents": contents_payload}

    target_model = item.model if item.model else "gemma-4-31b-it"
    last_error = ""

    for key in keys_to_try:
      if not key or len(key) < 10:
        continue
      gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={key}"
      gem_resp = requests.post(gemini_url, headers=headers, json=payload)

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
            chat_context[user_id].append(
                {"role": "user", "text": item.message}
            )
            chat_context[user_id].append({"role": "model", "text": answer})

            final_reply = answer
            if is_new_or_reset:
              final_reply = (
                  "✨ *¡Chat iniciado / abierto con éxito!* 🚀\n\n" + answer
              )

            return {"success": True, "reply": final_reply}
      else:
        last_error = f"HTTP {gem_resp.status_code}: {gem_resp.text}"

    return {"success": False, "error": f"Todas las llaves fallaron. {last_error}"}
  except Exception as e:
    return {"success": False, "error": f"Excepción en chat: {str(e)}"}


@app.post("/vision")
def analyze_vision(item: VisionItem):
  try:
    tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={item.file_id}"
    r = requests.get(tg_url)
    res_json = r.json()
    if not res_json.get("ok"):
      return {
          "success": False,
          "error": (
              "Telegram error:"
              f" {res_json.get('description', 'Desconocido')}"
          ),
      }

    file_path = res_json["result"]["file_path"]
    file_url = (
        f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    )
    file_response = requests.get(file_url)
    if file_response.status_code != 200:
      return {"success": False, "error": "No se pudo descargar de Telegram"}

    encoded_data = base64.b64encode(file_response.content).decode("utf-8")
    user_id = item.user_id
    if user_id not in chat_context:
      chat_context[user_id] = []

    history = chat_context[user_id][-10:]
    contents_payload = []
    for msg in history:
      contents_payload.append(
          {"role": msg["role"], "parts": [{"text": msg["text"]}]}
      )

    contents_payload.append({
        "role": "user",
        "parts": [
            {"text": item.caption},
            {"inline_data": {"mime_type": item.mime_type, "data": encoded_data}},
        ],
    })

    keys_to_try = item.gemini_keys if item.gemini_keys else []
    if not keys_to_try:
      return {
          "success": False,
          "error": "No se proporcionaron llaves de Gemini.",
      }

    headers = {"Content-Type": "application/json"}
    payload = {"contents": contents_payload}

    last_error = ""
    for key in keys_to_try:
      if not key or len(key) < 10:
        continue
      gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
      gem_resp = requests.post(gemini_url, headers=headers, json=payload)

      if gem_resp.status_code == 200:
        gem_data = gem_resp.json()
        if "candidates" in gem_data and len(gem_data["candidates"]) > 0:
          answer = gem_data["candidates"][0]["content"]["parts"][0].get(
              "text", ""
          )
          chat_context[user_id].append({"role": "user", "text": item.caption})
          chat_context[user_id].append({"role": "model", "text": answer})
          return {"success": True, "analysis": answer}
      else:
        last_error = f"HTTP {gem_resp.status_code}: {gem_resp.text}"

    return {"success": False, "error": f"Todas las llaves fallaron. {last_error}"}
  except Exception as e:
    return {"success": False, "error": f"Excepción: {str(e)}"}


@app.post("/image")
def generate_image(item: ImageItem):
  try:
    enhanced_prompt = item.prompt
    keys_to_try = item.gemini_keys if item.gemini_keys else []

    for key in keys_to_try:
      if key and len(key) > 10:
        try:
          genai.configure(api_key=key)
          model = genai.GenerativeModel(GEMINI_MODEL)
          opt_res = model.generate_content(
              "Translate and optimize this prompt into English for FLUX AI to"
              f" generate an ultra-realistic, highly detailed, cinematic"
              f" masterpiece image: {item.prompt}"
          )
          enhanced_prompt = opt_res.text.strip()
          break
        except Exception:
          continue

    encoded_prompt = urllib.parse.quote(enhanced_prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux-realism&width=1024&height=1024&nologo=true"

    img_resp = requests.get(image_url, timeout=55)
    if img_resp.status_code == 200:
      tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
      files = {"photo": ("image.png", img_resp.content, "image/png")}
      data = {
          "chat_id": item.chat_id,
          "caption": f"✨ *Prompt:* {item.prompt}",
          "parse_mode": "Markdown",
      }
      requests.post(tg_url, data=data, files=files)
      return {"success": True}
    else:
      return {"success": False, "error": "servidor lleno espera un minuto"}
  except Exception as e:
    return {"success": False, "error": "servidor lleno espera un minuto"}
