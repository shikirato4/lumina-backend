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
        
        # 2. Descargar el archivo multimedia desde Telegram
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        img_response = requests.get(file_url)
        if img_response.status_code != 200:
            return {"success": False, "error": "No se pudo descargar el archivo"}
        
        # 3. Codificar en base64 para enviarlo a la IA
        encoded_data = base64.b64encode(img_response.content).decode("utf-8")
        
        # 4. Usar un modelo multimedia en la nube exclusivo para visión/archivos
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
                                "mime_type": "image/jpeg",
                                "data": encoded_data
                            }
                        }
                    ]
                }
            ]
        }
        
        gem_resp = requests.post(gemini_url, headers=headers, json=payload)
        if gem_resp.status_code != 200:
            return {"success": False, "error": f"Error en API Multimedia: {gem_resp.text}"}
            
        gem_data = gem_resp.json()
        answer = gem_data["candidates"][0]["content"]["parts"][0]["text"]
        return {"success": True, "analysis": answer}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
