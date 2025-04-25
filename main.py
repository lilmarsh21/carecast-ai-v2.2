
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message

    full_prompt = (
        "You are a professional clinical AI assistant named Doctor AI.\n\n"
        f"The patient says: {user_message}\n\n"
        "Please ask the patient a thoughtful medical follow-up question to help determine a more accurate diagnosis."
    )

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are Doctor AI, a professional and clinical medical assistant. Always respond in a calm, helpful, and respectful tone."},
                {"role": "user", "content": full_prompt}
            ]
        )
        reply = completion['choices'][0]['message']['content']
        return {"message": reply}
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
