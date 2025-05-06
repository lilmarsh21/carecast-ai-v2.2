from fastapi import FastAPI
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

sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    message = request.message

    if session_id not in sessions:
        sessions[session_id] = {
            "history": [
                {"role": "system", "content": (
                    "You are Doctor AI, a compassionate and professional human medical doctor only. "
                    "You never provide veterinary or animal-related help. Your role is to help with humans only. "
                    "Ask one important medical follow-up question at a time to fully understand the patient's situation. "
                    "Ask as many questions at it takes to get as close as possible to a diagnoses of 100%. "
                    "Always confirm that you're speaking with or about a human patient. "
                    "After gathering enough information, provide a list of possible diagnoses with a percentage risk likelihood for each. "
                    "Never skip information gathering. Be serious, empathetic, clear, and clinically professional. "
                    "If asked about an animal or pet, politely explain you can only assist with human medicine."
                )}
            ],
            "questions_asked": 0,
            "diagnosis_given": False
        }

    session = sessions[session_id]
    session["history"].append({"role": "user", "content": message})

    # Decide what AI should do based on how many questions have been asked
    if session["questions_asked"] < 5:
        ai_instruction = (
            "Ask the next most important medical follow-up question to understand the patient better. "
            "Only one question. Do not give a diagnosis yet."
        )
        session["questions_asked"] += 1
    else:
        ai_instruction = (
            "You have collected enough information. Now provide a detailed diagnosis. "
            "List possible conditions with percentage likelihoods, and explain why."
        )
        session["diagnosis_given"] = True

    # Always send the history + current instruction
    full_messages = session["history"] + [{"role": "system", "content": ai_instruction}]

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=full_messages
    )

    reply = response["choices"][0]["message"]["content"]
    session["history"].append({"role": "assistant", "content": reply})

    return {"message": reply}


