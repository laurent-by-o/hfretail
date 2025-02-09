# hands free retail


from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

firebase
var admin = require("firebase-admin");

var serviceAccount = require("path/to/serviceAccountKey.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});


import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import openai

# Configure OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

def transcribe_audio(audio_bytes):
    """Convert audio to text using OpenAI Whisper"""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = 'recording.wav'
    
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']



# Initialize Firebase
cred = credentials.Certificate('path/to/firebase_credentials.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def query_inventory(query):
    # Query Firestore instead of Google Sheets
    inventory_ref = db.collection('inventory')
    results = inventory_ref.where('size', '==', '8').stream()
    
    # Process results with OpenAI
    inventory_data = [doc.to_dict() for doc in results]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Interpret inventory queries"},
            {"role": "user", "content": f"From this data {inventory_data}, answer: {query}"}
        ]
    )
    
    return response.choices[0].message.content
def main():
    st.title("Inventory Voice Assistant")
    
    # Audio recording
    audio_bytes = st.audio_recorder("Speak your inventory query")
    
    if audio_bytes:
        # Transcribe audio
        query = transcribe_audio(audio_bytes)
        st.write(f"You asked: {query}")
        
        # Query inventory
        inventory_response = query_inventory(query)
        
        # Display results
        st.write("Inventory Response:")
        st.write(inventory_response)

if __name__ == "__main__":
    main()