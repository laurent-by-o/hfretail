import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import openai
import os
import pandas as pd
import json
from streamlit_mic_recorder import mic_recorder
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# Configuration
FIREBASE_CRED_PATH = './hands-free-retail-firebase-adminsdk-fbsvc-a3c542b686.json'
INVENTORY_PATH = 'inventory.csv'

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_KEY)

# Use Streamlit's caching to initialize Firebase only once
@st.cache_resource
def get_firebase_app():
    """Initialize Firebase app with caching"""
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
    return firebase_admin.get_app()

# Initialize Firebase and get db client
app = get_firebase_app()
db = firestore.client()

def process_audio_input(audio_data, inventory_df):
    """Process either recorded or uploaded audio"""
    st.write("Transcribing audio...")
    transcript = transcribe_audio(audio_data)
    if transcript:
        st.write("You asked:", transcript)
        intent_data = parse_query_intent(transcript)
        if intent_data:
            st.write("Understanding:", intent_data)
            results = search_inventory(intent_data, inventory_df)
            st.write("Inventory Results:")
            st.dataframe(results)

def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI Whisper"""
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

@st.cache_data
def load_inventory():
    """Load inventory data from CSV"""
    try:
        df = pd.read_csv('inventory.csv')
        return df
    except Exception as e:
        st.error(f"Error loading inventory: {str(e)}")
        return pd.DataFrame()

def parse_query_intent(transcript):
    """Use GPT to understand the query intent and extract parameters"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a retail inventory assistant. Given a question, extract the key parameters and return ONLY a JSON object with: intent (check_stock/find_location/check_price) and parameters (size/category/color/product_name)."},
                {"role": "user", "content": transcript}
            ]
        )
        # Parse the response to ensure it's valid JSON
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error parsing intent: {str(e)}")
        return None

def search_inventory(intent_data, df):
    """Search inventory based on parsed intent"""
    try:
        filtered_df = df.copy()
        params = intent_data.get('parameters', {})
        
        if 'size' in params:
            filtered_df = filtered_df[filtered_df['Size'].astype(str) == str(params['size'])]
        if 'category' in params:
            filtered_df = filtered_df[filtered_df['Category'].str.lower() == params['category'].lower()]
            
        return filtered_df
    except Exception as e:
        st.error(f"Error searching inventory: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("Inventory Voice Assistant")
    inventory_df = load_inventory()

    tab1, tab2 = st.tabs(["Record", "Upload"])

    with tab1:
        st.write("Record your question")
        if 'recording_done' not in st.session_state:
            st.session_state.recording_done = False
            
        audio = mic_recorder(
            key="recorder",
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=True
        )
        
        if audio and not st.session_state.recording_done:
            st.session_state.audio_data = audio['bytes']
            st.session_state.recording_done = True
            st.audio(audio['bytes'])
            st.rerun()
            
        if st.session_state.recording_done:
            if st.button("Process Recording", key="process_recording"):
                audio_bytes = BytesIO(st.session_state.audio_data)
                audio_bytes.name = "recording.wav"
                process_audio_input(audio_bytes, inventory_df)
            
            if st.button("Record Again", key="record_again"):
                st.session_state.recording_done = False
                st.rerun()

    with tab2:
        audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])
        if audio_file:
            st.audio(audio_file)
            if st.button("Process Upload", key="process_upload"):
                process_audio_input(audio_file, inventory_df)

if __name__ == "__main__":
    main()