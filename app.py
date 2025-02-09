import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import openai
import os
import pandas as pd
import json
from streamlit_mic_recorder import mic_recorder
import io
from io import BytesIO
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()  # Load environment variables
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# Configuration
INVENTORY_PATH = 'inventory.csv'

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def get_firebase_app():
    """Initialize Firebase app with caching"""
    if not firebase_admin._apps:
        import json
        firebase_creds = json.loads(st.secrets["firebase"]["credentials"])
        cred = credentials.Certificate(firebase_creds)
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

def convert_audio_for_whisper(audio_bytes, mime_type):
    """Convert audio to WAV format that Whisper expects"""
    try:
        # Convert bytes to AudioSegment
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=mime_type.split('/')[-1])
        
        # Export as WAV
        wav_bytes = io.BytesIO()
        audio_segment.export(wav_bytes, format='wav')
        wav_bytes.seek(0)
        wav_bytes.name = "recording.wav"
        
        return wav_bytes
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        return None

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
            st.session_state.mime_type = audio.get('mime_type', 'audio/wav')
            st.session_state.recording_done = True
            st.audio(audio['bytes'])
            st.rerun()
            
        if st.session_state.recording_done:
            if st.button("Process Recording", key="process_recording"):
                try:
                    st.write("Converting audio...")
                    st.write(f"Audio format: {st.session_state.mime_type}")
                    st.write(f"Audio size: {len(st.session_state.audio_data)} bytes")
                    
                    audio_file = convert_audio_for_whisper(
                        st.session_state.audio_data, 
                        st.session_state.mime_type
                    )
                    if audio_file:
                        process_audio_input(audio_file, inventory_df)
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
            
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