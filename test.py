import streamlit as st
from pydub import AudioSegment
from io import BytesIO
import speech_recognition as sr
def text_input():
    return st.text_input("Enter Text:", "")

def record_and_transcribe():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10)

    try:
        transcript = recognizer.recognize_google(audio)
        return transcript
    except sr.UnknownValueError:
        return "Unable to recognize speech"
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

def main():
    st.title("Text or Speech Input App")

    input_type = st.radio("Select Input Type:", ["Text", "Speech"])

    if input_type == "Text":
        text = text_input()
        st.write("You entered:", text)
    elif input_type == "Speech":
        st.write("Recording...")
        transcript = record_and_transcribe()
        st.write("Transcription:")
        st.write(transcript)

if __name__ == "__main__":
    main()



