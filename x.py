# import torch
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(device)
# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.amp)
# print(torch.cuda.amp.autocast)
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())

import streamlit as st
import speech_recognition as sr


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
    st.title("Real-time Voice-to-Text")
    
    if st.button("Start Recording"):
        st.write("Recording...")
        transcript = record_and_transcribe()
        st.write("Transcription:")
        st.write(transcript)

if __name__ == "__main__":
    main()


