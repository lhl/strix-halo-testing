import torch, librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") 
audio_input, _ = librosa.load("test_audio.wav", sr=16000)
input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
result = model(input_values, output_hidden_states=True)  # <-- CRASHES HERE
