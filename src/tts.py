import nemo.collections.asr as nemo_asr
import torchaudio

# Check if CUDA is available and set the device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained ASR model from NeMo
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En')
asr_model = asr_model.to(device)

# Load a large audio file for ASR (replace 'path_to_large_audio_file' with the actual path)
audio_file = 'path_to_large_audio_file'

# Perform ASR on the GPU
audio_signal, sample_rate = torchaudio.load(audio_file)
audio_signal = torch.tensor(audio_signal).to(device)
transcription = asr_model.transcribe([audio_signal])

# Ensure that the computation is finished before measuring time
torch.cuda.synchronize()

# Measure the time taken for ASR
import time
start_time = time.time()
transcription = asr_model.transcribe([audio_signal])
torch.cuda.synchronize()
end_time = time.time()

# Print the time taken for ASR
print(f"ASR on GPU took {end_time - start_time} seconds.")
print("Transcription:", transcription)
