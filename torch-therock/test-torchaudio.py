#!/usr/bin/env python3

import sys
import torch
import torchaudio
import numpy as np

def test_torchaudio():
    print("Testing torchaudio...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchaudio version: {torchaudio.__version__}")
    
    # Test basic import and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple synthetic audio signal (sine wave)
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440.0  # A4 note
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)  # Add channel dimension
    
    print(f"Created synthetic waveform: {waveform.shape}")
    
    # Test basic transformations
    try:
        # Test resampling
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
        resampled = resampler(waveform)
        print(f"Resampling test passed: {resampled.shape}")
        
        # Test spectrogram
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)
        print(f"Spectrogram test passed: {spectrogram.shape}")
        
        # Test mel spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
        print(f"Mel spectrogram test passed: {mel_spectrogram.shape}")
        
        # Test MFCC
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
        print(f"MFCC test passed: {mfcc.shape}")
        
        # Test moving to device if CUDA available
        if torch.cuda.is_available():
            waveform_gpu = waveform.to(device)
            spectrogram_transform_gpu = torchaudio.transforms.Spectrogram().to(device)
            spectrogram_gpu = spectrogram_transform_gpu(waveform_gpu)
            print(f"GPU spectrogram test passed: {spectrogram_gpu.shape}")
        
        print("\n✓ All torchaudio tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_torchaudio()
    sys.exit(0 if success else 1)