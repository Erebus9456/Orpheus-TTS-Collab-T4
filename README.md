---

# Orpheus TTS – Colab T4 Edition

## Overview

Orpheus TTS is an open-source text-to-speech system built on the LLaMA 3B backbone. This version has been **forked and modified** to work seamlessly with **Google Colab's free-tier T4 GPU**, enabling fast and efficient speech synthesis even without high-end hardware.

Read more about the original project and comparisons with closed models like Eleven Labs and PlayHT in the [official blog post](https://canopylabs.ai/model-releases).

---

## 🚀 Quickstart on Colab (T4 Compatible)

### ✅ Setup

```python
!git clone https://github.com/Erebus9456/Orpheus-TTS-Collab-T4.git
!cd Orpheus-TTS-Collab-T4/orpheus_tts_pypi && pip install .

!pip install -U "huggingface_hub[cli]"
!huggingface-cli login
```

### ✅ Usage

```python
from orpheus_tts import OrpheusModel
from vllm import AsyncEngineArgs, AsyncLLMEngine
import wave
import time
import types

# Define patched setup engine function
def patched_setup_engine(self):
    engine_args = AsyncEngineArgs(
        model=self.model_name,
        dtype=self.dtype,
        max_model_len=52400  # Set to match T4's memory limit
    )
    return AsyncLLMEngine.from_engine_args(engine_args)

# Initialize model
model = OrpheusModel(
    model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
    dtype="half"  # Use float16 for Tesla T4 compatibility
)

# Monkeypatch engine
model._setup_engine = types.MethodType(patched_setup_engine, model)

# Example prompt
prompt = '''"I have heard that Sanaubar's suggestive stride and oscillating hips sent men to reveries of infidelity..."'''

start_time = time.monotonic()
syn_tokens = model.generate_speech(prompt=prompt, voice="tara")

with wave.open("output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)

    total_frames = 0
    for audio_chunk in syn_tokens:
        frame_count = len(audio_chunk) // 2
        total_frames += frame_count
        wf.writeframes(audio_chunk)

end_time = time.monotonic()
print(f"Generated {total_frames / 24000:.2f}s of audio in {end_time - start_time:.2f}s")
```

---

## 🔥 Why This Fork?

This fork modifies:

* Engine config for **Tesla T4 compatibility**.
* Model dtype to **float16** to save memory.
* Includes a **monkeypatch** to bypass internal engine size limits in vLLM.
* Designed for **quick testing and prototyping** on Colab.

---

## Core Capabilities (Same as Original)

* ✅ **Human-Like Speech**
* ✅ **Zero-Shot Voice Cloning**
* ✅ **Emotion Control Tags**
* ✅ **Low Latency Streaming**

> For full documentation of features, finetuning, and pretraining, refer to the [original README](https://github.com/canopyai/Orpheus-TTS).

---

## 📌 Notes

* This fork is ideal for **lightweight, personal use**, especially for **students, researchers, and hobbyists** using free-tier Colab.
* For more advanced or production deployment, we recommend reviewing the full project capabilities from [Canopy AI](https://github.com/canopyai/Orpheus-TTS).

---
