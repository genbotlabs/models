import torch
import torchaudio
import pandas as pd
from datasets import Dataset
from jiwer import cer
import nemo.collections.asr as nemo_asr
import wandb
import os
from dotenv import load_dotenv

'''
You need to set nemo

Try use:
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_conformer_transducer_large/versions/1.8.0/files/stt_zh_conformer_transducer_large.nemo


def load_model(model_path):
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model_path, map_location=DEVICE)
    model.eval()
    return model

MODEL_PATH = "stt_zh_conformer_transducer_large.nemo"
model = load_model(MODEL_PATH)


'''

load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")

# ✅ 설정
MODEL_NAME = "stt_zh_conformer_transducer_large"  # 변경된 모델명
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "/workspace/path_and_transcript_validation.csv"
SAVE_CSV = True
OUTPUT_DIR = "output"

def load_model(model_name):
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
    model.to(DEVICE)
    model.eval()
    return model

def map_to_pred(batch, model):
    speech_array, sr = torchaudio.load(batch["raw_data"])
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech_array = resampler(speech_array)

    # RNN-T 모델은 EncDecRNNTBPEModel로 transcribe 사용
    transcription = model.transcribe([speech_array.squeeze().numpy()])[0]['text'].strip().lower()
    batch["prediction"] = transcription
    return batch

def run_evaluation():
    wandb.init(
        project="stt-model-eval-0630",
        name=MODEL_NAME.replace("/", "_"),
        config={
            "model": MODEL_NAME,
            "device": DEVICE,
            "sample_rate": 16000
        }
    )

    # ✅ 모델 및 데이터 로드
    model = load_model(MODEL_NAME)
    df = pd.read_csv(DATA_PATH)
    dataset = Dataset.from_pandas(df[["raw_data", "transcript"]])

    # ✅ 예측 수행
    result = dataset.map(lambda x: map_to_pred(x, model))

    # ✅ CER 계산
    score = cer(result["transcript"], result["prediction"]) * 100
    print(f"[{MODEL_NAME}] CER: {score:.2f}%")

    wandb.log({"CER": score})

    if SAVE_CSV:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME.replace('/', '_')}_result.csv")
        result.to_pandas().to_csv(save_path, index=False)
        print(f"📁 결과 저장: {save_path}")

    wandb.finish()

if __name__ == "__main__":
    run_evaluation()
