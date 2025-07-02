import torch
import torchaudio
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import cer
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")
hf_token = os.getenv("HF_TOKEN")

wandb.login(key=wandb_api_key)

# ✅ 설정
MODEL_NAME = "SungBeom/whisper-small-ko"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "/workspace/path_and_transcript_validation.csv"   # 평가용 CSV 파일 경로
SAVE_CSV = True                     # 결과 CSV 저장 여부
OUTPUT_DIR = "output"              # 결과 저장 폴더

def load_model_and_processor(model_name):
    processor = WhisperProcessor.from_pretrained(model_name, token=hf_token)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, token=hf_token).to(DEVICE)
    model.eval()
    return processor, model

def map_to_pred(batch, processor, model):
    speech_array, sr = torchaudio.load(batch["raw_data"])
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech_array = resampler(speech_array)
    
    input_features = processor(
        speech_array.squeeze(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().lower()
    batch["prediction"] = pred
    return batch

def run_evaluation():
    # ✅ wandb 실험 초기화
    wandb.init(
        project="stt-model-eval",
        name=MODEL_NAME.replace("/", "_"),
        config={
            "model": MODEL_NAME,
            "device": DEVICE,
            "sample_rate": 16000
        }
    )

    # ✅ 모델 및 데이터 로드
    processor, model = load_model_and_processor(MODEL_NAME)
    df = pd.read_csv(DATA_PATH)
    dataset = Dataset.from_pandas(df[["raw_data", "transcript"]])

    # ✅ 예측 실행
    result = dataset.map(lambda x: map_to_pred(x, processor, model))

    # ✅ CER 계산
    score = cer(result["transcript"], result["prediction"]) * 100
    print(f"[{MODEL_NAME}] CER: {score:.2f}%")

    # ✅ wandb 로깅
    wandb.log({"CER": score})

    # ✅ 결과 CSV 저장
    if SAVE_CSV:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME.replace('/', '_')}_result.csv")
        result.to_pandas().to_csv(save_path, index=False)
        print(f"📁 결과 저장: {save_path}")

    # ✅ wandb 종료
    wandb.finish()

if __name__ == "__main__":
    run_evaluation()
