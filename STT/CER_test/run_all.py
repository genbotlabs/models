import subprocess

files = [
    "whisper-small.py",
    "whisper-small-ko.py",
    "whisper-small-jw.py",
]

for f in files:
    print(f"\n🚀 Running: {f}")
    subprocess.run(["python", f"{f}"])