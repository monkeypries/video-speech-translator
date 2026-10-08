import argparse
import os
import whisper

def transcribe_video(file_path, task="translate", model_size="medium"):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        return

    print(f"Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    print(f"Processing '{file_path}' for task: {task}...")
    result = model.transcribe(file_path, task=task)

    output_file = f"{os.path.splitext(file_path)[0]}_transcript.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print("\n--- OUTPUT ---")
    print(result["text"])
    print(f"\nSaved transcript to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Audio Transcription using OpenAI Whisper")
    parser.add_argument("--file", type=str, required=True, help="Path to video file")
    parser.add_argument("--task", type=str, default="translate", choices=["transcribe", "translate"], help="Task type")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size")

    args = parser.parse_args()
    transcribe_video(args.file, args.task, args.model)