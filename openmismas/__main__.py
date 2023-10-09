import whisper_timestamped as whisper
import pandas as pd
from ultralytics import YOLO
import os
import tqdm 
import torch
import cv2
import time


def transcribe_audio(source: str, model: str  = "large", device: str = "cpu",  language: str = "en") -> dict:
    """
    Transcribes an audio file using a Whisper model.
    
    Args:
    - source: Path to the audio file.
    - model: Whisper model.
    - language: Language of the audio file.
    
    Returns:
    - A dictionary containing the transcription results.
    """
    model = whisper.load_model(model, device=device)
    audio = whisper.load_audio(source)
    result = whisper.transcribe(model, audio, language=language)
    return result


def transcription_to_df(result):
    rows = []

    for entry in result['segments']:
        for word in entry['words']:
            rows.append({
                'Word': word['text'],
                'Start': word['start'],
                'End': word['end'],
                'Confidence': word['confidence']
            })

    return pd.DataFrame(rows)


def define_device() -> str:
    if torch.cuda.is_available():
        return"cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"
    

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def get_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return duration


def track_objects(source: str, model: YOLO = YOLO("yolov8n.pt"), device: str = define_device()):
    """
    Track objects in a video using a YOLO model.
    :param source: Path to the video file.
    :param model: YOLO model.
    :param device: Device to use for inference.
    :return: List of results.
    """
    # Track objects
    results = model.track(source, verbose=False, task="detect", imgsz=640,
                          save=True, save_txt=True, stream=True, save_crop=False, device=device)
    return results


def object_tracking_to_df(results, source: str) -> pd.DataFrame:
    """
    Processes YOLO results and returns a pandas DataFrame containing object detection information.
    
    Args:
    - results: A list of YOLO results.
    - source: The source of the results.
    
    Returns:
    - A pandas DataFrame containing object detection information, with the following columns:
        - Frame_num: The frame number.
        - Object_name: The name of the detected object.
        - Object_id: The ID of the detected object.
        - Object_conf: The confidence score of the detected object.
        - Object_bbox_left: The left coordinate of the bounding box of the detected object.
        - Object_bbox_top: The top coordinate of the bounding box of the detected object.
        - Object_bbox_right: The right coordinate of the bounding box of the detected object.
        - Object_bbox_bottom: The bottom coordinate of the bounding box of the detected object.
    """
    data = []
    
    for i, result in tqdm.tqdm(enumerate(results), total=get_frame_count(source)):
        for box in result.boxes:
            frame_num = i
            object_name = result.names[int(box.cls)]
            object_id = int(box.id) if box.id is not None else 0
            object_conf = float(box.conf)
            object_bbox = box.xyxyn.tolist()[0]
            data.append([frame_num, object_name, object_id, object_conf, *object_bbox])
    
    df = pd.DataFrame(data, columns=['Frame_num', 'Object_name', 'Object_id', 'Object_conf', 'Object_bbox_left', 'Object_bbox_top', 'Object_bbox_right', 'Object_bbox_bottom'])
    return df


def main_function():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== MISMAS v2 ===")
    print("\n")
    source = input("Enter the path to the video file: ")
    output = input("Enter the path to the output directory: ")
    duration = get_duration(source)
    print("\n")
    print("Video duration: {} seconds".format(duration))
    t0 = time.time()
    transcription = transcribe_audio(source)
    transcription_df = transcription_to_df(transcription)
    transcription_df.to_csv(os.path.join(output, f"{source}_transcription.csv"), index=False)
    t1 = time.time()
    print("Transcription saved to {}".format(os.path.join(output, f"{source}_transcription.csv")))
    print("Transcription completed in {} seconds".format(t1 - t0))
    t2 = time.time()
    object_tracking = track_objects(source)
    object_tracking_df = object_tracking_to_df(object_tracking, source)
    object_tracking_df.to_csv(os.path.join(output, f"{source}_object_tracking.csv"), index=False)
    t3 = time.time()
    print("Object tracking saved to {}".format(os.path.join(output, f"{source}_object_tracking.csv")))
    print("Object tracking completed in {} seconds".format(t3 - t2))
    print("\n")
    print("Transcription took {} times the duration of the video".format((t1 - t0) / duration))
    print("Object tracking took {} times the duration of the video".format((t3 - t2) / duration))
    



if __name__ == "__main__":
    main_function()