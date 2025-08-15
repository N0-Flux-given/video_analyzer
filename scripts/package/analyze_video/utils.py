import os
import cv2
import base64
import requests
import json
import matplotlib.pyplot as plt
import shutil
import numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .config import (
    CUTS_DESCRIPTION_PATH,
    CUTS_PATH,
    PERCENT_FRAMES_TO_SAMPLE,
    OLLAMA_URL,
)


def describe_image_with_llava(image_path: str) -> str:
    """
    Sends an image to LLaVA 7B running on Ollama and returns the description.

    :param image_path: Path to the image file.
    :return: Description text returned by the model.
    """
    # Convert image to base64
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # Prepare request payload
    payload = {
        "model": "llava:7b",
        "prompt": "Describe the main objects in this image which is a screenshot from a video briefly. The image may not be a real photo, can be a frame fo a special computer effect.",
        "images": [img_base64],
        "stream": False,  # single response instead of token stream
    }

    # Send request to Ollama API
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"


def get_frames(cut_path):
    frame_buffers_to_return = []
    print(f"Cut name: {cut_path}")
    cap = cv2.VideoCapture(os.path.join(CUTS_PATH, cut_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    nth_frame_to_sample = int(1 / PERCENT_FRAMES_TO_SAMPLE)

    for frame_num in range(0, total_frames, nth_frame_to_sample):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_num}")
            continue

        success, buffer = cv2.imencode(".jpg", frame)
        if success:
            print(f"read frame {frame_num}")
            show_frame(frame)
            frame_buffers_to_return.append(buffer)

    cap.release()
    return frame_buffers_to_return


def show_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis("off")  # optional: hides axes
    plt.show()


def write_frame_descriptions(cut, description, frame_index):
    cut = cut.split(".")[0]
    # if os.path.isdir(os.path.join(CUTS_DESCRIPTION_PATH, cut)):
    #     shutil.rmtree(os.path.join(CUTS_DESCRIPTION_PATH, cut))
    print(f"directory to delete : {os.path.join(CUTS_DESCRIPTION_PATH, cut)}")
    os.mkdir(os.path.join(CUTS_DESCRIPTION_PATH, cut))

    file_name = os.path.join(CUTS_DESCRIPTION_PATH, cut, f"{cut}_{frame_index}.txt")
    with open(file_name, "w") as f:
        f.write(description)


def get_activity_per_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total Frames: {total_frames}")

    activity_per_frame = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing (optional)
        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Compute pixel-wise absolute difference
            diff = cv2.absdiff(gray, prev_frame)
            diff_score = np.sum(diff) / diff.size  # Mean change
            activity_per_frame.append(diff_score)

        prev_frame = gray

    cap.release()

    # Normalize to 0-1
    activity_arr = np.array(activity_per_frame)
    if len(activity_arr) == 0:
        print("No frame difference data collected.")
        return []

    normalized = (activity_arr - activity_arr.min()) / (np.ptp(activity_arr) + 1e-5)
    return normalized


def smooth_signal_gaussian(activity, sigma=2):
    return gaussian_filter1d(activity, sigma=sigma)


def adaptive_frame_sampling(activity, min_sample_rate=0.1, max_sample_rate=1.0):
    """
    activity: normalized activity per frame (0 to 1)
    min_sample_rate: lowest sampling rate (e.g. 10% in flat areas)
    max_sample_rate: highest sampling rate (e.g. 100% near peaks)
    Returns: indices of selected frames
    """
    # Scale activity into sampling probability
    sampling_probs = min_sample_rate + activity * (max_sample_rate - min_sample_rate)

    sampled_indices = []
    for i, prob in enumerate(sampling_probs):
        if np.random.rand() < prob:
            sampled_indices.append(i)
    return sampled_indices


def plot_and_save_cut_activity(path, activity, sampled_indices=None):
    plt.figure()
    plt.plot(activity)
    plt.xlabel("Frame number")
    plt.ylabel("Activity (0=still, 1=high change)")
    plt.title("Visual Activity per Second")
    plt.grid(True)

    if sampled_indices is not None:
        for idx in sampled_indices:
            plt.axvline(x=idx, color="red", linestyle="--", alpha=0.6)

        plt.legend(["Activity", "Sampled Frames"])
    plt.savefig(path, dpi=300)


def save_frames_from_video(video_path, frame_list, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in frame_list:
        if frame_number >= total_frames or frame_number < 0:
            print(f"Skipping invalid frame number: {frame_number}")
            continue

        # Set the video to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            # Construct output file path
            filename = os.path.join(output_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved frame {frame_number} to {filename}")
        else:
            print(f"Failed to read frame {frame_number}")

    cap.release()
