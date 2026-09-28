from detect import detect_people
from carried_object import detect_carried_object
import cv2
import json
import glob
import os


def build_events_for_frame(image_path, camera_id, frame_ref):
    """
    Run detection + carried-object heuristic on a single frame.

    Returns:
        list of event dicts for this frame
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  WARNING: could not load {image_path}, skipping")
        return []

    people = detect_people(image_path)
    events = []
    for p in people:
        obj = detect_carried_object(img, p["bbox"])
        events.append({
            "camera_id": camera_id,
            "frame_ref": frame_ref,
            "bbox": list(p["bbox"]),
            "object_class": obj,
            "confidence": p["confidence"]
        })
    return events


def build_events_for_camera(camera_folder, camera_id, limit=None):
    """
    Run detection + carried-object heuristic across every frame
    in a camera's image folder.

    Args:
        camera_folder (str): path to the camera's image folder, e.g. "../data/Image_subsets/C1"
        camera_id (str): label to store in each event, e.g. "cam_0"
        limit (int, optional): only process the first N frames (useful for testing)

    Returns:
        list of all event dicts across all processed frames
    """
    image_paths = sorted(glob.glob(os.path.join(camera_folder, "*.png")))
    if limit:
        image_paths = image_paths[:limit]

    all_events = []
    for i, image_path in enumerate(image_paths):
        frame_ref = os.path.join(os.path.basename(camera_folder), os.path.basename(image_path))
        print(f"[{i+1}/{len(image_paths)}] Processing {frame_ref}...")
        events = build_events_for_frame(image_path, camera_id, frame_ref)
        all_events.extend(events)

    return all_events


if __name__ == "__main__":
    # Adjust these to match your actual folder structure / camera mapping
    CAMERAS = [
        {"folder": "../data/Image_subsets/C1", "camera_id": "cam_0"},
        # Add more cameras here as needed, e.g.:
        # {"folder": "../data/Image_subsets/C2", "camera_id": "cam_1"},
    ]

    # Set a limit while testing (e.g. limit=10) to avoid processing
    # hundreds of frames on your first run. Remove or set to None
    # once you're confident it works.
    LIMIT = 3

    for cam in CAMERAS:
        print(f"\n=== Processing {cam['camera_id']} ({cam['folder']}) ===")
        events = build_events_for_camera(cam["folder"], cam["camera_id"], limit=LIMIT)

        output_path = f"../output/events_{cam['camera_id']}.json"
        with open(output_path, "w") as f:
            json.dump(events, f, indent=2)

        print(f"Saved {len(events)} events to {output_path}")