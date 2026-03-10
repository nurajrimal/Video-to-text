import os
import json
import base64
import time
import threading
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import cv2
import anthropic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# In-memory job store: { job_id: { status, progress, log, result } }
jobs = {}

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_timestamp(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"


def image_to_base64(frame):
    """Convert OpenCV frame to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def describe_frame_with_claude(client, frame_b64, timestamp_str, frame_num):
    """Use Claude's vision to describe a video frame."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": frame_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This is frame from a video at timestamp {timestamp_str}. "
                                "Describe what is happening in this frame in 1-2 concise sentences. "
                                "Focus on the main subject, action, or content visible. "
                                "Be specific and informative. Do not mention that this is a video frame."
                            )
                        }
                    ]
                }
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"[Frame {frame_num}] Unable to describe: {str(e)}"


def get_simple_embedding(text):
    """
    Simple TF-IDF-style word vector for cosine similarity
    (used as fallback if sentence-transformers not available).
    Works well enough for deduplication.
    """
    words = text.lower().split()
    vocab = list(set(words))
    vec = np.zeros(len(vocab))
    for i, w in enumerate(vocab):
        vec[i] = words.count(w)
    return vec


def compute_cosine_similarity_pair(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    # Pad to same length
    max_len = max(len(emb1), len(emb2))
    e1 = np.pad(emb1, (0, max_len - len(emb1)))
    e2 = np.pad(emb2, (0, max_len - len(emb2)))
    if np.linalg.norm(e1) == 0 or np.linalg.norm(e2) == 0:
        return 0.0
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))


def deduplicate_descriptions(descriptions, threshold=0.85):
    """
    Remove semantically similar descriptions using cosine similarity.
    Returns list of (timestamp, text) that are sufficiently unique.
    """
    if not descriptions:
        return []

    unique = [descriptions[0]]
    embeddings = [get_simple_embedding(descriptions[0]['text'])]

    for item in descriptions[1:]:
        emb = get_simple_embedding(item['text'])
        is_duplicate = False
        for prev_emb in embeddings:
            sim = compute_cosine_similarity_pair(emb, prev_emb)
            if sim >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(item)
            embeddings.append(emb)

    return unique


def process_video_job(job_id, video_path, fps_target, similarity_threshold, api_key):
    """Background worker: extract frames → describe → deduplicate → output."""
    job = jobs[job_id]
    job['status'] = 'processing'
    job['progress'] = 0
    job['log'] = []

    def log(msg):
        job['log'].append(msg)
        print(f"[{job_id}] {msg}")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        log("Opening video file...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file.")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        log(f"Video loaded: {duration:.1f}s, {video_fps:.1f} FPS native, {total_frames} total frames")

        # Determine frame interval
        if fps_target <= 0 or fps_target > video_fps:
            fps_target = min(1.0, video_fps)
        frame_interval = int(video_fps / fps_target)
        if frame_interval < 1:
            frame_interval = 1

        frames_to_extract = list(range(0, total_frames, frame_interval))
        log(f"Extracting ~{len(frames_to_extract)} frames at {fps_target} FPS...")

        descriptions = []
        for idx, frame_num in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp_sec = frame_num / video_fps
            timestamp_str = format_timestamp(timestamp_sec)

            # Convert and describe
            frame_b64 = image_to_base64(frame)
            log(f"Describing frame {idx+1}/{len(frames_to_extract)} at [{timestamp_str}]...")

            desc = describe_frame_with_claude(client, frame_b64, timestamp_str, frame_num)
            descriptions.append({
                'timestamp': timestamp_str,
                'seconds': timestamp_sec,
                'text': desc
            })

            job['progress'] = int(((idx + 1) / len(frames_to_extract)) * 80)
            time.sleep(0.1)  # mild rate limiting

        cap.release()
        log(f"Frame extraction complete. {len(descriptions)} descriptions generated.")

        # Deduplicate
        log(f"Running semantic deduplication (threshold={similarity_threshold})...")
        unique_descriptions = deduplicate_descriptions(descriptions, threshold=similarity_threshold)
        removed = len(descriptions) - len(unique_descriptions)
        log(f"Deduplication complete. Removed {removed} similar descriptions. {len(unique_descriptions)} unique entries remain.")

        job['progress'] = 90

        # Write output
        output_filename = f"transcript_{job_id}.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("VIDEO-TO-TEXT TRANSCRIPT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated by Video2Text Converter\n")
            f.write(f"Video duration: {format_timestamp(duration)}\n")
            f.write(f"Frames analyzed: {len(descriptions)}\n")
            f.write(f"Unique descriptions: {len(unique_descriptions)}\n")
            f.write(f"Similarity threshold: {similarity_threshold}\n")
            f.write("=" * 60 + "\n\n")

            for item in unique_descriptions:
                f.write(f"[{item['timestamp']}] {item['text']}\n\n")

        job['progress'] = 100
        job['status'] = 'done'
        job['result'] = {
            'output_file': output_filename,
            'total_frames': len(descriptions),
            'unique_entries': len(unique_descriptions),
            'removed': removed,
            'duration': format_timestamp(duration),
            'descriptions': unique_descriptions
        }
        log("Done! Transcript saved.")

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        log(f"ERROR: {str(e)}")
    finally:
        # Clean up uploaded video
        try:
            os.remove(video_path)
        except:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    api_key = request.form.get('api_key', '').strip()
    fps = float(request.form.get('fps', 1.0))
    threshold = float(request.form.get('threshold', 0.85))

    if not api_key:
        return jsonify({'error': 'Anthropic API key is required'}), 400
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    filename = secure_filename(file.filename)
    job_id = str(int(time.time() * 1000))
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(save_path)

    jobs[job_id] = {'status': 'queued', 'progress': 0, 'log': [], 'result': None}

    thread = threading.Thread(
        target=process_video_job,
        args=(job_id, save_path, fps, threshold, api_key),
        daemon=True
    )
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'log': job['log'][-10:],  # last 10 log lines
        'result': job.get('result'),
        'error': job.get('error')
    })


@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
