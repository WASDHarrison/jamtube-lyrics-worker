#!/usr/bin/env python3
"""JamTube Lyrics worker: takes the vocals out of songs for a JamTube Lyrics server.

It connects to the server (JAMTUBE_LYRICS_URL, with its WORKER_KEY), asks for a song, takes its
vocals out the way the server would (same separator, model and settings), sends it back, and
asks for the next. Run as many as you like, anywhere: the server gives each song to a free one,
and does it itself when none is free, or when one fails or goes quiet. Nothing is kept: each
song is downloaded into this run's temporary folder and deleted once it's sent back, and the
folder goes when the worker stops. Only the model is kept (MODELS_DIR), so it's downloaded once.

    python worker.py --server https://lyrics.example.com --key <WORKER_KEY>
"""

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

# Waits before trying the server again while it can't be reached (then the last, again and again).
RETRY_SECONDS = [1, 2, 5, 10, 30]
# How long an answer to "a job?" can take: the server holds it until there's one, up to its own
# WORKER_POLL_SECONDS (25), so this is only for a server that stopped answering.
POLL_TIMEOUT_SECONDS = 120
# Downloading the song parts and sending back the instrumental (tens of MB on a slow connection).
TRANSFER_TIMEOUT_SECONDS = 600
CHUNK_BYTES = 1 << 20


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def use_ffmpeg():
    """Vocal removal writes its result with ffmpeg: the computer's own, or else the one that came
    with the worker (imageio-ffmpeg), put in .ffmpeg here as `ffmpeg` and used from there."""
    if shutil.which('ffmpeg'):
        return
    import imageio_ffmpeg
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ffmpeg')
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, 'ffmpeg.exe' if platform.system() == 'Windows' else 'ffmpeg')
    if not os.path.exists(target):
        source = imageio_ffmpeg.get_ffmpeg_exe()
        shutil.copy(source, target) if platform.system() == 'Windows' else os.symlink(source, target)
    os.environ['PATH'] = folder + os.pathsep + os.environ.get('PATH', '')


def empty(folder):
    """Everything in `folder` deleted (files, and folders, which Demucs makes)."""
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)


class Server:
    """The JamTube Lyrics server's worker API (see workers.js there)."""

    def __init__(self, url, key):
        self.url = url.rstrip('/')
        self.key = key

    def request(self, method, path, *, body=None, data=None, headers=None, timeout=60):
        """(status, response body). HTTP errors are answers too; it raises only when the server
        can't be reached."""
        headers = {'Authorization': f'Bearer {self.key}', 'User-Agent': 'jamtube-lyrics-worker', **(headers or {})}
        if body is not None:
            data = json.dumps(body).encode()
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def download(self, path, target):
        request = urllib.request.Request(self.url + path, headers={'Authorization': f'Bearer {self.key}', 'User-Agent': 'jamtube-lyrics-worker'})
        try:
            with urllib.request.urlopen(request, timeout=TRANSFER_TIMEOUT_SECONDS) as response, open(target, 'wb') as file:
                shutil.copyfileobj(response, file, CHUNK_BYTES)
            return 200
        except urllib.error.HTTPError as error:
            return error.code

    def upload(self, path, source):
        with open(source, 'rb') as file:
            return self.request('POST', path, data=file, timeout=TRANSFER_TIMEOUT_SECONDS,
                                headers={'Content-Type': 'audio/flac', 'Content-Length': str(os.path.getsize(source))})[0]


class Separator:
    """Takes the vocals out the way JamTube Lyrics does (remove_vocals in its prepare.py), with
    the model it says, into `out_dir`. The model stays loaded between songs."""

    def __init__(self, models_dir, out_dir):
        self.models_dir = models_dir
        self.loaded = None  # ((separator, model, overlap), audio_separator's Separator)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def load(self, how):
        """The model `how` says ({ separator, model, overlap }), downloaded the first time and
        loaded, if it isn't already. Demucs loads its own each time (see demucs)."""
        if how['separator'] != 'mdx':
            return
        key = (how['separator'], how['model'], how['overlap'])
        if self.loaded and self.loaded[0] == key:
            return
        from audio_separator.separator import Separator as AudioSeparator
        separator = AudioSeparator(
            output_dir=self.out_dir,
            model_file_dir=self.models_dir,
            output_format='FLAC',
            output_single_stem='Instrumental',
            mdx_params={'hop_length': 1024, 'segment_size': 256, 'overlap': how['overlap'], 'batch_size': 1,
                        'enable_denoise': False},
        )
        log(f"Loading {how['model']} (downloaded into {self.models_dir} the first time)")
        separator.load_model(model_filename=how['model'])
        self.loaded = (key, separator)

    def instrumental(self, job, source):
        """The instrumental of `source` (FLAC), as a FLAC file, removed once it's sent."""
        empty(self.out_dir)
        if job['separator'] == 'demucs':
            return self.demucs(job, source)
        if job['separator'] != 'mdx':
            raise RuntimeError(f"Unknown separator: {job['separator']}")
        self.load(job)
        for name in self.loaded[1].separate(source):
            path = name if os.path.isabs(name) else os.path.join(self.out_dir, name)
            if 'instrumental' in os.path.basename(path).lower() and os.path.exists(path):
                return path
        raise RuntimeError('Vocal removal made no instrumental.')

    def demucs(self, job, source):
        result = subprocess.run([sys.executable, '-m', 'demucs', '--two-stems', 'vocals', '-n', job['model'], '--flac', '-o', self.out_dir, source],
                                capture_output=True, text=True)
        stem = os.path.splitext(os.path.basename(source))[0]
        path = os.path.join(self.out_dir, job['model'], stem, 'no_vocals.flac')
        if result.returncode != 0 or not os.path.exists(path):
            raise RuntimeError(f"Demucs failed: {(result.stderr or '').strip()[-300:]}")
        return path


def keep_saying(server, job, stop, dropped):
    """While a job's being done: tells the server every heartbeatSeconds, and notes when it's
    said it doesn't want it anymore."""
    while not stop.wait(job['heartbeatSeconds']):
        try:
            if server.request('POST', f"/workers/jobs/{job['id']}/heartbeat", timeout=30)[0] == 410:
                dropped.set()
                return
        except OSError:
            pass  # a moment's trouble reaching it: the next one may get through


def work(server, separator, job, folder):
    """Does a job: its song is downloaded into `folder` (emptied after), its vocals taken out,
    and the instrumental sent back."""
    started = time.time()
    stop, dropped = threading.Event(), threading.Event()
    threading.Thread(target=keep_saying, args=(server, job, stop, dropped), daemon=True).start()
    os.makedirs(folder, exist_ok=True)
    try:
        source = os.path.join(folder, 'song-parts.flac')
        if server.download(f"/workers/jobs/{job['id']}/audio", source) != 200:
            log('The song was taken back before it was downloaded.')
            return
        log(f"Removing the vocals ({job['separator']}, {job['model']})")
        result = separator.instrumental(job, source)
        if dropped.is_set():
            log("Done, but it's not wanted anymore.")
            return
        status = server.upload(f"/workers/jobs/{job['id']}/result", result)
        log(f'Sent back, {time.time() - started:.0f}s in all.' if status == 204 else f"Done, but it's not wanted anymore ({status}).")
    except (Exception, KeyboardInterrupt) as error:  # anything at all, or stopped: the server gives the song to someone else
        reason = 'the worker was stopped' if isinstance(error, KeyboardInterrupt) else str(error)[:300]
        log(f"Couldn't: {reason}")
        try:
            server.request('POST', f"/workers/jobs/{job['id']}/failed", body={'error': reason}, timeout=30)
        except OSError:
            pass  # it'll notice the silence instead
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        stop.set()
        shutil.rmtree(folder, ignore_errors=True)
        empty(separator.out_dir)


def accelerator():
    """What vocal removal runs on here, in words, as audio-separator picks it: an NVIDIA GPU when
    PyTorch can use it, a Mac's GPU on Apple silicon, or else the CPU."""
    try:
        import onnxruntime
        import torch
    except ImportError:
        return 'the CPU'
    providers = onnxruntime.get_available_providers()
    if torch.cuda.is_available() and 'CUDAExecutionProvider' in providers:
        return 'an NVIDIA GPU (CUDA)'
    if torch.backends.mps.is_available() and platform.machine() == 'arm64' and 'CoreMLExecutionProvider' in providers:
        return "the Mac's GPU (Core ML)"
    return 'the CPU'


def main():
    parser = argparse.ArgumentParser(description='Takes the vocals out of songs for a JamTube Lyrics server.')
    parser.add_argument('--server', default=os.environ.get('JAMTUBE_LYRICS_URL', ''), help="The server's address (or JAMTUBE_LYRICS_URL).")
    parser.add_argument('--key', default=os.environ.get('WORKER_KEY', ''), help="The server's WORKER_KEY (or WORKER_KEY).")
    parser.add_argument('--name', default=os.environ.get('WORKER_NAME') or socket.gethostname(), help="What the server calls it (or WORKER_NAME; this computer's name).")
    parser.add_argument('--models-dir', default=os.environ.get('MODELS_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
                        help="Where the model is kept between runs (or MODELS_DIR; models, next to this).")
    args = parser.parse_args()
    if not args.server or not args.key:
        parser.error('Say which server and its WORKER_KEY: --server and --key (or JAMTUBE_LYRICS_URL and WORKER_KEY).')

    # Stopped by `docker stop`, or a cloud taking back its machine (SIGTERM): as with Ctrl+C,
    # the server's told, so a song it was working on goes elsewhere straight away.
    def stopped(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stopped)

    use_ffmpeg()
    server = Server(args.server, args.key)
    me = {'id': str(uuid.uuid4()), 'name': args.name}  # its own id, so several can share a name
    # This run's one temporary folder: each song, and its vocal removal's output. Gone when it
    # stops (a worker killed outright leaves it to the system, which clears its temporary files).
    scratch = tempfile.mkdtemp(prefix='jamtube-worker-')
    separator = Separator(args.models_dir, os.path.join(scratch, 'out'))
    log(f'"{args.name}" working for {server.url}, on {accelerator()}')
    failures = 0
    ready = False  # the server's model loaded (see Separator.load), before its first song
    try:
        while True:
            try:
                if not ready:
                    status, body = server.request('GET', '/workers/separator')
                    if status == 200:
                        try:
                            separator.load(json.loads(body))
                            log('Ready for songs.')
                        except Exception as error:  # the first song tries again
                            log(f"Couldn't get the model ready ({str(error)[:300]}): the first song will try again.")
                    ready = status < 500  # an older server (404) doesn't say: the first song loads it
                status, body = server.request('POST', '/workers/jobs/next', body=me, timeout=POLL_TIMEOUT_SECONDS)
            except OSError as error:
                wait = RETRY_SECONDS[min(failures, len(RETRY_SECONDS) - 1)]
                if failures == 0:
                    log(f"Can't reach the server ({error}): trying again until it can be.")
                failures += 1
                time.sleep(wait)
                continue
            if failures:
                log('Connected again.')
                failures = 0
            if status == 401:
                sys.exit('The server refused the key: it has to be its WORKER_KEY.')
            if status == 404:
                sys.exit("The server has no workers (WORKER_KEY isn't set there), or that isn't its address.")
            if status == 200:
                job = json.loads(body)
                work(server, separator, job, os.path.join(scratch, 'song'))
            elif status != 204:
                time.sleep(RETRY_SECONDS[-1 if status >= 500 else 0])
    except KeyboardInterrupt:
        log('Stopped.')
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    main()
