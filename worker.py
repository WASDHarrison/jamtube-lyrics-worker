#!/usr/bin/env python3
"""JamTube Lyrics worker: takes the vocals out of songs for a JamTube Lyrics server.

It connects to the server (JAMTUBE_LYRICS_URL, with its WORKER_KEY), asks for a song, takes its
vocals out the way the server would (same separator, model and settings), hears what's sung in
them (with the server's singing model and its own code for it, singing.py), sends both back, and
asks for the next: as many songs at once as the server says (its MAX_WORKER_JOBS), each in a slot
of its own, so while one has its vocals taken out, the next downloads and the last is sent back
(on an NVIDIA GPU, only as many remove vocals at once as its memory fits: GPU_MEMORY_PER_SONG_MB).
Run as many as you like, anywhere: the server gives each song to a free one, and does it itself
when none is free, or when one fails or goes quiet. Nothing is kept: each
song is downloaded into this run's temporary folder and deleted once it's sent back, and the
folder goes when the worker stops. Only the models are kept (MODELS_DIR), so they're downloaded
once, from the server.

    python worker.py --server https://lyrics.example.com --key <WORKER_KEY>
"""

import argparse
import importlib.util
import json
import os
import platform
import queue
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

# Waits before trying the server again while it can't be reached (then the last, again and
# again): soon enough to be back within seconds of a restart, when its songs are waiting.
RETRY_SECONDS = [1, 2, 3, 5]
# How long an answer to "a job?" can take: the server holds it until there's one, up to its own
# WORKER_POLL_SECONDS (25), so this is only for a server that stopped answering.
POLL_TIMEOUT_SECONDS = 120
# Downloading the song's sound and sending back the instrumental (tens of MB on a slow connection).
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
                                headers={'Content-Type': 'application/octet-stream', 'Content-Length': str(os.path.getsize(source))})[0]

    def model(self, name, folder):
        """The server's model `name`, in `folder`: downloaded from it unless it's here already at
        the same size (another size: the server's changed it). False if the server doesn't give
        it out (vocal removal then downloads its own)."""
        target = os.path.join(folder, name)
        request = urllib.request.Request(f'{self.url}/workers/models/{name}', method='HEAD',
                                         headers={'Authorization': f'Bearer {self.key}', 'User-Agent': 'jamtube-lyrics-worker'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                size = int(response.headers.get('Content-Length') or -1)
        except urllib.error.HTTPError:
            return False
        if os.path.exists(target) and os.path.getsize(target) == size:
            return True
        os.makedirs(folder, exist_ok=True)
        log(f'Downloading {name} from the server')
        if self.download(f'/workers/models/{name}', f'{target}.download') != 200:
            return False
        os.replace(f'{target}.download', target)
        return True


class Separator:
    """Takes the vocals out the way JamTube Lyrics does (remove_vocals in its prepare.py), with
    the model it says, into `out_dir`. The model stays loaded between songs."""

    def __init__(self, models_dir, out_dir):
        self.models_dir = models_dir
        self.loaded = None  # ((separator, model, overlap), audio_separator's Separator)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def load(self, how):
        """The model `how` says ({ separator, model, overlap }), loaded, if it isn't already (it's
        downloaded first, see Server.model). Demucs loads its own each time (see demucs)."""
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
        log(f"Loading {how['model']}")
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


class Hearing:
    """Hears what's sung in a song's vocals the way the server does: its singing.py and its
    singing model, both from the server (see /workers/setup there). Off when it has none."""

    def __init__(self, models_dir, code_dir):
        self.models_dir = models_dir
        self.code_dir = code_dir
        self.loaded = None  # (model name, singing module, its Ears)
        self.lock = threading.Lock()  # shared by the slots: one hears at a time

    def load(self, server, name):
        """Gets the server's singing.py and model `name` (None: it has none), if it hasn't already."""
        if not name or (self.loaded and self.loaded[0] == name):
            return
        code = os.path.join(self.code_dir, 'singing.py')
        if server.download('/workers/singing.py', code) != 200 or not server.model(name, self.models_dir):
            log("The server doesn't hear the singing here: it does that itself.")
            return
        spec = importlib.util.spec_from_file_location('singing', code)
        singing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(singing)
        log(f'Loading {name}')
        self.loaded = (name, singing, singing.Ears(os.path.join(self.models_dir, name)))

    def hear(self, job, source, instrumental, target):
        """What's sung in the song (`source` less `instrumental`), if the job says to hear it,
        saved to `target` (.npy). False if it doesn't, or can't."""
        if not job.get('hear') or not self.loaded:
            return False
        import numpy
        _, singing, ears = self.loaded
        vocals, rate = singing.vocals_to_hear(source, instrumental)
        with self.lock:
            heard = ears.hear(vocals, rate)
        numpy.save(target, heard.astype(numpy.float32))
        return True


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


def work(server, separators, hearing, job, folder, say):
    """Does a job: its song is downloaded into `folder` (emptied after), its vocals taken out
    and heard, and both sent back: what was heard first, then the instrumental, which ends it.
    Its vocals are taken out by one of `separators` (a queue shared by the slots: as many as the
    GPU's memory fits), waiting for one to be free. `say` logs, saying which slot."""
    started = time.time()
    stop, dropped = threading.Event(), threading.Event()
    threading.Thread(target=keep_saying, args=(server, job, stop, dropped), daemon=True).start()
    os.makedirs(folder, exist_ok=True)
    try:
        source = os.path.join(folder, 'song-parts.flac')
        if server.download(f"/workers/jobs/{job['id']}/audio", source) != 200:
            say('The song was taken back before it was downloaded.')
            return
        separator = separators.get()
        try:
            say(f"Removing the vocals ({job['separator']}, {job['model']})")
            result = os.path.join(folder, 'instrumental.flac')
            shutil.move(separator.instrumental(job, source), result)
        finally:
            empty(separator.out_dir)
            separators.put(separator)
        heard = os.path.join(folder, 'heard.npy')
        try:
            if hearing.hear(job, source, result, heard) and not dropped.is_set():
                say('Heard the singing')
                server.upload(f"/workers/jobs/{job['id']}/heard", heard)
        except Exception as error:  # the server hears it itself
            say(f"Couldn't hear the singing ({str(error)[:300]}): the server will.")
        if dropped.is_set():
            say("Done, but it's not wanted anymore.")
            return
        status = server.upload(f"/workers/jobs/{job['id']}/result", result)
        say(f'Sent back, {time.time() - started:.0f}s in all.' if status == 204 else f"Done, but it's not wanted anymore ({status}).")
    except Exception as error:  # anything at all: the server gives the song to someone else
        say(f"Couldn't: {str(error)[:300]}")
        give_back(server, job, str(error)[:300])
    finally:
        stop.set()
        shutil.rmtree(folder, ignore_errors=True)


def give_back(server, job, reason):
    """Tells the server this worker won't finish `job`, so it goes elsewhere straight away."""
    try:
        server.request('POST', f"/workers/jobs/{job['id']}/failed", body={'error': reason}, timeout=30)
    except OSError:
        pass  # it'll notice the silence instead


def take_songs(server, separators, hearing, folder, me, doing, say, quit):
    """One slot: asks for a song, does it (see work), and asks for the next, again and again.
    `doing` is the jobs the slots are doing (so they can be given back if the worker's stopped),
    and `quit(why)` stops the worker when the server won't have it."""
    failures = 0
    while True:
        try:
            status, body = server.request('POST', '/workers/jobs/next', body=me, timeout=POLL_TIMEOUT_SECONDS)
        except OSError as error:
            wait = RETRY_SECONDS[min(failures, len(RETRY_SECONDS) - 1)]
            if failures == 0:
                say(f"Can't reach the server ({error}): trying again until it can be.")
            failures += 1
            time.sleep(wait)
            continue
        if failures:
            say('Connected again.')
            failures = 0
        if status == 401:
            return quit('The server refused the key: it has to be its WORKER_KEY.')
        if status == 404:
            return quit("The server has no workers (WORKER_KEY isn't set there), or that isn't its address.")
        if status == 200:
            job = json.loads(body)
            doing[job['id']] = job
            try:
                work(server, separators, hearing, job, folder, say)
            finally:
                doing.pop(job['id'], None)
        elif status != 204:
            time.sleep(RETRY_SECONDS[-1 if status >= 500 else 0])


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


def gpu_memory_mb():
    """The NVIDIA GPU's memory in MB, or None without one."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else None


def fitting(slots, per_song_mb):
    """How many songs can have their vocals removed at once: one for each slot, but on an NVIDIA
    GPU, only as many as its memory fits (each loads the model, and works in GPU memory)."""
    total_mb = gpu_memory_mb()
    if total_mb is None or per_song_mb <= 0:
        return slots
    return max(1, min(slots, int(total_mb // per_song_mb)))


def main():
    parser = argparse.ArgumentParser(description='Takes the vocals out of songs for a JamTube Lyrics server.')
    parser.add_argument('--server', default=os.environ.get('JAMTUBE_LYRICS_URL', ''), help="The server's address (or JAMTUBE_LYRICS_URL).")
    parser.add_argument('--key', default=os.environ.get('WORKER_KEY', ''), help="The server's WORKER_KEY (or WORKER_KEY).")
    parser.add_argument('--name', default=os.environ.get('WORKER_NAME') or socket.gethostname(), help="What the server calls it (or WORKER_NAME; this computer's name).")
    parser.add_argument('--models-dir', default=os.environ.get('MODELS_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
                        help="Where the model is kept between runs (or MODELS_DIR; models, next to this).")
    parser.add_argument('--gpu-memory-per-song', type=int, default=int(os.environ.get('GPU_MEMORY_PER_SONG_MB') or 2500),
                        help="GPU memory one song's vocal removal takes, in MB (or GPU_MEMORY_PER_SONG_MB; 2500): on an NVIDIA GPU, only as many songs as fit have their vocals removed at once. 0: one for each slot.")
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
    # This run's one temporary folder: each slot's song, and its vocal removal's output. Gone when
    # it stops (a worker killed outright leaves it to the system, which clears its temporary files).
    scratch = tempfile.mkdtemp(prefix='jamtube-worker-')
    hearing = Hearing(args.models_dir, scratch)
    log(f'"{args.name}" working for {server.url}, on {accelerator()}')
    doing = {}  # job id -> job: the songs the slots are doing
    stop = threading.Event()
    stopped_why = []
    try:
        # How the server removes vocals and hears them, and how many songs to take at once: asked
        # until it answers (an older server doesn't say: one at a time, each loading its model).
        failures = 0
        while True:
            try:
                status, body = server.request('GET', '/workers/setup')
                if status < 500:
                    break
                problem = f'it answered {status}'
            except OSError as error:
                problem = str(error)
            if failures == 0:
                log(f"Can't reach the server ({problem}): trying again until it can be.")
            time.sleep(RETRY_SECONDS[min(failures, len(RETRY_SECONDS) - 1)])
            failures += 1
        setup = json.loads(body) if status == 200 else {}
        slots = max(1, int(setup.get('jobs') or 1))
        # The slots download and send back songs side by side, sharing the separators: as many as
        # the GPU's memory fits (each holds its model and works in GPU memory).
        removing = fitting(slots, args.gpu_memory_per_song)
        separators = queue.Queue()
        for number in range(removing):
            separators.put(Separator(args.models_dir, os.path.join(scratch, f'out-{number}')))
        if setup:
            try:
                if setup['separator'] == 'mdx':
                    server.model(setup['model'], args.models_dir)
                for separator in list(separators.queue):
                    separator.load(setup)
                hearing.load(server, setup.get('singing'))
                fits = f', removing the vocals of {removing} at once (what the GPU\'s memory fits)' if removing < slots else ''
                log(f"Ready for songs{f', {slots} at once' if slots > 1 else ''}{fits}.")
            except Exception as error:  # the first song tries again
                log(f"Couldn't get the models ready ({str(error)[:300]}): the first song will try again.")

        def quit(why):
            stopped_why.append(why)
            stop.set()
        for slot in range(1, slots + 1):
            def say(message, slot=slot):
                log(f'[{slot}] {message}' if slots > 1 else message)
            threading.Thread(target=take_songs, daemon=True,
                             args=(server, separators, hearing, os.path.join(scratch, f'song-{slot}'), me, doing, say, quit)).start()
        while not stop.wait(1):
            pass
        sys.exit(stopped_why[0])
    except KeyboardInterrupt:
        # Stopped (Ctrl+C, docker stop, a cloud taking back its machine): the server's told about
        # each song it was doing, so they go elsewhere straight away.
        songs = list(doing.values())
        for job in songs:
            give_back(server, job, 'the worker was stopped')
        log(f"Stopped: {'its song goes' if len(songs) == 1 else f'its {len(songs)} songs go'} back to the server." if songs else 'Stopped.')
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    main()
