#!/usr/bin/env python3
"""JamTube Lyrics worker: takes the vocals out of songs for a JamTube Lyrics server.

It connects to the server (JAMTUBE_LYRICS_URL, with its WORKER_KEY), asks for a song, takes its
vocals out the way the server would (same separator, model and settings), hears what's sung in
them (with the server's singing model and its own code for it, singing.py), sends both back, and
asks for the next: as many songs at once as the server says (its MAX_WORKER_JOBS), each in a slot
of its own, so while one has its vocals taken out, the next downloads and the last is sent back
(on an NVIDIA GPU, only as many remove vocals at once as its memory fits: GPU_MEMORY_PER_SONG_MB).
Run as many as you like, anywhere: the server gives each song to a free one, and does it itself
when none is free, or when one fails or goes quiet. With what the server's prepare.py needs
installed (see requirements-gpu.txt), it makes whole songs too, when the server gives it one (its
WORKER_SONGS): it runs the server's prepare.py here, which downloads the song (through the
server's downloaders, then its residential proxy, then from here), finds its lyrics, matches it,
has its vocals removed and heard by this worker's GPU, and encodes it; what's made is sent to the
server, which only keeps it. Nothing is kept: each
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
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

# PyTorch's own GPU kernels only: not the ones it writes with Triton to stand in for some of them
# (torch._native), which Triton builds with a C compiler the first time, and there's none here
# ("Failed to find C compiler", Whisper's matrix multiplies). Before anything loads PyTorch.
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
# Hugging Face's nag about unauthenticated downloads (Whisper is public, and downloaded once): not
# a problem, so not in the log.
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')

# Vocal removal's progress bars redraw themselves in place, which only works in a live terminal:
# in a saved log (a cloud's log viewer, docker logs) each redraw is lines of blank space. So
# they're only shown in a terminal; its own lines (and the separator's) are logged either way.
if not sys.stderr.isatty():
    os.environ.setdefault('TQDM_DISABLE', '1')

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


# What the server's prepare.py needs to make a whole song here (see make_song): its Python
# packages (as they're imported), ffmpeg and ffprobe, and a JavaScript runtime for yt-dlp (Deno,
# from requirements-gpu.txt, or Node or Bun).
SONG_MODULES = ['yt_dlp', 'yt_dlp_ejs', 'certifi', 'numpy', 'scipy', 'librosa', 'soundfile', 'audioread', 'audio_separator',
                'unidecode', 'pypinyin', 'ToJyutping', 'taibun', 'opencc', 'cutlet', 'unidic_lite', 'korean_romanizer',
                'torch', 'phonemizer', 'espeakng_loader', 'transformers']
SONG_CODE = ['prepare.py', 'singing.py', 'gemini.py']  # the server's, got for each song when they change
LINES_SECONDS = 0.5  # what prepare.py says is sent to the server this often, together


def tools_path():
    """PATH with this Python's own scripts folder first (a venv's: Deno, installed with pip, is
    there), for prepare.py."""
    return os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')


def songs_missing():
    """What this worker lacks to make whole songs (see make_song), in words, or None."""
    missing = [module for module in SONG_MODULES if not importlib.util.find_spec(module)]
    tools = [tool for tool in ('ffmpeg', 'ffprobe') if not shutil.which(tool)]
    if not any(shutil.which(runtime, path=tools_path()) for runtime in ('deno', 'node', 'bun')):
        tools.append('a JavaScript runtime (Deno, Node or Bun)')
    if missing:
        return f"the Python packages {', '.join(missing)} (see requirements-gpu.txt)"
    if tools:
        return ' and '.join(tools)
    return None


def use_ffmpeg():
    """Vocal removal writes its result with ffmpeg: the computer's own, or else the one that came
    with the worker (imageio-ffmpeg), put in .ffmpeg here as `ffmpeg` and used from there. A
    whole song needs ffprobe too: without the computer's, static-ffmpeg's (downloaded once)."""
    if not (shutil.which('ffmpeg') and shutil.which('ffprobe')) and importlib.util.find_spec('static_ffmpeg'):
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths(weak=True)  # (only what isn't here already)
        except Exception as error:  # vocal removal does without it
            log(f"Couldn't get ffmpeg and ffprobe for whole songs ({str(error)[:200]})")
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
    singing model, both from the server (see /workers/setup there), and Whisper when a job asks
    for it (see Whisper in singing.py). Off when it has none."""

    def __init__(self, models_dir, code_dir):
        self.models_dir = models_dir
        self.code_dir = code_dir
        self.loaded = None  # (model name, singing module, its Ears)
        self.whisper = None  # (its model, singing.Whisper or why it can't be had)
        self.lock = threading.Lock()  # shared by the slots: one hears with the singing model at a time
        # Whisper has its own: a song read with it (slower) doesn't hold up the next one being
        # heard with the singing model.
        self.whisper_lock = threading.Lock()
        # Songs sent back, to read with Whisper (see read_later): [(server, job, folder, say)],
        # read one at a time, songs the server's waiting for first.
        self.later = []
        self.later_ready = threading.Condition()
        self.later_thread = None

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

    def whisper_for(self, model):
        """Whisper (`model`), loaded the first time (downloaded the first time ever). Raises why
        it can't be."""
        if not self.whisper or self.whisper[0] != model:
            log(f'Loading Whisper ({model}; downloaded the first time)')
            try:
                self.whisper = (model, self.loaded[1].Whisper(self.models_dir, model))
            except Exception as error:
                self.whisper = (model, error)
                log(f"Couldn't load Whisper ({error}): the singing model alone hears songs here")
        if isinstance(self.whisper[1], Exception):
            raise self.whisper[1]
        return self.whisper[1]

    def get_ready(self, model):
        """Whisper loaded as soon as the worker starts (`model`: the server's, see /workers/setup),
        in the background, so the first song doesn't wait for it to load."""
        if not self.loaded or not model:
            return

        def load():
            with self.whisper_lock:
                try:
                    self.whisper_for(model)
                except Exception:
                    pass  # said by whisper_for; the first song that asks says so again
                finally:
                    free_gpu_memory()
        threading.Thread(target=load, daemon=True).start()

    def hear(self, job, source, instrumental, target):
        """What's sung in the song (`source` less `instrumental`), if the job says to hear it,
        saved to `target` (see save_heard in singing.py). False if it doesn't, or can't."""
        if not job.get('hear') or not self.loaded:
            return False
        _, singing, ears = self.loaded
        vocals, rate = singing.vocals_to_hear(source, instrumental)
        with self.lock:
            heard = ears.hear(vocals, rate)
        singing.save_heard(target, heard)
        return True

    def wants_whisper(self, job):
        """Whether the job asks for Whisper too, and the singing's heard here."""
        return bool(job.get('hear') and self.loaded and job.get('whisper'))

    def read(self, job, source, instrumental, target, news):
        """What Whisper reads sung in the song (see wants_whisper), saved to `target` (see
        save_whisper in singing.py). Raises why it couldn't. `news` says how it's going (to the
        log here and the server's)."""
        singing = self.loaded[1]
        with self.whisper_lock:
            if not self.whisper or self.whisper[0] != job['whisper']:
                news(f"Loading Whisper ({job['whisper']}) first{gpu_memory_note()}")
            started = time.time()
            whisper = self.whisper_for(job['whisper'])
            if time.time() - started > 1:
                news(f'Loaded it in {time.time() - started:.0f}s{gpu_memory_note()}')
            news(f'Reading it with Whisper{gpu_memory_note()}')
            reset_gpu_peak()
            try:
                read = whisper.hear(*singing.vocals_to_hear(source, instrumental))
            except Exception as error:
                raise RuntimeError(f'{singing.first_line(error)}{gpu_memory_note(peak=True)}') from error
            finally:
                free_gpu_memory()  # what it held for this song, back for vocal removal
        news(f"Read {len(read['words'])} words in {read['seconds']:.0f}s{gpu_memory_note(peak=True)}")
        singing.save_whisper(target, read)

    def read_later(self, server, job, folder, say):
        """Reads a song sent back with Whisper (see read_sent), in turn: one at a time, a song the
        server's waiting for (the singing model wasn't sure of its lyrics) before songs it may not
        need, and songs it no longer needs (made without it) not at all."""
        with self.later_ready:
            self.later.append((server, job, folder, say))
            self.later_ready.notify()
            if not self.later_thread or not self.later_thread.is_alive():
                self.later_thread = threading.Thread(target=self.read_in_turn, daemon=True)
                self.later_thread.start()

    def read_in_turn(self):
        while True:
            with self.later_ready:
                while not self.later:
                    self.later_ready.wait()
                waiting = list(self.later)
            # Each asked whether it's still needed (none asked: a moment's trouble, asked again).
            answers = [(entry, still_wanted(entry[0], entry[1])) for entry in waiting]
            for entry, answer in answers:
                if answer == 'no':
                    entry[3]("Not read with Whisper: the song's made without it.")
                    self.drop_later(entry)
            choice = next((entry for entry, answer in answers if answer == 'wanted'), None) \
                or next((entry for entry, answer in answers if answer == 'maybe'), None)
            if not choice:
                time.sleep(RETRY_SECONDS[0])
                continue
            self.drop_later(choice, remove=False)
            server, job, folder, say = choice
            read_sent(server, self, job, folder, say)

    def drop_later(self, entry, remove=True):
        with self.later_ready:
            if entry in self.later:
                self.later.remove(entry)
        if remove:
            shutil.rmtree(entry[2], ignore_errors=True)


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
    Then, if it asks, it's read by Whisper too, in the background (see read_sent), sent
    after it (the song only waits for that if the singing model wasn't sure of its lyrics).
    Its vocals are taken out by one of `separators` (a queue shared by the slots: as many as the
    GPU's memory fits), waiting for one to be free. `say` logs, saying which slot."""
    started = time.time()
    tell = news_for(server, job, say)  # in the server's log too, with the song
    stop, dropped = threading.Event(), threading.Event()
    threading.Thread(target=keep_saying, args=(server, job, stop, dropped), daemon=True).start()
    os.makedirs(folder, exist_ok=True)
    try:
        source = os.path.join(folder, 'song-parts.flac')
        if server.download(f"/workers/jobs/{job['id']}/audio", source) != 200:
            say('The song was taken back before it was downloaded.')
            return
        separator = separators.get()
        if dropped.is_set():  # taken back while it waited (for a room's song, say): not started
            separators.put(separator)
            say("Not wanted anymore: it's not started.")
            return
        try:
            say(f"Removing the vocals ({job['separator']}, {job['model']})")
            result = os.path.join(folder, 'instrumental.flac')
            removing = time.time()
            reset_gpu_peak()
            shutil.move(separator.instrumental(job, source), result)
            tell(f'Took the vocals out in {time.time() - removing:.0f}s{gpu_memory_note(peak=True)}')
        finally:
            empty(separator.out_dir)
            separators.put(separator)
        heard = os.path.join(folder, 'heard.npy')
        try:
            if hearing.hear(job, source, result, heard) and not dropped.is_set():
                say('Heard the singing')
                server.upload(f"/workers/jobs/{job['id']}/heard", heard)
        except Exception as error:  # the server hears it itself
            tell(f"Couldn't hear the singing ({str(error)[:300]}): the server will.")
        if dropped.is_set():
            say("Done, but it's not wanted anymore.")
            return
        status = server.upload(f"/workers/jobs/{job['id']}/result", result)
        say(f'Sent back, {time.time() - started:.0f}s in all.' if status == 204 else f"Done, but it's not wanted anymore ({status}).")
        if status == 204 and hearing.wants_whisper(job):
            # Read with Whisper in the background, from a folder of its own: the slot takes its
            # next song straight away (a room's song isn't kept waiting for it).
            later = tempfile.mkdtemp(prefix='whisper-', dir=os.path.dirname(folder))
            shutil.move(source, os.path.join(later, 'song-parts.flac'))
            shutil.move(result, os.path.join(later, 'instrumental.flac'))
            hearing.read_later(server, job, later, say)
    except Exception as error:  # anything at all: the server gives the song to someone else
        tell(f"Couldn't: {str(error)[:300]}{gpu_memory_note()}")
        give_back(server, job, str(error)[:300])
    finally:
        stop.set()
        shutil.rmtree(folder, ignore_errors=True)


def news_for(server, job, say):
    """Says `text` here (`say`) and in the server's log, with the job's song (see /news there:
    while Whisper reads it, it also keeps the song waiting for it)."""
    def news(text):
        say(text)
        try:
            server.request('POST', f"/workers/jobs/{job['id']}/news", body={'text': text}, timeout=15)
        except OSError:
            pass  # just news
    return news


def still_wanted(server, job):
    """Whether the server still wants what Whisper reads in a song sent back: 'wanted' (the
    song's waiting for it), 'maybe' (not yet: the singing model may be sure), 'no' (the song's
    made), or None (the server can't be reached just now)."""
    try:
        status, body = server.request('GET', f"/workers/jobs/{job['id']}/whisper", timeout=15)
    except OSError:
        return None
    if status == 410:
        return 'no'
    if status != 200:
        return 'wanted'
    try:
        return 'wanted' if json.loads(body).get('wanted') else 'maybe'
    except ValueError:
        return 'wanted'


def read_sent(server, hearing, job, folder, say):
    """What Whisper reads in a song sent back (its sound and instrumental in `folder`, which goes
    after), sent after it, or why it couldn't, so it isn't waited for."""
    news = news_for(server, job, say)
    try:
        target = os.path.join(folder, 'whisper.json')
        try:
            hearing.read(job, os.path.join(folder, 'song-parts.flac'), os.path.join(folder, 'instrumental.flac'), target, news)
            read = True
        except Exception as error:  # the song does without it: told so, and why, so it doesn't wait for it
            why = str(error).strip()[:600] or type(error).__name__
            news(f"Couldn't read it with Whisper: {why}")
            hearing.loaded[1].save_whisper(target, {'error': why})
            read = False
        started = time.time()
        status = server.upload(f"/workers/jobs/{job['id']}/whisper", target)
        if read:
            say(f'Sent what Whisper read ({time.time() - started:.1f}s).' if status == 204
                else "Read with Whisper, but it's not needed anymore (the song's made).")
    except Exception as error:  # the server stops waiting for it on its own
        say(f"Couldn't send what Whisper read: {str(error)[:300]}")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def give_back(server, job, reason):
    """Tells the server this worker won't finish `job`, so it goes elsewhere straight away."""
    try:
        server.request('POST', f"/workers/jobs/{job['id']}/failed", body={'error': reason}, timeout=30)
    except OSError:
        pass  # it'll notice the silence instead


class Lines:
    """What prepare.py says, sent to the server in order (see /lines there), every LINES_SECONDS
    together. `dropped` is set when the server doesn't want the song anymore."""

    def __init__(self, server, job, dropped):
        self.server, self.job, self.dropped = server, job, dropped
        self.pending = []
        self.lock = threading.Lock()  # around `pending`
        self.sending = threading.Lock()  # one send at a time, so they arrive in order
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.keep_sending, daemon=True)
        self.thread.start()

    def add(self, stream, text):
        with self.lock:
            self.pending.append([stream, text])

    def keep_sending(self):
        while not self.done.wait(LINES_SECONDS):
            self.send()

    def send(self, tries=1):
        """Sends what's waiting. Raises OSError if the server can't be reached `tries` times."""
        with self.sending:
            with self.lock:
                lines, self.pending = self.pending, []
            if not lines:
                return
            for attempt in range(tries):
                try:
                    status, _ = self.server.request('POST', f"/workers/jobs/{self.job['id']}/lines", body={'lines': lines}, timeout=30)
                    if status == 410:
                        self.dropped.set()
                    return
                except OSError:
                    if attempt == tries - 1:
                        with self.lock:
                            self.pending[:0] = lines  # sent with the next
                        raise
                    time.sleep(RETRY_SECONDS[min(attempt, len(RETRY_SECONDS) - 1)])

    def close(self):
        """Stops sending by itself, and sends the rest (trying for a while)."""
        self.done.set()
        self.thread.join()
        self.send(tries=10)


class SongCode:
    """The server's prepare.py, and what it loads (SONG_CODE), for each `code` it says (see
    /workers/code there): got once, kept while the worker runs."""

    def __init__(self, folder):
        self.folder = folder
        self.lock = threading.Lock()

    def get(self, server, code):
        target = os.path.join(self.folder, re.sub(r'[^\w-]', '', code) or 'code')
        with self.lock:
            if not os.path.exists(os.path.join(target, 'ready')):
                os.makedirs(target, exist_ok=True)
                for name in SONG_CODE:
                    if server.download(f'/workers/code/{name}', os.path.join(target, name)) != 200:
                        raise RuntimeError(f"Couldn't get the server's {name}.")
                open(os.path.join(target, 'ready'), 'w').close()
        return target


def custom_lyrics(server, job, cache, folder):
    """The server's custom lyrics files (the job's `lyrics`), in `folder`, as they are there (with
    when they were changed, which the server compares). Each is downloaded once, into `cache`."""
    os.makedirs(folder, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    for entry in job.get('lyrics') or []:
        name, modified = str(entry.get('name', '')), float(entry.get('modified') or 0)
        if not name or name.startswith('.') or '/' in name or '\\' in name:
            continue
        kept = os.path.join(cache, f"{uuid.uuid5(uuid.NAMESPACE_URL, f'{name} {modified}')}")
        if not os.path.exists(kept):
            if server.download(f"/workers/jobs/{job['id']}/lyrics/{urllib.parse.quote(name)}", f'{kept}.download') != 200:
                continue  # (changed there since: the server makes the song again, as it would)
            os.replace(f'{kept}.download', kept)
            os.utime(kept, (modified / 1000, modified / 1000))
        shutil.copy2(kept, os.path.join(folder, name))


def send_made(server, job, out, sent):
    """The files prepare.py made so far (in `out`), sent to the server: the ones not sent yet
    (`sent`: name -> (size, time), as they were), manifest.json last."""
    names = sorted((name for name in os.listdir(out) if os.path.isfile(os.path.join(out, name))
                    and not name.startswith('.') and not name.endswith('.tmp')), key=lambda name: name == 'manifest.json')
    for name in names:
        path = os.path.join(out, name)
        stamp = (os.path.getsize(path), os.path.getmtime(path))
        if sent.get(name) == stamp:
            continue
        status = server.upload(f"/workers/jobs/{job['id']}/files/{urllib.parse.quote(name)}", path)
        if status == 410:
            return False
        if status != 204:
            raise RuntimeError(f"The server didn't take {name} ({status}).")
        sent[name] = stamp
    return True


def make_song(server, separators, hearing, job, folder, say, settings):
    """Makes a whole song (a job of kind 'song', see sendSong in the server's workers.js): runs the
    server's prepare.py here with the job's `args` and `env`, saying what it says to the server
    (see Lines), and sending what it makes (see send_made) before saying each version's ready.
    Its slow part is done by this worker, as songs sent to it are (see answer_heavy): the vocals
    removed by one of `separators`, heard, and read with Whisper if prepare.py asks. If anything
    goes wrong here, the song goes back to the server, which has it made again elsewhere."""
    started = time.time()
    tell = news_for(server, job, say)
    stop, dropped = threading.Event(), threading.Event()
    threading.Thread(target=keep_saying, args=(server, job, stop, dropped), daemon=True).start()
    os.makedirs(folder, exist_ok=True)
    process, lines = None, None
    try:
        code_dir = settings['code'].get(server, job['code'])
        lyrics = os.path.join(folder, 'lyrics')
        custom_lyrics(server, job, settings['lyrics'], lyrics)
        out = os.path.join(folder, 'song')
        threads = str(settings['threads'])
        command = [sys.executable, os.path.join(code_dir, 'prepare.py'), *job['args'],
                   f'--out={out}', f"--models-dir={settings['models']}", f'--custom-lyrics-dir={lyrics}',
                   f'--downloader-url={server.url}/downloaders/ask', f'--threads={threads}', '--heavy-gate']
        env = {**os.environ, **{key: str(value) for key, value in (job.get('env') or {}).items()},
               'JAMTUBE_DOWNLOADER_TOKEN': server.key, 'PYTHONUNBUFFERED': '1', 'PATH': tools_path(),
               'OMP_NUM_THREADS': threads, 'MKL_NUM_THREADS': threads,
               # This worker's GPU is its own (its models stay loaded there): prepare.py's slow
               # part is done by it (see answer_heavy), and what's left there runs on the CPU.
               'CUDA_VISIBLE_DEVICES': ''}
        say(f"Making {job.get('videoId')}")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=env, cwd=folder, text=True, encoding='utf-8', errors='replace', bufsize=1)
        settings['making'].add(process)
        lines = Lines(server, job, dropped)

        def read_log():
            for line in process.stderr:
                lines.add('err', line.rstrip('\n'))
        log_thread = threading.Thread(target=read_log, daemon=True)
        log_thread.start()

        def end_when_dropped():
            while process.poll() is None:
                if dropped.wait(1):
                    process.kill()
                    return
        threading.Thread(target=end_when_dropped, daemon=True).start()

        heavy, sent = None, {}
        wanted, ended = threading.Event(), threading.Event()  # (see read_for_song)
        for line in process.stdout:
            text = line.rstrip('\n')
            try:
                update = json.loads(text)
            except ValueError:
                update = None
            if not isinstance(update, dict):
                lines.add('out', text)
                continue
            if update.get('heavy') == 'wait':
                answer, heavy = answer_heavy(update, separators, hearing, folder, tell, settings['name'])
                process.stdin.write(f'{answer}\n')
                process.stdin.flush()
                if heavy and heavy[2]:
                    threading.Thread(target=read_for_song, args=(hearing, *heavy, tell, wanted, ended), daemon=True).start()
            elif update.get('heavy') == 'done':
                pass
            elif update.get('whisper') == 'wanted':
                wanted.set()
            else:
                if update.get('ready'):
                    lines.send(tries=10)  # what it said before, first
                    if not send_made(server, job, out, sent):
                        dropped.set()
                lines.add('out', text)
                if update.get('ready'):
                    lines.send(tries=10)
        code = process.wait()
        ended.set()
        log_thread.join(timeout=10)
        lines.close()
        if dropped.is_set():
            say("Stopped: it's not wanted anymore.")
            return
        server.request('POST', f"/workers/jobs/{job['id']}/exit", body={'code': code}, timeout=60)
        say(f'Made it in {time.time() - started:.0f}s.' if code == 0 else f'prepare.py ended with {code}, in {time.time() - started:.0f}s.')
    except Exception as error:  # anything at all: the server has it made elsewhere
        if process and process.poll() is None:
            process.kill()
        if lines:
            try:
                lines.close()
            except OSError:
                pass
        tell(f"Couldn't make it: {str(error)[:300]}{gpu_memory_note()}")
        give_back(server, job, str(error)[:300])
    finally:
        stop.set()
        if 'ended' in locals():
            ended.set()
        if process:
            if process.poll() is None:
                process.kill()
            settings['making'].discard(process)
        shutil.rmtree(folder, ignore_errors=True)


def answer_heavy(offer, separators, hearing, folder, tell, name):
    """prepare.py's slow part (it said {"heavy": "wait", audio, separator, model, overlap, hear,
    whisper}, see take_out_vocals there), done here as the server's songs are (see work): its
    answer ({file, heard, whisper, worker, seconds}, or "go": it does it itself, with nothing to
    take out), and (offer, instrumental, where Whisper's read goes, or None) for read_for_song."""
    if not offer.get('audio'):
        return 'go', None
    started = time.time()
    separator = separators.get()
    try:
        result = os.path.join(folder, 'instrumental.flac')
        reset_gpu_peak()
        shutil.move(separator.instrumental(offer, offer['audio']), result)
        tell(f'Took the vocals out in {time.time() - started:.0f}s{gpu_memory_note(peak=True)}')
    finally:
        empty(separator.out_dir)
        separators.put(separator)
    heard = os.path.join(folder, 'heard.npy')
    try:
        heard = heard if hearing.hear(offer, offer['audio'], result, heard) else None
    except Exception as error:  # prepare.py hears it itself
        tell(f"Couldn't hear the singing ({str(error)[:300]}): it's heard on the CPU.")
        heard = None
    whisper = os.path.join(folder, 'whisper.json') if hearing.wants_whisper(offer) else None
    answer = {'file': result, 'heard': heard, 'whisper': whisper, 'worker': name, 'seconds': time.time() - started}
    return json.dumps(answer), (offer, result, whisper)


def read_for_song(hearing, offer, instrumental, target, tell, wanted, ended):
    """What Whisper reads in a song being made here, saved where prepare.py waits for it, if it
    asks (the singing model wasn't sure of its lyrics: `wanted`), all at once, so it's never read
    half written, with how it's going touching `target`.alive (prepare.py waits while it does).
    Read as soon as Whisper's free, before it's asked (as songs sent back are, see read_sent),
    so it's mostly read by then; while Whisper's reading another song, only once it's asked; and
    not at all once the song's made (`ended`) without it."""
    while not wanted.is_set() and hearing.whisper_lock.locked():
        if ended.wait(0.5):
            return
    if ended.is_set():
        return

    def news(text):
        tell(text)
        open(f'{target}.alive', 'w').close()
    part = f'{target}.part'
    try:
        try:
            hearing.read(offer, offer['audio'], instrumental, part, news)
        except Exception as error:  # it lines the lyrics up without it
            why = str(error).strip()[:600] or type(error).__name__
            if ended.is_set():
                return  # (made meanwhile, its files gone: nothing to say)
            news(f"Couldn't read it with Whisper: {why}")
            hearing.loaded[1].save_whisper(part, {'error': why})
        os.replace(part, target)
    except OSError:
        pass  # the song's made, and its folder gone


def take_songs(server, separators, hearing, folder, me, doing, say, quit, outdated, settings):
    """One slot: asks for a song, does it (see work), and asks for the next, again and again.
    `doing` is the jobs the slots are doing (so they can be given back if the worker's stopped),
    `quit(why)` stops the worker when the server won't have it, and `outdated()` says the server
    was updated (this slot then stops: see update)."""
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
        if status == 409:
            return outdated()
        if status == 200:
            job = json.loads(body)
            doing[job['id']] = job
            try:
                if job.get('kind') == 'song':
                    make_song(server, separators, hearing, job, folder, say, settings)
                else:
                    work(server, separators, hearing, job, folder, say)
            finally:
                doing.pop(job['id'], None)
        elif status != 204:
            time.sleep(RETRY_SECONDS[-1 if status >= 500 else 0])


def same_file(a, b):
    try:
        with open(a, 'rb') as first, open(b, 'rb') as second:
            return first.read() == second.read()
    except OSError:
        return False


def pip(action):
    """The command that installs (or uninstalls) packages for this worker's Python: run.sh's and
    run.ps1's uv when it has it, or else pip."""
    here = os.path.dirname(os.path.abspath(__file__))
    uv = os.path.join(here, '.uv', 'bin', 'uv.exe' if os.name == 'nt' else 'uv')
    if os.path.exists(uv):
        return [uv, 'pip', action, '--python', sys.executable, *(['--index-strategy', 'unsafe-best-match'] if action == 'install' else [])]
    return [sys.executable, '-m', 'pip', action, *(['--no-cache-dir'] if action == 'install' else ['-y'])]


# Started by run.sh or run.ps1 (they say so in JAMTUBE_WORKER_RUNNER), it exits with UPDATED when
# the server's updated, and they install what the update needs, with nothing loaded (Windows can't
# replace PyTorch's files while they are), and start it again. Otherwise (a container, by hand) it
# installs it itself and starts itself again.
UPDATED = 75
RESTARTED_BY_RUNNER = bool(os.environ.get('JAMTUBE_WORKER_RUNNER'))
UPDATE_TRIES = 60  # the update's files, asked for every RETRY_SECONDS[-1] (the server may be restarting)

# With an NVIDIA GPU, uv installs PyTorch's CUDA build for this computer's driver (as run.sh and
# run.ps1 do), in place of any CPU-only one: PyPI's for Windows has no CUDA.
CUDA_TORCH = ['--torch-backend', 'auto', '--reinstall-package', 'torch', '--reinstall-package', 'torchvision']


def torch_has_cuda():
    """Whether the PyTorch installed here is a CUDA build (None: no PyTorch), without loading it
    (Windows can't replace a loaded one's files): its version.py says which CUDA, or None."""
    spec = importlib.util.find_spec('torch')
    if not spec or not spec.submodule_search_locations:
        return None
    try:
        with open(os.path.join(spec.submodule_search_locations[0], 'version.py')) as file:
            found = re.search(r'^cuda\b[^=]*=\s*(.+)$', file.read(), re.MULTILINE)
    except OSError:
        return None
    return bool(found) and found.group(1).strip() != 'None'


def repair_torch():
    """With an NVIDIA GPU: a CPU-only PyTorch (set up before run.ps1 asked for the CUDA one) means
    vocal removal on the CPU. Puts the CUDA build for this driver in its place, with uv. Before
    PyTorch is loaded."""
    command = pip('install')
    if torch_has_cuda() is not False or command[0] == sys.executable:  # (pip has no --torch-backend)
        return
    log("PyTorch here is its CPU-only build, so vocal removal would be on the CPU: installing its CUDA build (a few minutes, once)")
    subprocess.run([*command, *CUDA_TORCH, 'torch', 'torchvision'], check=True)


def repair_onnxruntime():
    """With an NVIDIA GPU: ONNX Runtime's CPU-only package, installed beside its GPU one (another
    package asking for it), takes its place, and vocal removal runs on the CPU. Takes it out, and
    puts the GPU one back as it was (they share their files). Before ONNX Runtime is loaded."""
    from importlib import metadata
    try:
        gpu = metadata.version('onnxruntime-gpu')
        metadata.version('onnxruntime')
    except metadata.PackageNotFoundError:
        return
    log("ONNX Runtime's CPU-only package is installed beside its GPU one, so vocal removal would be on the CPU: taking it out")
    subprocess.run([*pip('uninstall'), 'onnxruntime'], check=True)
    subprocess.run([*pip('install'), '--force-reinstall', '--no-deps', f'onnxruntime-gpu=={gpu}'], check=True)


def update(server, scratch):
    """Starts again with the server's update (its songs are all done): its files (worker.py, and
    what it needs installed, when they changed: not in a git checkout, which is updated with git),
    then this worker again as it was started, which gets the rest (models, singing.py, setup)."""
    here = os.path.dirname(os.path.abspath(__file__))
    kind = 'gpu' if shutil.which('nvidia-smi') else 'cpu'
    if os.path.isdir(os.path.join(here, '.git')):
        log('This is a git checkout: its own files are updated with git pull.')
    else:
        got = {}
        for name in ('worker.py', f'requirements-{kind}.txt'):
            for _ in range(UPDATE_TRIES):
                try:
                    status = server.download(f'/workers/files/{name}', os.path.join(scratch, name))
                except OSError as error:  # the server restarting with its update
                    status = error
                if status == 200:
                    got[name] = os.path.join(scratch, name)
                    break
                if status == 404:  # (it hasn't got it: no WORKER_FILES_DIR)
                    break
                log(f"Couldn't get the update's {name} ({status}): trying again in {RETRY_SECONDS[-1]}s")
                time.sleep(RETRY_SECONDS[-1])
        requirements, mine = got.get(f'requirements-{kind}.txt'), os.path.join(here, f'requirements-{kind}.txt')
        if requirements and not same_file(requirements, mine) and RESTARTED_BY_RUNNER:
            shutil.copy(requirements, mine)  # run.ps1 installs it, with nothing loaded
        elif requirements and not same_file(requirements, mine):
            log(f'Installing what the update needs ({kind})')
            command = pip('install')
            cuda = CUDA_TORCH if kind == 'gpu' and command[0] != sys.executable else []
            if subprocess.run([*command, *cuda, '-r', requirements]).returncode == 0:
                shutil.copy(requirements, mine)
                if os.path.isdir(os.path.join(here, '.venv')):
                    shutil.copy(requirements, os.path.join(here, '.venv', 'installed'))  # so run.sh doesn't again
            else:  # (what it can do with what's installed, it does: see songs_missing)
                log("Couldn't install what the update needs: starting with what's installed, trying again at the next update")
        if 'worker.py' in got and not same_file(got['worker.py'], os.path.abspath(__file__)):
            os.replace(got['worker.py'], os.path.abspath(__file__))
    log('Starting again with the update.')
    shutil.rmtree(scratch, ignore_errors=True)
    if RESTARTED_BY_RUNNER:
        sys.exit(UPDATED)
    command = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    if os.name == 'nt':  # Windows' execv doesn't quote: a folder with a space ("Harrison Chen") broke in two
        sys.exit(subprocess.run(command).returncode)
    os.execv(sys.executable, command)


def accelerator():
    """What vocal removal runs on here, in words, as audio-separator picks it: an NVIDIA GPU when
    PyTorch can use it, a Mac's GPU on Apple silicon, or else the CPU."""
    try:
        # PyTorch first: its CUDA build brings the CUDA libraries (cuBLAS, cuDNN, ...) that ONNX
        # Runtime's GPU build looks for, and loads them as it's imported.
        import torch
        import onnxruntime
    except ImportError:
        return 'the CPU'
    providers = onnxruntime.get_available_providers()
    if torch.cuda.is_available() and 'CUDAExecutionProvider' in providers:
        return 'an NVIDIA GPU (CUDA)'
    if torch.backends.mps.is_available() and platform.machine() == 'arm64' and 'CoreMLExecutionProvider' in providers:
        return "the Mac's GPU (Core ML)"
    return 'the CPU'


def free_gpu_memory():
    """PyTorch's GPU memory that's no longer used, given back (it keeps it otherwise)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def reset_gpu_peak():
    """Starts counting the most GPU memory PyTorch takes from now (see gpu_memory_note)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def gpu_memory_note(peak=False):
    """' (GPU: 3120 of 12288 MB free)', for the log, with the most PyTorch took since
    reset_gpu_peak (`peak`); '' without a GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ''
        free, total = torch.cuda.mem_get_info(0)
        most = f', up to {torch.cuda.max_memory_allocated() / 2**20:.0f} MB taken' if peak else ''
        return f' (GPU: {free / 2**20:.0f} of {total / 2**20:.0f} MB free{most})'
    except Exception:  # just for the log
        return ''


def gpu_memory_mb():
    """The NVIDIA GPU's memory free now, in MB (other programs', and this one's models already
    loaded, taken off), or None without one."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info(0)
    return free / 2**20


def machine():
    """What this worker runs on, as it finds it (nothing assumed: a CPU, any GPU), for the log."""
    parts = [f'{platform.system()} {platform.machine()}', f'{os.cpu_count() or "?"} CPU threads']
    try:
        with open('/proc/meminfo', encoding='utf-8') as meminfo:
            total_kb = int(next(line for line in meminfo if line.startswith('MemTotal:')).split()[1])
        parts.append(f'{total_kb / 2**20:.0f} GB RAM')
    except (OSError, StopIteration, ValueError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                parts.append(f'GPU {index}: {props.name}, {total / 2**30:.1f} GB ({free / 2**30:.1f} GB free)')
        elif torch.backends.mps.is_available():
            parts.append("the Mac's GPU (Metal)")
        else:
            parts.append('no GPU PyTorch can use')
    except ImportError:
        parts.append("PyTorch isn't installed")
    return ', '.join(parts)


def fitting(slots, per_song_mb, kept_mb=0):
    """How many songs can have their vocals removed at once: on an NVIDIA GPU, one for each slot,
    but only as many as the memory free now fits (each loads the model, and works in GPU memory),
    less `kept_mb` kept for reading the singing with Whisper; on a CPU (or a Mac's GPU),
    one at a time, as each takes every core (a slot's next song still downloads meanwhile).
    `per_song_mb` 0: one for each slot, whatever it runs on."""
    if per_song_mb <= 0:
        return slots
    free_mb = gpu_memory_mb()
    if free_mb is None:
        return 1
    return max(1, min(slots, int((free_mb - kept_mb) // per_song_mb)))


def main():
    parser = argparse.ArgumentParser(description='Takes the vocals out of songs for a JamTube Lyrics server.')
    parser.add_argument('--server', default=os.environ.get('JAMTUBE_LYRICS_URL', ''), help="The server's address (or JAMTUBE_LYRICS_URL).")
    parser.add_argument('--key', default=os.environ.get('WORKER_KEY', ''), help="The server's WORKER_KEY (or WORKER_KEY).")
    parser.add_argument('--name', default=os.environ.get('WORKER_NAME') or socket.gethostname(), help="What the server calls it (or WORKER_NAME; this computer's name).")
    parser.add_argument('--models-dir', default=os.environ.get('MODELS_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
                        help="Where the model is kept between runs (or MODELS_DIR; models, next to this).")
    parser.add_argument('--gpu-memory-per-song', type=int, default=int(os.environ.get('GPU_MEMORY_PER_SONG_MB') or 2500),
                        help="GPU memory one song's vocal removal takes, in MB (or GPU_MEMORY_PER_SONG_MB; 2500): on an NVIDIA GPU, only as many songs as fit have their vocals removed at once. 0: one for each slot.")
    parser.add_argument('--no-songs', action='store_true', default=os.environ.get('WORKER_SONGS', '').lower() in ('0', 'false', 'no', 'off'),
                        help="Only remove vocals, never make whole songs (or WORKER_SONGS=false).")
    parser.add_argument('--gpu-memory-whisper', type=int, default=int(os.environ.get('GPU_MEMORY_WHISPER_MB') or 4500),
                        help="GPU memory Whisper takes, in MB (or GPU_MEMORY_WHISPER_MB; 4500), when the server has it: kept free of vocal removal.")
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
    if shutil.which('nvidia-smi'):
        repair_torch()
    hearing = Hearing(args.models_dir, scratch)
    if shutil.which('nvidia-smi'):
        repair_onnxruntime()
    log(f'"{args.name}" working for {server.url}, on {accelerator()}')
    log(f'This machine: {machine()}')
    doing = {}  # job id -> job: the songs the slots are doing
    making = set()  # the prepare.py processes making whole songs here: stopped with the worker
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
        me['version'] = setup.get('version')  # what the server's workers run (see update)
        slots = max(1, int(setup.get('jobs') or 1))
        # The slots download and send back songs side by side, sharing the separators: as many as
        # the GPU's memory fits (each holds its model and works in GPU memory).
        whisper = setup.get('whisper') or ''  # kept in GPU memory too
        removing = fitting(slots, args.gpu_memory_per_song, args.gpu_memory_whisper if whisper else 0)
        log(f"The server sends up to {slots} song{'s' if slots != 1 else ''} at once; {removing} can have their vocals removed at once here"
            + (f' (about {args.gpu_memory_per_song} MB of GPU memory each, {args.gpu_memory_whisper} MB kept for Whisper)' if gpu_memory_mb() is not None and whisper
               else f' (about {args.gpu_memory_per_song} MB of GPU memory each)' if gpu_memory_mb() is not None else ' (no NVIDIA GPU: one at a time)'))
        settings = {'code': SongCode(os.path.join(scratch, 'code')), 'lyrics': os.path.join(scratch, 'lyrics'), 'models': args.models_dir,
                    'name': args.name, 'threads': max(1, (os.cpu_count() or 2) // slots),
                    'making': making}
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
                hearing.get_ready(whisper)
                fits = f', removing the vocals of {removing} at once (what the GPU\'s memory fits)' if removing < slots else ''
                log(f"Ready for songs{f', {slots} at once' if slots > 1 else ''}{fits}.")
            except Exception as error:  # the first song tries again
                log(f"Couldn't get the models ready ({str(error)[:300]}): the first song will try again.")

        # Whole songs too (see make_song), when it has what they need, and hears them as the
        # server does (its singing model's loaded here).
        missing = None if args.no_songs else songs_missing()
        if not args.no_songs and not missing and (hearing.loaded or setup.get('separator') == 'none'):
            me['songs'] = True
            log('It makes whole songs too, when the server gives it one.')
        elif missing:
            log(f'It only removes vocals: to make whole songs too, it needs {missing}.')
        elif not args.no_songs:
            log("It only removes vocals: it can't hear the singing as the server does, which whole songs need.")

        def quit(why):
            stopped_why.append(why)
            stop.set()
        updated = threading.Event()

        def outdated():
            if not updated.is_set():
                updated.set()
                log('The server was updated: starting again with the update once the songs it has are done.' if doing
                    else 'The server was updated: starting again with the update.')
        threads = []
        for slot in range(1, slots + 1):
            def say(message, slot=slot):
                log(f'[{slot}] {message}' if slots > 1 else message)
            threads.append(threading.Thread(target=take_songs, daemon=True,
                                            args=(server, separators, hearing, os.path.join(scratch, f'song-{slot}'), me, doing, say, quit, outdated, settings)))
            threads[-1].start()
        while not stop.wait(1):
            if updated.is_set() and not any(thread.is_alive() for thread in threads):
                update(server, scratch)
        sys.exit(stopped_why[0])
    except KeyboardInterrupt:
        # Stopped (Ctrl+C, docker stop, a cloud taking back its machine): the server's told about
        # each song it was doing, so they go elsewhere straight away.
        songs = list(doing.values())
        for process in list(making):  # (whole songs': see make_song)
            process.kill()
        for job in songs:
            give_back(server, job, 'the worker was stopped')
        log(f"Stopped: {'its song goes' if len(songs) == 1 else f'its {len(songs)} songs go'} back to the server." if songs else 'Stopped.')
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    main()
