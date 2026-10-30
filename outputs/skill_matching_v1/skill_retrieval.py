"""Opt-in semantic candidates; cosine is NOT an execution success probability."""
import atexit
import copy
import hashlib
import json
import math
import queue
import subprocess
import threading
import time
from pathlib import Path

from match_skills import read_json, write_json

PROFILES = Path(__file__).with_name('skill_semantic_profiles.json')


def enrich(library):
    result = copy.deepcopy(library)
    profiles = read_json(PROFILES)['profiles']
    for skill in result['skills']:
        profile = profiles[skill['skill_id']]
        skill['description_en'] = profile['description_en']
        skill['semantic_profile'] = copy.deepcopy(profile)
    result['semantic_metadata_version'] = read_json(PROFILES)['version']
    return result


def model_fingerprint(path):
    root = Path(path).resolve()
    files = sorted(p for p in root.rglob('*') if p.is_file() and
                   ('safetensors' in p.name or p.suffix in ('.bin', '.json', '.txt', '.model')))
    if not files:
        raise ValueError('No local embedding model files; download explicitly before experiment')
    digest = hashlib.sha256()
    for p in files:
        digest.update(str(p.relative_to(root)).encode())
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


class LocalEncoder:
    """One resident model per case worker; startup and encoding time are counted."""
    def __init__(self, python, model_path, log_dir, timeout=180):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (log_dir/'embedding.stderr.log').open('w', encoding='utf-8')
        self.timeout, self.lines = timeout, queue.Queue()
        self.process = subprocess.Popen([str(python), str(Path(__file__).with_name('embedding_worker.py')),
            '--model-path', str(model_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, encoding='utf-8', bufsize=1)
        self.seconds, self.calls, self.texts, self.cache_hits = 0., 0, 0, 0
        self.cache = {}
        def pump():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        atexit.register(self.close)
        started = time.monotonic()
        try:
            ready = self.receive()
            if ready.get('ready') is not True:
                raise ValueError('Embedding worker did not initialize')
            self.environment = ready
        except Exception:
            self.close()
            raise
        self.seconds += time.monotonic()-started

    def receive(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            raise ValueError('EMBEDDING_WORKER_TIMEOUT')
        if line is None:
            raise ValueError('Embedding worker exited; inspect embedding.stderr.log')
        value = json.loads(line)
        if 'error' in value:
            raise ValueError(value['error'])
        return value

    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        self.cache_hits += len(texts)-len(missing)
        if missing:
            started = time.monotonic()
            self.process.stdin.write(json.dumps({'texts': missing}, ensure_ascii=True)+'\n')
            self.process.stdin.flush()
            vectors = self.receive()['vectors']
            if len(vectors) != len(missing):
                raise ValueError('Embedding response length mismatch')
            self.cache.update(zip(missing, vectors))
            self.calls += 1
            self.texts += len(missing)
            self.seconds += time.monotonic()-started
        return [self.cache[t] for t in texts]

    def stats(self):
        return {'seconds_including_startup': self.seconds, 'batch_calls': self.calls,
                'encoded_texts': self.texts, 'cache_hits': self.cache_hits,
                'provider_tokens': 0, 'backend': 'local_cpu_sentence_transformers',
                'environment': self.environment}

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if not self.log.closed:
            self.log.close()


def cosine(a, b):
    if (not a or len(a) != len(b) or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in list(a)+list(b))):
        raise ValueError('Invalid embedding vector')
    norm = math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    if norm == 0:
        raise ValueError('Zero embedding vector')
    return sum(x*y for x,y in zip(a,b))/norm


def candidates(alignment, library, encoder, top_k=2, threshold=.35, margin=.05):
    if not 1 <= top_k <= len(library['skills']) or not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError('Invalid retrieval limits')
    documents, owners = [], []
    by_id = {s['skill_id']: s for s in library['skills']}
    for s in library['skills']:
        profile = s['semantic_profile']
        # Negated counterexamples stay out of positive embedding documents.
        for text in [profile['description_en'], *profile['examples'], *profile['aliases']]:
            documents.append(text)
            owners.append(s['skill_id'])
    doc_vectors = encoder.encode(documents)
    subtasks = alignment['task_decomposition']['subtasks']
    queries = [s['description'] for s in subtasks]
    query_vectors = encoder.encode(queries)
    rows = []
    for subtask, query, vector in zip(subtasks, queries, query_vectors):
        scores = {skill: -1. for skill in by_id}
        for owner, dv in zip(owners, doc_vectors):
            scores[owner] = max(scores[owner], cosine(vector, dv))
        ranked = sorted(scores, key=lambda k: (-scores[k], k))
        confident = scores[ranked[0]] >= threshold and (len(ranked)==1 or
                     scores[ranked[0]]-scores[ranked[1]] >= margin)
        selected = set(ranked[:top_k]) if confident else set(by_id)
        reason = 'top_k_with_prerequisite_closure' if confident else 'ambiguous_keep_full_adapter_inventory'
        # Closure is a candidate set, NOT automatically executed prerequisites.
        todo = list(selected)
        while todo:
            skill = todo.pop()
            for dep in by_id[skill]['semantic_profile']['prerequisite_skills']:
                if dep in by_id and dep not in selected:
                    selected.add(dep)
                    todo.append(dep)
        rows.append({'subtask_id': subtask['subtask_id'], 'query': query,
            'ranking': [{'skill_id': k, 'cosine': scores[k]} for k in ranked],
            'candidate_skill_ids': sorted(selected), 'confident': confident, 'policy': reason})
    return {'subtasks': rows, 'top_k': top_k, 'threshold': threshold, 'margin': margin,
        'calibrated': False, 'score_is_probability': False,
        'scope': 'Candidate retrieval, not independent LLM agreement or physical feasibility'}


def check_membership(proposal, retrieval, widened=False):
    allowed = {s['subtask_id']: set(s['candidate_skill_ids']) for s in retrieval['subtasks']}
    conflicts = []
    for subtask in proposal['subtask_plans']:
        for step in subtask['steps']:
            if step['skill_id'] not in allowed[subtask['subtask_id']]:
                conflicts.append({'subtask_id': subtask['subtask_id'], 'skill_id': step['skill_id']})
    if conflicts and not widened:
        raise ValueError('SKILL_RETRIEVAL_CONFLICT: '+json.dumps(conflicts))
    return {'candidate_membership': not conflicts, 'widened_inventory': widened, 'conflicts': conflicts,
            'independent_agreement_claim': False}
