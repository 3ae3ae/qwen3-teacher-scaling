"""Teacher-size comparison using pinned Cartridges benchmark recipes."""
import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'external/cartridges'
os.environ['CARTRIDGES_DIR'] = str(UPSTREAM)
os.environ['CARTRIDGES_OUTPUT_DIR'] = str(ROOT / 'artifacts')
sys.path.insert(0, str(UPSTREAM))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files_hash(paths):
    if not paths:
        raise ValueError('No data files')
    return hashlib.sha256(json.dumps([(p.name, sha(p)) for p in paths]).encode()).hexdigest()


def verify_source(cfg):
    for name, spec in [('cartridges', cfg['upstream']), ('tokasaurus', cfg['tokasaurus'])]:
        directory = ROOT / 'external' / name
        revision = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != spec['revision']:
            raise ValueError(f'{name}: revision mismatch')
        actual = subprocess.check_output(['git', '-C', str(directory), 'diff', '--binary'])
        expected = (ROOT / 'patches/cartridges.patch').read_bytes() if name == 'cartridges' else b''
        if actual != expected:
            raise ValueError(f'{name}: source differs from pinned patch')


def reference_configs(benchmark):
    """Load the published Qwen recipe rather than restating its hyperparameters."""
    old = {k: os.environ.get(k) for k in ['MODEL', 'NUM_TOKENS']}
    os.environ['MODEL'] = 'qwen'
    os.environ.pop('NUM_TOKENS', None)
    try:
        module = f'examples.benchmarks.{benchmark}.{benchmark}'
        training = importlib.import_module(module + '_train')
        synthesis = importlib.import_module(module + '_synthesize')
        native = training.config if benchmark == 'longhealth' else training.configs[0]
        return native.model_copy(deep=True), synthesis.config.model_copy(deep=True)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def tokenizer_for(cfg, size='4B'):
    from transformers import AutoTokenizer
    spec = cfg['models'][size]
    return AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'])


def recipe(cfg, benchmark, profile):
    train, synth = reference_configs(benchmark)
    overrides = cfg['profiles'][profile]
    return train, synth, {'samples': synth.num_samples * len(train.dataset.data_sources),
                         'max_steps': -1, 'eval_questions': None, **overrides}


def bind_data(cfg, out):
    from cartridges.data.longhealth import utils
    from cartridges.data.mtob import load
    ds = cfg['benchmarks']['longhealth']
    utils.DATASET_PATH = f"https://raw.githubusercontent.com/kbressem/LongHealth/{ds['revision']}/{ds['file']}"
    load.dataset_root = out / 'data/mtob'


def resource_config(cfg, benchmark, profile):
    _, synth = reference_configs(benchmark)
    resource = synth.synthesizer.resources[0]
    if benchmark == 'longhealth' and profile == 'smoke':
        resource.patient_ids = cfg['benchmarks'][benchmark]['development_patients']
    if benchmark == 'mtob':
        resource.tokenizer = cfg['models']['4B']['id']
        resource.tokenizer_revision = cfg['models']['4B']['revision']
    return resource


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Dataset hash mismatch')
        path.write_bytes(data)
    if sha(path) != expected:
        raise ValueError('Dataset hash mismatch')


def prepare(cfg, benchmark, profile, out):
    train, synth, s = recipe(cfg, benchmark, profile)
    data = out / 'data'
    data.mkdir(parents=True, exist_ok=True)
    ds = cfg['benchmarks'][benchmark]
    repo = ds['repository'].removeprefix('https://github.com/')
    url = f"https://raw.githubusercontent.com/{repo}/{ds['revision']}/{ds['file']}"
    if benchmark == 'longhealth':
        download(url, data / 'longhealth.json', ds['sha256'])
    else:
        archive = data / 'mtob.zip'
        download(url, archive, ds['sha256'])
        (data / 'mtob').mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.filename.startswith(('resources/', 'splits/')) and not member.is_dir():
                    (data / 'mtob' / Path(member.filename).name).write_bytes(z.read(member, pwd=b'kalamang'))
    tokenizer = tokenizer_for(cfg)
    other = tokenizer_for(cfg, '8B')
    if tokenizer.get_vocab() != other.get_vocab() or tokenizer.chat_template != other.chat_template:
        raise ValueError('Teacher/student tokenizers differ')
    resource = resource_config(cfg, benchmark, profile).instantiate()
    corpus = resource.to_string()
    (data / 'corpus.txt').write_text(corpus)
    prompts = []
    random.seed(cfg['prompt_seed'])
    for start in range(0, s['samples'], synth.batch_size):
        context, seeds = asyncio.run(resource.sample_prompt(min(synth.batch_size, s['samples'] - start)))
        prompts.append({'context': context, 'seed_prompts': seeds})
    write_json(data / 'prompts.json', prompts)
    return {'benchmark': benchmark, 'samples_per_generator': s['samples'], 'generation_batch': synth.batch_size,
            'corpus_tokens': len(tokenizer.encode(corpus)), 'source_url': url,
            'files': {p.relative_to(data).as_posix(): sha(p) for p in sorted(data.rglob('*')) if p.is_file()},
            'tokenizer_vocab_sha256': hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
            'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}


from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.initialization import KVFromText
from cartridges.cache import TrainableCache


class RecordingClient(TokasaurusClient):
    """Preserve native completions and logprobs; record prompts for forced scoring."""
    class Config(TokasaurusClient.Config):
        revision: str

    def __init__(self, config):
        from transformers import AutoTokenizer
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision)
        self.answer_prompts = []

    async def _send_requests(self, requests, modal_upstream_id=None, use_cartridge_endpoint=False):
        if requests and requests[0].get('top_logprobs'):
            self.answer_prompts = []
            for request in requests:
                prompt = self.tokenizer.apply_chat_template(request['messages'], tokenize=False,
                    add_generation_prompt=True, continue_final_message=False,
                    **request.get('apply_chat_template_overrides', {}))
                self.answer_prompts.append(self.tokenizer.encode(prompt, add_special_tokens=False))
        return await super()._send_requests(requests, modal_upstream_id, use_cartridge_endpoint=use_cartridge_endpoint)


class SharedInitializer(KVFromText):
    class Config(KVFromText.Config):
        path: str

    def initialize_kv_cache(self, tokenizer, model, attn_config):
        path = Path(self.config.path)
        if path.exists():
            return TrainableCache.from_pretrained(str(path), device=str(model.device))
        cache = super().initialize_kv_cache(tokenizer, model, attn_config)
        temporary = path.with_suffix('.tmp')
        cache.save(str(temporary))
        temporary.replace(path)
        return cache


@contextmanager
def server(cfg, out, generator):
    """Run the original server in its own process, releasing its GPU after synthesis."""
    from huggingface_hub import snapshot_download
    import requests
    spec = cfg['models'][generator]
    snapshot = Path(snapshot_download(spec['id'], revision=spec['revision'],
        allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.tiktoken']))
    work = out / 'models'
    link = work / spec['id']
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != snapshot.resolve():
            raise ValueError('Model snapshot mismatch')
    elif link.exists():
        raise FileExistsError(link)
    else:
        link.symlink_to(snapshot, target_is_directory=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [sys.executable, '-m', 'tokasaurus.entry', f"model={spec['id']}", f'port={port}',
               f"kv_cache_num_tokens={cfg['tokasaurus']['kv_cache_num_tokens']}",
               'max_topk_logprobs=20', 'max_seqs_per_forward=128', 'dp_size=1', 'wandb_enabled=False']
    with (out / f'server-{generator}.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'Tokasaurus exited: see server-{generator}.log')
                try:
                    response = requests.get(url + '/v1/models', timeout=2)
                    if response.ok and response.json()['data'][0]['id'].lower() == spec['id'].lower():
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Tokasaurus startup timeout')
            yield url
        finally:
            # The server spawns model/manager children; terminate its whole process group.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def conversation_paths(out, generator):
    return sorted((out / f'G{generator}').glob('conversations-[0-9][0-9][0-9][0-9][0-9].parquet'))


def synthesize(cfg, benchmark, profile, out, generator):
    from cartridges.structs import write_conversations, read_conversations
    _, reference, s = recipe(cfg, benchmark, profile)
    prompts = read_json(out / 'data/prompts.json')
    directory = out / f'G{generator}'
    directory.mkdir(exist_ok=True)
    paths = conversation_paths(out, generator)
    if [p.name for p in paths] != [f'conversations-{i:05d}.parquet' for i in range(len(paths))]:
        raise ValueError('Conversation shards are not contiguous')
    for i, path in enumerate(paths):
        if len(read_conversations(path)) != len(prompts[i]['seed_prompts']):
            raise ValueError('Incomplete conversation batch')
    if len(paths) < len(prompts):
        spec = cfg['models'][generator]
        with server(cfg, out, generator) as url:
            config = reference.synthesizer
            config.client = RecordingClient.Config(model_name=spec['id'], revision=spec['revision'],
                                                  url=url, base_timeout=3600, max_retries=1)
            config.resources = []
            if 'question_max_tokens' in s:
                config.max_completion_tokens_a = s['question_max_tokens']
                config.max_completion_tokens_b = s['answer_max_tokens']
            synth = config.instantiate()
            class PreparedResource:
                async def sample_prompt(self, batch_size):
                    assert batch_size == len(self.row['seed_prompts'])
                    return self.row['context'], self.row['seed_prompts']
            resource = PreparedResource()
            async def run():
                await synth.setup()
                synth.resources = [resource]
                try:
                    for i in range(len(paths), len(prompts)):
                        resource.row = prompts[i]
                        random.seed(cfg['prompt_seed'] + i)
                        rows = await synth.sample_convos(i, len(resource.row['seed_prompts']), len(prompts))
                        for j, row in enumerate(rows):
                            if row.messages[-1].top_logprobs is None:
                                raise ValueError('Server did not return logprobs')
                            row.metadata.update(sample_id=i * reference.batch_size + j,
                                                prompt_ids=synth.client.answer_prompts[j])
                        path = directory / f'conversations-{i:05d}.parquet'
                        temporary = path.with_suffix('.tmp.parquet')
                        write_conversations(rows, temporary)
                        temporary.replace(path)
                        print(json.dumps({'batch': i + 1, 'total': len(prompts)}), flush=True)
                finally:
                    await synth.cleanup()
            asyncio.run(run())
    paths = conversation_paths(out, generator)
    return {'generator': generator, 'samples': s['samples'], 'conversation_sha256': files_hash(paths)}


def score(cfg, benchmark, profile, out, generator, teacher):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from cartridges.clients.base import TopLogprobs
    from cartridges.structs import read_conversations, write_conversations
    sources = conversation_paths(out, generator)
    if read_json(out / f'G{generator}/synthesize.summary.json')['conversation_sha256'] != files_hash(sources):
        raise ValueError('Conversation hash mismatch')
    spec = cfg['models'][teacher]
    model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': 0}).eval().requires_grad_(False)
    _, reference, _ = recipe(cfg, benchmark, profile)
    k, threshold = reference.synthesizer.num_top_logprobs, reference.synthesizer.min_prob_mass
    mass_sum, tokens, below, delta_sum, delta_count, delta_max = 0., 0, 0, 0., 0, 0.
    destinations = []
    for source in sources:
        rows = read_conversations(source)
        for row in rows:
            prompt, answer = list(row.metadata['prompt_ids']), list(row.messages[-1].token_ids)
            if not prompt or not answer:
                raise ValueError('Empty prompt/completion')
            ids = torch.tensor([prompt + answer], device='cuda')
            positions = torch.arange(len(prompt) - 1, len(prompt) + len(answer) - 1, device='cuda')
            with torch.inference_mode():
                logp = model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0].float().log_softmax(-1)
                values, indices = logp.topk(k, dim=-1)
            original = row.messages[-1].top_logprobs
            if generator == teacher:
                delta = (logp[torch.as_tensor(original.token_idx, device='cuda'),
                              torch.as_tensor(original.token_id, device='cuda')] -
                         torch.as_tensor(original.logprobs, device='cuda')).abs()
                delta_sum += delta.sum().item()
                delta_count += delta.numel()
                delta_max = max(delta_max, delta.max().item())
            dense = TopLogprobs(logprobs=values.cpu().numpy(), token_ids=indices.cpu().numpy())
            row.messages[-1].top_logprobs = dense.flatten(threshold=threshold)
            masses = np.exp(dense.logprobs).sum(-1)
            mass_sum += float(masses.sum()); tokens += len(answer); below += int((masses < threshold).sum())
        destination = source.with_name(source.name.replace('conversations-', f'scores-{teacher}-'))
        temporary = destination.with_suffix('.tmp.parquet')
        write_conversations(rows, temporary)
        temporary.replace(destination)
        destinations.append(destination)
    return {'generator': generator, 'teacher': teacher, 'conversation_sha256': files_hash(sources),
            'scores_sha256': files_hash(destinations), 'target_tokens': tokens,
            'top20_mass_mean': mass_sum / tokens, 'below_threshold_fraction': below / tokens,
            'same_teacher_logprob_mae': delta_sum / delta_count if delta_count else None,
            'same_teacher_logprob_max_abs': delta_max if delta_count else None}


def train_config(cfg, benchmark, profile, out, condition, seed):
    from cartridges.datasets import DataSource
    native, _, s = recipe(cfg, benchmark, profile)
    pair = cfg['conditions'][condition]
    paths = sorted((out / f"G{pair['generator']}").glob(f"scores-{pair['teacher']}-[0-9][0-9][0-9][0-9][0-9].parquet"))
    native.dataset.data_sources = [DataSource(path=str(p), type='local') for p in paths]
    spec = cfg['models']['4B']
    native.model.pretrained_model_name_or_path = spec['id']
    native.model.load_kwargs = {'revision': spec['revision'], 'torch_dtype': 'bfloat16'}
    original_init = native.kv_cache_initializer
    native.kv_cache_initializer = SharedInitializer.Config(max_tokens=s.get('cartridge_tokens', original_init.max_tokens),
        text_source=original_init.text_source, system_prompt_template=original_init.system_prompt_template,
        num_frozen_tokens=original_init.num_frozen_tokens, path=str(out / 'initial-cache.pt'))
    native.name = f'{benchmark}-{condition}-{seed}'
    native.run_dir = str(out / f'seed-{seed}/{condition}')
    native.wandb = None
    native.save_to_wandb = False
    native.seed = seed
    native.max_optimizer_steps = s['max_steps']
    native.global_batch_size = s.get('global_batch', native.global_batch_size)
    for evaluation in native.generate_evals:
        if benchmark == 'longhealth' and profile == 'smoke':
            evaluation.dataset.patient_ids = cfg['benchmarks'][benchmark]['development_patients']
        evaluation.batch_size = s.get('eval_batch_size', evaluation.batch_size)
        if benchmark == 'longhealth':
            evaluation.dataset.max_questions = s['eval_questions']
    return native


def metrics(benchmark, dataset, rows):
    if benchmark == 'longhealth':
        return {'questions': len(rows), 'accuracy': sum(r['score'] for r in rows) / len(rows),
                'missing_answer_tags': sum(r['extracted_pred'] is None for r in rows)}
    return {'questions': len(rows), 'chrf': dataset.batch_score_with_answers([r['pred'] for r in rows], [r['answer'] for r in rows])}


def train(cfg, benchmark, profile, out, condition, seed):
    import cartridges.train as module
    config = train_config(cfg, benchmark, profile, out, condition, seed)
    pair = cfg['conditions'][condition]
    source = out / f"G{pair['generator']}"
    scored = read_json(source / f"score-{pair['teacher']}.summary.json")
    if scored['conversation_sha256'] != files_hash(conversation_paths(out, pair['generator'])) or scored['scores_sha256'] != files_hash([Path(d.path) for d in config.dataset.data_sources]):
        raise ValueError('Scoring manifest mismatch')
    directory = Path(config.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob('cache*.pt')):
        raise FileExistsError('Existing training checkpoint; use a fresh run')
    config.to_yaml(str(directory / 'native-config.yaml'))
    dataset = config.dataset.instantiate(tokenizer=tokenizer_for(cfg), seed=seed)
    if not len(dataset) or len(dataset) * config.epochs < config.global_batch_size:
        raise ValueError('Insufficient batches for an optimizer step')
    packed_batches = len(dataset)
    del dataset
    original = module.evaluate_generations
    evaluations = []
    def record(*args, **kwargs):
        import torch
        step, final = kwargs['optimizer_step'], kwargs.get('final', False)
        if benchmark == 'mtob' and profile == 'smoke':
            kwargs['dataset'].data = kwargs['dataset'].data[:cfg['profiles']['smoke']['eval_questions']]
        torch.save({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()},
                   directory / f'rng-step{step}{"-final" if final else ""}.pt')
        rows = original(*args, **kwargs)
        write_json(directory / f'predictions-step{step}{"-final" if final else ""}.json', rows)
        result = {'step': step, 'final': final, **metrics(benchmark, kwargs['dataset'], rows)}
        evaluations.append(result)
        write_json(directory / 'metrics.json', evaluations)
        return rows
    module.evaluate_generations = record
    try:
        config.run()
    finally:
        module.evaluate_generations = original
    checkpoint = directory / 'cache_last.pt'
    return {'condition': condition, 'seed': seed, 'packed_batches_per_epoch': packed_batches,
            'optimizer_steps': int(checkpoint.resolve().stem.removeprefix('cache-step')),
            'initial_cache_sha256': sha(out / 'initial-cache.pt'), 'checkpoint_sha256': sha(checkpoint),
            'evaluations': evaluations}


def evaluate(cfg, benchmark, profile, out, condition, seed):
    from cartridges.train import CacheAndModel, evaluate_generations
    native = train_config(cfg, benchmark, profile, out, condition, seed)
    model = native.model.instantiate().to('cuda').eval().requires_grad_(False)
    path = out / f'seed-{seed}/{condition}/cache_last.pt'
    cache = TrainableCache.from_pretrained(str(path), device='cuda').to('cuda')
    tokenizer = tokenizer_for(cfg)
    config = native.generate_evals[0]
    dataset = config.dataset.instantiate(tokenizer=tokenizer, seed=seed)
    if benchmark == 'mtob' and profile == 'smoke':
        dataset.data = dataset.data[:cfg['profiles']['smoke']['eval_questions']]
    from cartridges.utils import seed_everything
    seed_everything(seed)
    import torch
    rng_files = list(path.parent.glob('rng-step*-final.pt'))
    if len(rng_files) != 1:
        raise ValueError('Expected one final evaluation RNG snapshot')
    rng = torch.load(rng_files[0], map_location='cpu', weights_only=True)
    torch.set_rng_state(rng['cpu'])
    torch.cuda.set_rng_state_all(rng['cuda'])
    rows = evaluate_generations(config, CacheAndModel(cache, model), tokenizer, dataset, 0, 'cuda', log_to_wandb=False)
    write_json(out / f'seed-{seed}/{condition}/reevaluation.json', rows)
    return {'condition': condition, 'seed': seed, **metrics(benchmark, dataset, rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'synthesize', 'score', 'train', 'evaluate'])
    parser.add_argument('--config', default=str(ROOT / 'configs/experiment.json'))
    parser.add_argument('--benchmark', choices=['longhealth', 'mtob'], default='longhealth')
    parser.add_argument('--profile', choices=['smoke', 'pilot', 'main'], default='smoke')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--generator', choices=['4B', '8B'], default='4B')
    parser.add_argument('--teacher', choices=['4B', '8B'], default='4B')
    parser.add_argument('--condition', choices=['A', 'B', 'C', 'D'], default='A')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cfg = read_json(args.config)
    verify_source(cfg)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    resolved = {'config': cfg, 'benchmark': args.benchmark, 'profile': args.profile,
                'code_sha256': sha(__file__), 'patch_sha256': sha(ROOT / 'patches/cartridges.patch'),
                'lock_sha256': sha(ROOT / 'requirements.lock')}
    manifest = out / 'resolved-config.json'
    if manifest.exists() and read_json(manifest) != resolved:
        raise ValueError('Run directory belongs to a different protocol')
    write_json(manifest, resolved)
    bind_data(cfg, out)
    import torch
    environment = ROOT / f'results/environment-{"gpu" if torch.cuda.is_available() else "local"}.json'
    environment_sha256 = sha(environment) if environment.exists() else None
    if environment_sha256:
        write_json(out / 'environments' / f'{environment_sha256}.json', read_json(environment))
    if args.stage != 'prepare':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA with native BF16 is required')
        torch.cuda.reset_peak_memory_stats()
        prepared = read_json(out / 'prepare.summary.json')
        for filename, expected in prepared['files'].items():
            if sha(out / 'data' / filename) != expected:
                raise ValueError('Prepared data changed')
    start = time.perf_counter()
    common = cfg, args.benchmark, args.profile, out
    if args.stage == 'prepare':
        result = prepare(*common); summary = out / 'prepare.summary.json'
    elif args.stage == 'synthesize':
        result = synthesize(*common, args.generator); summary = out / f'G{args.generator}/synthesize.summary.json'
    elif args.stage == 'score':
        result = score(*common, args.generator, args.teacher); summary = out / f'G{args.generator}/score-{args.teacher}.summary.json'
    else:
        result = (train if args.stage == 'train' else evaluate)(*common, args.condition, args.seed)
        summary = out / f'seed-{args.seed}/{args.stage}-{args.condition}.summary.json'
    result.update(stage=args.stage, seconds=time.perf_counter() - start, command=sys.argv,
                  peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                      if args.stage != 'synthesize' and torch.cuda.is_available() else None),
                  code_sha256=resolved['code_sha256'], patch_sha256=resolved['patch_sha256'],
                  lock_sha256=resolved['lock_sha256'], environment_sha256=environment_sha256)
    write_json(summary, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
