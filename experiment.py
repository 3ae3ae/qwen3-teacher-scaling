"""Teacher-size comparison using pinned Cartridges benchmark recipes."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'external/cartridges'
os.environ['CARTRIDGES_DIR'] = str(UPSTREAM)
os.environ['CARTRIDGES_OUTPUT_DIR'] = str(ROOT / 'artifacts')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
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
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def files_hash(paths):
    if not paths:
        raise ValueError('No data files')
    return hashlib.sha256(json.dumps([(p.name, sha(p)) for p in paths]).encode()).hexdigest()


def verify_source(cfg):
    revision = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != cfg['upstream']['revision']:
        raise ValueError('Cartridges revision mismatch')
    actual = subprocess.check_output(['git', '-C', str(UPSTREAM), 'diff', '--binary'])
    if actual != (ROOT / 'patches/cartridges.patch').read_bytes():
        raise ValueError('Cartridges source differs from pinned patch')


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
    sources = cfg['benchmarks'][benchmark]['conversations']
    if [s['repository'] for s in sources] != [s.path for s in train.dataset.data_sources]:
        raise ValueError('Public conversation sources differ from the upstream training recipe')
    return train, synth, {'samples': sum(s['rows'] for s in sources),
                         'max_steps': -1, 'eval_questions': None, **overrides}


def bind_data(cfg, out):
    from cartridges.data.longhealth import utils
    from cartridges.data.mtob import load
    ds = cfg['benchmarks']['longhealth']
    utils.DATASET_PATH = f"https://raw.githubusercontent.com/kbressem/LongHealth/{ds['revision']}/{ds['file']}"
    load.dataset_root = out / 'data/mtob'


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
    _, _, s = recipe(cfg, benchmark, profile)
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
    sources = prepare_conversations(ds['conversations'], s['samples'], data, tokenizer)
    return {'benchmark': benchmark, 'samples': s['samples'], 'public_sources': sources,
            'conversation_sha256': files_hash(conversation_paths(out)), 'source_url': url,
            'files': {p.relative_to(data).as_posix(): sha(p) for p in sorted(data.rglob('*')) if p.is_file()},
            'tokenizer_vocab_sha256': hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
            'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}


from cartridges.initialization import KVFromText
from cartridges.cache import TrainableCache


def public_prompt(row, tokenizer):
    """Reconstruct a one-round scoring prompt; infer thinking from stored tokens."""
    if [m.role for m in row.messages] != ['user', 'assistant'] or not row.system_prompt:
        raise ValueError('Expected a public one-round conversation with source context')
    answer = list(row.messages[-1].token_ids)
    original = row.messages[-1].top_logprobs
    if not answer or original is None or int(original.shape[0]) != len(answer):
        raise ValueError('Public answer tokens/logprobs are missing or misaligned')
    if row.metadata.get('tool_calls') or row.metadata.get('initial_system_prompt') != row.system_prompt:
        raise ValueError('Unexpected public conversation context')
    think = tokenizer.convert_tokens_to_ids('<think>')
    thinking = answer[0] == think
    if not thinking and any(t in answer for t in [think, tokenizer.convert_tokens_to_ids('</think>')]):
        raise ValueError('Ambiguous public thinking token sequence')
    messages = [{'role': 'system', 'content': row.system_prompt}, row.messages[0].to_message_dict()]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          continue_final_message=False, enable_thinking=thinking)
    return tokenizer.encode(prompt, add_special_tokens=False), thinking


def prepare_conversations(sources, samples, data, tokenizer):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from cartridges.structs import Conversation, write_conversations
    used, count, batch_idx = [], 0, 0
    for source in sources:
        source_count = 0
        for filename in source['files']:
            if count == samples:
                return used
            path = Path(hf_hub_download(source['repository'], filename, repo_type='dataset', revision=source['revision']))
            parquet = pq.ParquetFile(path)
            selected = min(parquet.metadata.num_rows, samples - count)
            used.append({'repository': source['repository'], 'revision': source['revision'],
                         'file': filename, 'sha256': sha(path), 'rows': parquet.metadata.num_rows,
                         'selected_rows': selected})
            offset = 0
            for batch in parquet.iter_batches(batch_size=32):
                rows = [Conversation.from_dict(r) for r in batch.to_pylist()[:samples - count]]
                for i, row in enumerate(rows):
                    prompt, thinking = public_prompt(row, tokenizer)
                    row.metadata.update(prompt_ids=prompt, inferred_enable_thinking=thinking,
                        source_repository=source['repository'], source_file=filename, source_row=offset + i)
                destination = data / f'conversations-{batch_idx:05d}.parquet'
                temporary = destination.with_suffix('.tmp.parquet')
                write_conversations(rows, temporary)
                temporary.replace(destination)
                count += len(rows); offset += len(rows); batch_idx += 1
                if count == samples:
                    break
            source_count += parquet.metadata.num_rows
            print(json.dumps({'public_file': filename, 'samples': count, 'total': samples}), flush=True)
        if count < samples and source_count != source['rows']:
            raise ValueError('Public source row count mismatch')
    if count != samples:
        raise ValueError('Insufficient public conversations')
    return used


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


def conversation_paths(out):
    return sorted((out / 'data').glob('conversations-[0-9][0-9][0-9][0-9][0-9].parquet'))


def score(cfg, benchmark, profile, out, teacher):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from cartridges.clients.base import TopLogprobs
    from cartridges.structs import read_conversations, write_conversations
    sources = conversation_paths(out)
    if read_json(out / 'prepare.summary.json')['conversation_sha256'] != files_hash(sources):
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
            if teacher == '4B':
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
        destination = out / 'scores' / source.name.replace('conversations-', f'scores-{teacher}-')
        destination.parent.mkdir(exist_ok=True)
        temporary = destination.with_suffix('.tmp.parquet')
        write_conversations(rows, temporary)
        temporary.replace(destination)
        destinations.append(destination)
        print(json.dumps({'teacher': teacher, 'scored_shards': len(destinations), 'total': len(sources)}), flush=True)
    return {'teacher': teacher, 'conversation_sha256': files_hash(sources),
            'scores_sha256': files_hash(destinations), 'target_tokens': tokens,
            'top20_mass_mean': mass_sum / tokens, 'below_threshold_fraction': below / tokens,
            'same_teacher_logprob_mae': delta_sum / delta_count if delta_count else None,
            'same_teacher_logprob_max_abs': delta_max if delta_count else None}


def train_config(cfg, benchmark, profile, out, condition, seed):
    from cartridges.datasets import DataSource
    native, _, s = recipe(cfg, benchmark, profile)
    pair = cfg['conditions'][condition]
    paths = sorted((out / 'scores').glob(f"scores-{pair['teacher']}-[0-9][0-9][0-9][0-9][0-9].parquet"))
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
    source = out / 'scores'
    scored = read_json(source / f"score-{pair['teacher']}.summary.json")
    if scored['conversation_sha256'] != files_hash(conversation_paths(out)) or scored['scores_sha256'] != files_hash([Path(d.path) for d in config.dataset.data_sources]):
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
    parser.add_argument('stage', choices=['prepare', 'score', 'train', 'evaluate'])
    parser.add_argument('--config', default=str(ROOT / 'configs/experiment.json'))
    parser.add_argument('--benchmark', choices=['longhealth', 'mtob'], default='longhealth')
    parser.add_argument('--profile', choices=['smoke', 'pilot', 'main'], default='smoke')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--teacher', choices=['4B', '8B'], default='4B')
    parser.add_argument('--condition', choices=['A', 'B'], default='A')
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
    elif args.stage == 'score':
        result = score(*common, args.teacher); summary = out / f'scores/score-{args.teacher}.summary.json'
    else:
        result = (train if args.stage == 'train' else evaluate)(*common, args.condition, args.seed)
        summary = out / f'seed-{args.seed}/{args.stage}-{args.condition}.summary.json'
    result.update(stage=args.stage, seconds=time.perf_counter() - start, command=sys.argv,
                  peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                      if torch.cuda.is_available() else None),
                  code_sha256=resolved['code_sha256'], patch_sha256=resolved['patch_sha256'],
                  lock_sha256=resolved['lock_sha256'], environment_sha256=environment_sha256)
    write_json(summary, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
