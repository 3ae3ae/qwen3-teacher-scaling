"""CPU checks against the pinned benchmark recipes; no model weights are loaded."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def check_notebook(cfg):
    checks = []
    for path in [ROOT/'experiment.py', ROOT/'scripts/setup.py', Path(__file__)]:
        compile(path.read_text(), str(path), 'exec')
    assert cfg['conditions'] == {'A': {'teacher': '4B'}, 'B': {'teacher': '8B'}}
    notebook = json.loads((ROOT/'notebooks/qwen3_teacher_scaling.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'notebook','exec')
            assert cell['execution_count'] is None and not cell['outputs']
    checks.append('notebook_syntax')
    titles = ['# @title 데이터 준비', '# @title Teacher 재채점', '# @title 학습 또는 재평가']
    cells = {''.join(c['source']).splitlines()[0]: ''.join(c['source'])
             for c in notebook['cells'] if c['cell_type'] == 'code'}
    for mode in ['train', 'evaluate']:
        for selected in [['A'], ['B'], ['A', 'B']]:
            calls = []
            def fake_stage(stage, **kwargs):
                calls.append((stage, kwargs))
                return {}
            scope = {'MODE': mode, 'CONDITIONS': selected, 'BASE': cfg, 'TRAIN_SEEDS': [42, 123],
                     'run_stage': fake_stage, 'display': lambda _: None}
            for title in titles:
                exec(compile(cells[title], 'notebook-routing', 'exec'), scope)
            expected = []
            if mode == 'train':
                expected.append(('prepare', {}))
                expected.extend(('score', {'teacher': cfg['conditions'][c]['teacher']}) for c in selected)
            expected.extend((mode, {'condition': c, 'seed': seed})
                            for seed in [42, 123] for c in selected)
            assert calls == expected, (mode, selected, calls)
    checks.append('notebook_train_evaluate_routing')
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--static-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/experiment.json').read_text())
    checks = check_notebook(cfg)
    if args.static_only:
        print(json.dumps({'status':'passed','scope':'static','checks':checks}))
        return
    import numpy as np
    import torch
    import experiment as ex
    from cartridges.cache import AttnConfig, TrainableCache
    from cartridges.clients.base import TopLogprobs
    from cartridges.datasets import TrainDataset, DataSource, qwen_messages_to_element
    from cartridges.structs import Conversation, read_conversations, write_conversations
    ex.verify_source(cfg)
    assert torch.__version__.split('+')[0] == '2.6.0'
    import transformers
    assert transformers.__version__ == '4.53.0'
    tokenizer = ex.tokenizer_for(cfg)
    other = ex.tokenizer_for(cfg, '8B')
    assert tokenizer.get_vocab() == other.get_vocab() and tokenizer.chat_template == other.chat_template
    checks.extend(['pinned_sources', 'shared_pytorch_transformers_environment', 'tokenizer_identity'])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for benchmark in cfg['benchmarks']:
            original, synth, settings = ex.recipe(cfg, benchmark, 'main')
            assert settings['samples'] == 131072
            assert synth.batch_size == 32
            for condition, pair in cfg['conditions'].items():
                directory = out / 'scores'
                directory.mkdir(exist_ok=True)
                (directory / f"scores-{pair['teacher']}-00000.parquet").touch()
                actual = ex.train_config(cfg, benchmark, 'main', out, condition, 42)
                for field in ['lr','epochs','global_batch_size','optimizer','weight_decay','lr_scheduler',
                              'generate_eval_every_n_steps','generate_before_training','save_every_n_steps']:
                    assert getattr(actual,field) == getattr(original,field), (benchmark,field)
                for field in ['packing_mode','packed_seq_length','top_k_logits','targets']:
                    assert getattr(actual.dataset,field) == getattr(original.dataset,field)
                for field in ['max_tokens','text_source','system_prompt_template','num_frozen_tokens']:
                    assert getattr(actual.kv_cache_initializer,field) == getattr(original.kv_cache_initializer,field)
                assert actual.generate_evals[0].model_dump() == original.generate_evals[0].model_dump()
                assert actual.wandb is None and not actual.save_to_wandb
                assert Path(actual.dataset.data_sources[0].path).name == f"scores-{pair['teacher']}-00000.parquet"
                actual.to_yaml(str(out/'native.yaml'))
        checks.append('main_recipes_match_published_training_and_evaluation_settings')
        answer = tokenizer.encode('Answer.<|im_end|>', add_special_tokens=False)
        logp = np.log(np.tile([.8,.1995], (len(answer),1))).astype(np.float32)
        flat = TopLogprobs(logp,np.tile([20,21],(len(answer),1))).flatten()
        raw_user_ids = tokenizer.encode('Question?<|im_end|>',add_special_tokens=False)
        row = Conversation(messages=[Conversation.Message('Question?','user',raw_user_ids),
                                     Conversation.Message('Answer.','assistant',answer,flat)],
                           system_prompt='Source.',metadata={'initial_system_prompt':'Source.','tool_calls':[]})
        path = out/'conversations.parquet'
        write_conversations([row],path)
        restored = read_conversations(path)[0]
        assert list(restored.messages[0].token_ids) == raw_user_ids
        element = qwen_messages_to_element(restored.messages,tokenizer=tokenizer)
        assert element.input_ids[element.topk_token_idxs.unique()].tolist() == answer
        batch = TrainDataset.Config(data_sources=[DataSource(path=str(path),type='local')],packing_mode='truncate',packed_seq_length=128).instantiate(tokenizer=tokenizer,seed=42)[0]
        assert batch.input_ids[batch.topk_token_idxs.unique()].tolist() == answer
        assert np.array_equal(restored.messages[-1].top_logprobs.logprobs,flat.logprobs)
        checks.append('native_tokens_logprobs_and_packing_roundtrip')
        # Public parquet ingestion preserves order/tokens and reconstructs both prompt modes.
        import copy
        plain = copy.deepcopy(row)
        thinking = copy.deepcopy(row)
        thinking.messages[-1].token_ids = tokenizer.encode('<think>Thought.</think>Answer.<|im_end|>', add_special_tokens=False)
        thinking.messages[-1].top_logprobs = TopLogprobs(
            np.log(np.tile([.8,.1995], (len(thinking.messages[-1].token_ids),1))).astype(np.float32),
            np.tile([20,21],(len(thinking.messages[-1].token_ids),1))).flatten()
        for example, enabled in [(plain, False), (thinking, True)]:
            ids, inferred = ex.public_prompt(example, tokenizer)
            expected = tokenizer.apply_chat_template(
                [{'role':'system','content':'Source.'},{'role':'user','content':'Question?'}],
                add_generation_prompt=True, enable_thinking=enabled)
            assert ids == expected and inferred == enabled
        source1, source2 = out/'public-1.parquet', out/'public-2.parquet'
        write_conversations([plain, thinking], source1)
        write_conversations([thinking, plain], source2)
        spec = {'repository':'test/public','revision':'pinned','rows':4,'files':[source1.name,source2.name]}
        data = out/'data'; data.mkdir()
        with patch('huggingface_hub.hf_hub_download', side_effect=lambda repo, filename, **kw: str(out/filename)) as fetch:
            manifest = ex.prepare_conversations([spec], 3, data, tokenizer)
            assert all(c.kwargs == {'repo_type':'dataset','revision':'pinned'} for c in fetch.call_args_list)
        selected = [r for path in ex.conversation_paths(out) for r in read_conversations(path)]
        assert [r.metadata['source_row'] for r in selected] == [0,1,0]
        assert [r.metadata['inferred_enable_thinking'] for r in selected] == [False,True,True]
        assert sum(f['selected_rows'] for f in manifest) == 3
        for actual, expected in zip(selected, [plain,thinking,thinking], strict=True):
            assert actual.system_prompt == expected.system_prompt
            for a,b in zip(actual.messages,expected.messages,strict=True):
                assert a.content == b.content and list(a.token_ids) == list(b.token_ids)
            assert list(actual.metadata['prompt_ids']) == ex.public_prompt(expected,tokenizer)[0]
        bad = copy.deepcopy(plain); bad.system_prompt = ''
        try:
            ex.public_prompt(bad, tokenizer)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing public context was accepted')
        checks.append('public_parquet_selection_tokens_and_prompt_reconstruction')
        edge = TopLogprobs(np.log(np.array([[.995,.004],[.8,.1995],[.4,.3]])),np.tile([2,3],(3,1))).flatten(.99)
        assert edge.token_idx.tolist() == [0,1,1,2]
        checks.append('upstream_sparse_threshold_behavior')
        with contextlib.redirect_stdout(io.StringIO()):
            cache = TrainableCache(AttnConfig(2,3,4),init_keys=[torch.randn(1,3,5,4) for _ in range(2)],
                                  init_values=[torch.randn(1,3,5,4) for _ in range(2)],num_frozen_tokens=1)
            cache.save(str(out/'cache.pt'))
            restored = TrainableCache.from_pretrained(str(out/'cache.pt'),device='cpu')
            assert restored._num_frozen_tokens == 1
            for k,v in cache.state_dict().items():
                assert torch.equal(v,restored.state_dict()[k])
            init = ex.SharedInitializer.Config(path=str(out/'cache.pt'),max_tokens=5).instantiate()
            assert init.initialize_kv_cache(None,SimpleNamespace(device='cpu'),None)._num_frozen_tokens == 1
        checks.append('cache_roundtrip_and_shared_initialization')
        if args.data_root:
            for benchmark, count in [('longhealth',200),('mtob',50)]:
                run = args.data_root/benchmark
                ex.bind_data(cfg,run)
                native = ex.train_config(cfg,benchmark,'main',run,'A',42)
                ds = native.generate_evals[0].dataset.instantiate(tokenizer=tokenizer,seed=42)
                assert len(ds) == count
                if benchmark == 'longhealth':
                    q = ds.questions[0]
                    assert ds.score(f'<answer>{q.correct}</answer>',q.correct,q.question_id)[0]
                    assert ds.score('no tags',q.correct,q.question_id)[1]['extracted_pred'] is None
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        assert ds.batch_score_with_answers(['the book'],['the book']) == 100
            checks.append('public_evaluation_counts_and_native_scorers')
    report = {'status':'passed','device':'cpu','model_weights_loaded':False,'checks':checks,
              'protocol_version':cfg['protocol_version'],'code_sha256':ex.sha(ROOT/'experiment.py'),
              'patch_sha256':ex.sha(ROOT/'patches/cartridges.patch'),'notebook_sha256':ex.sha(ROOT/'notebooks/qwen3_teacher_scaling.ipynb')}
    ex.write_json(ROOT/'results/local-checks.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
