"""CPU checks against the pinned benchmark recipes; no model weights are loaded."""
import argparse
import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment as ex
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.clients.base import TopLogprobs
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.datasets import TrainDataset, DataSource, qwen_messages_to_element
from cartridges.structs import Conversation, read_conversations, write_conversations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    cfg = ex.read_json(ROOT / 'configs/experiment.json')
    ex.verify_source(cfg)
    assert torch.__version__.split('+')[0] == '2.6.0'
    import transformers
    assert transformers.__version__ == '4.53.0'
    tokenizer = ex.tokenizer_for(cfg)
    other = ex.tokenizer_for(cfg, '8B')
    assert tokenizer.get_vocab() == other.get_vocab() and tokenizer.chat_template == other.chat_template
    checks = ['pinned_sources', 'shared_pytorch_transformers_environment', 'tokenizer_identity']
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for benchmark in cfg['benchmarks']:
            original, synth, settings = ex.recipe(cfg, benchmark, 'main')
            assert settings['samples'] == 131072
            assert synth.batch_size == 32
            assert ex.resource_config(cfg, benchmark, 'main').seed_prompts == synth.synthesizer.resources[0].seed_prompts
            for condition, pair in cfg['conditions'].items():
                directory = out / f"G{pair['generator']}"
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
        # Native client request/response payloads stay unchanged; only prompt IDs are recorded.
        client = ex.RecordingClient.__new__(ex.RecordingClient)
        client.tokenizer = tokenizer
        for thinking in [True, False]:
            request = {'messages':[{'role':'system','content':'Source.'},{'role':'user','content':'Question?'}],
                       'top_logprobs':20,'apply_chat_template_overrides':{'enable_thinking':thinking}}
            with patch.object(TokasaurusClient, '_send_requests', new=AsyncMock(return_value=['native_response'])) as send:
                result = asyncio.run(client._send_requests([request], 'batch-0'))
                assert result == ['native_response']
                send.assert_awaited_once_with([request], 'batch-0', use_cartridge_endpoint=False)
            expected = tokenizer.apply_chat_template(request['messages'], add_generation_prompt=True, enable_thinking=thinking)
            assert client.answer_prompts == [expected]
        checks.append('native_client_passthrough_and_scoring_prompt_capture')
        answer = tokenizer.encode('Answer.<|im_end|>', add_special_tokens=False)
        logp = np.log(np.tile([.8,.1995], (len(answer),1))).astype(np.float32)
        flat = TopLogprobs(logp,np.tile([20,21],(len(answer),1))).flatten()
        raw_user_ids = tokenizer.encode('Question?<|im_end|>',add_special_tokens=False)
        row = Conversation(messages=[Conversation.Message('Question?','user',raw_user_ids),
                                     Conversation.Message('Answer.','assistant',answer,flat)],
                           system_prompt='Source.',metadata={'prompt_ids':client.answer_prompts[0]})
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
    notebook = ex.read_json(ROOT/'notebooks/qwen3_teacher_scaling.ipynb')
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'notebook','exec')
            assert cell['execution_count'] is None and not cell['outputs']
    checks.append('notebook_syntax')
    titles = ['# @title 데이터 준비', '# @title 합성과 scoring', '# @title 학습 또는 재평가']
    cells = {''.join(c['source']).splitlines()[0]: ''.join(c['source'])
             for c in notebook['cells'] if c['cell_type'] == 'code'}
    for mode in ['train', 'evaluate']:
        for selected in [['A'], ['B'], ['C'], ['D'], list('ABCD')]:
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
                for generator, group in [('4B', 'AB'), ('8B', 'CD')]:
                    if any(c in selected for c in group):
                        expected.append(('synthesize', {'generator': generator}))
                        for c in group:
                            if c in selected:
                                expected.append(('score', {'generator': generator,
                                    'teacher': '4B' if c in 'AC' else '8B'}))
            expected.extend((mode, {'condition': c, 'seed': seed})
                            for seed in [42, 123] for c in selected)
            assert calls == expected, (mode, selected, calls)
    checks.append('notebook_train_evaluate_routing')
    report = {'status':'passed','device':'cpu','model_weights_loaded':False,'checks':checks,
              'protocol_version':cfg['protocol_version'],'code_sha256':ex.sha(ROOT/'experiment.py'),
              'patch_sha256':ex.sha(ROOT/'patches/cartridges.patch'),'notebook_sha256':ex.sha(ROOT/'notebooks/qwen3_teacher_scaling.ipynb')}
    ex.write_json(ROOT/'results/local-checks.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
