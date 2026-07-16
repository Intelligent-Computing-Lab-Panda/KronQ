# --- repo path bootstrap: make common/ runtime/ calib/ importable from any entry point ---
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not all(_os.path.isdir(_os.path.join(_d, s)) for s in ('common', 'runtime', 'calib')):
    _d = _os.path.dirname(_d)
for _s in ('common', 'runtime', 'calib'):
    _p = _os.path.join(_d, _s)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import gptaq_utils
import kronq_utils
import eval_utils
import hadamard_utils


def add_aq(model, args):
    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio

            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)

            if 'lm_head' in name: #Skip lm_head quantization
                layer_input_bits = 16

            if 'down_proj' in name:  #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                          "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn,
                            rope_function_name,
                            config=model.config,
                            **k_quant_config)


def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    # Rotate the weights
    if args.rotate:
        assert "llama" in args.model.lower(), \
            "--rotate (QuaRot) is validated on Llama only; for Qwen use the weight-only recipe without --rotate."
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had

    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.enable_aq_calibration:
        add_aq(model, args)

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            # Unrotated-save ckpts carry biip_* buffers: pre-register so load_state_dict
            # fills them, then wrap each Linear with online BiIP (no-op for GPTQ/GPTAQ).
            from biip_linear import register_biip_buffers_from_state_dict, wrap_linears_with_biip
            register_biip_buffers_from_state_dict(model, save_dict["model"])
            model.load_state_dict(save_dict["model"], strict=False)
            n_wrapped = wrap_linears_with_biip(model)
            if n_wrapped > 0:
                print(f"Wrapped {n_wrapped} Linears with online BiIP")
            # Optional: compress to real low-bit storage to validate the deploy
            # pipeline (pass saved w_quantizers so scale/zero match).
            simulate_bits = args.simulate_lowbit_storage
            if simulate_bits:
                from lowbit_utils import compress_model_to_lowbit
                w_q = save_dict.get("w_quantizers", None)
                n_comp, err = compress_model_to_lowbit(model, simulate_bits, w_quantizers=w_q)
                print(f"Compressed {n_comp} Linears to {simulate_bits}-bit (max err {err:.2e})")
            
        elif not args.w_rtn: # Calibrated weight quantization (KronQ / GPTAQ / GPTQ)
            assert any(k in args.model.lower() for k in ('llama', 'qwen')), \
                "This release supports Llama and Qwen."

            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            if args.bi_calibration:
                quantizers = kronq_utils.kronq_fwrd(model, trainloader, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
            elif args.asym_calibrate:
                quantizers = gptaq_utils.gptaq_fwrd(model, trainloader, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
            else:
                quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

        # After --save_unrotated_kronq the Linears hold Q in rotated space + biip_*
        # buffers; wrap them so post-save eval uses the online-BiIP forward.
        if args.save_unrotated_kronq and not args.load_qmodel_path:
            from biip_linear import wrap_linears_with_biip
            n_wrapped = wrap_linears_with_biip(model)
            if n_wrapped > 0:
                print(f"Wrapped {n_wrapped} Linears with online BiIP for eval")

        # --save_packed: write the packed int4/int2 HF artifact directly (no fp16
        # step); compresses in place, so the PPL/lm_eval below run on the packed model.
        if args.save_packed and not args.load_qmodel_path:
            assert args.save_unrotated_kronq, "--save_packed requires --save_unrotated_kronq"
            assert args.w_bits in (2, 4), "--save_packed supports w_bits 2 or 4 only (int3 not packable)"
            from lowbit_utils import compress_model_to_lowbit
            import packed_io
            n_comp, err = compress_model_to_lowbit(model, args.w_bits,
                                                   w_quantizers=save_dict.get("w_quantizers"))
            nt, nc = packed_io.save_packed(model, args.save_packed, args.w_bits)
            print(f"Saved PACKED int{args.w_bits}: {nt} tensors, {nc} linears, "
                  f"max pack err {err:.2e} -> {args.save_packed}")

    if not args.enable_aq_calibration:
        add_aq(model, args)

    # Evaluating on dataset
    if args.eval_seqlen:
        model.seqlen = args.eval_seqlen
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )

    try:
        dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    except torch.cuda.OutOfMemoryError:
        # Large-vocab models can OOM in the logits pass of the PPL eval.
        # PPL isn't needed for --lm_eval; skip it and continue.
        print("[WARN] WikiText2 PPL eval OOM — skipping (continuing to lm_eval).", flush=True)
        torch.cuda.empty_cache()
        dataset_ppl = float('nan')

    if args.wandb:
            wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    if not args.lm_eval:
        return
    else:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, token=args.hf_token)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    task_names = args.tasks
    gen_kwargs_str = "max_gen_toks=2048"
    # Filter kwargs to those the installed lm_eval supports (older versions lack
    # apply_chat_template / fewshot_as_multiturn / gen_kwargs).
    import inspect as _inspect
    _se_kwargs = dict(
        tasks=task_names, batch_size=args.lm_eval_batch_size,
        gen_kwargs=gen_kwargs_str,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.apply_chat_template,
    )
    _accepted = _inspect.signature(lm_eval.simple_evaluate).parameters
    _se_kwargs = {k: v for k, v in _se_kwargs.items() if k in _accepted}
    results = lm_eval.simple_evaluate(hflm, **_se_kwargs)['results']

    # Pick the most informative scalar metric per task. lm-eval names metrics
    # like "acc,none", "acc_norm,none", "exact_match,none", "exact_match,custom-extract".
    def pick_metric(r):
        for k in ('acc_norm,none', 'acc,none', 'exact_match,strict-match',
                  'exact_match,flexible-extract', 'exact_match,none'):
            if k in r:
                return r[k]
        # fallback: first numeric value
        for k, v in r.items():
            if isinstance(v, (int, float)):
                return v
        return float('nan')

    metric_vals = {task: round(pick_metric(result), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)
    print("FULL RESULTS:")
    import json
    print(json.dumps(results, indent=2, default=str))

    if args.wandb:
        wandb.log(metric_vals)


if __name__ == '__main__':
    main()
