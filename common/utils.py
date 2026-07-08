"""Argument parsing, multi-GPU model distribution, and memory utilities."""

import argparse
import pprint
import torch
import os
from datetime import datetime
import logging

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-70b-hf',
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-70B',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.2-1B-Instruct',
]
supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def llama_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, 'groupsize should be greater than 1!'

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsize = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize


# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(message)s"
            else:
                self._style._fmt = "%(levelname)s: %(message)s"
            return super().format(record)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # clear existing handlers

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())
    root.addHandler(file_handler)


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;', choices=supported_models)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets, )
    parser.add_argument('--eval_seqlen', type=int, default=None,
                        help='Override model.seqlen for PPL eval only (default: use model.seqlen). '
                             'e.g. 8192 for Llama-3 models.')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--bsz', type=int, default=32,
                        help='Batch-size for PPL evaluation (default:32)')

    # Rotation Arguments
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False,
                        help='Rotate the model, including online rotation for down- and out-projection. '
                             'K/Q are only rotated when quantizing the keys.')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--incoh_rotate', action=argparse.BooleanOptionalAction, default=False,
                        help='Incoherence pre/post-processing')
    parser.add_argument('--save_unrotated_kronq', action=argparse.BooleanOptionalAction, default=False,
                        help='With --incoh_rotate, skip the BiIP postprocess and save Q in rotated space '
                             'plus U/V/S_X/S_G buffers on each Linear (required for online-BiIP deployment). '
                             'No effect when --incoh_rotate is off.')
    parser.add_argument('--inter_layer_mp', type=int, default=None,
                        help='Inter-layer mixed precision: number of top sensitive '
                             'sublayer types to upgrade by +1 bit (see inter_layer_mp.SUBLAYER_RANK)')
    parser.add_argument('--use_gptaq', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.25,
                        help='Scale of the asymmetric-calibration P matrix (KronQ path)')
    parser.add_argument('--incoh_mode', type=str, default='full',
                        choices=['full', 'sv_only', 'sv_only_no_rescale', 'no_rescale', 'su_only',
                                 'rescale_x_only', 'rescale_g_only'])
    parser.add_argument('--incoh_kernel', type=str, default='kron', choices=['kron', 'had'],
                        help='Incoherence rotation kernel. `kron`: dense random orthogonal '
                             'butterfly (default). `had`: randomized Hadamard transform '
                             '(matches QuIP# deployment). Both give similar PPL.')
    parser.add_argument('--simulate_lowbit_storage', type=int, default=0, choices=[0, 2, 4],
                        help='If 2/4, compress every BiIPLinear to real packed int<bits> storage '
                             'with on-the-fly fused dequant+matvec, to demonstrate real-quant deploy PPL.')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='Number of bits for inputs of all Linear layers '
                             '(including down-projection and out-projection)')
    parser.add_argument('--a_groupsize', type=int, default=-1,
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Asymmetric activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--enable_aq_calibration', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable activation quantization in GPTQ(v2) (default: False)')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16,
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1,
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Asymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='Optimize clipping range for weight quantization '
                             '(the best clip ratio is found during quantization).')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPT(A)Q')
    parser.add_argument('--static_groups', action=argparse.BooleanOptionalAction, default=False,
                        help='static groups in GPT(A)Q (GPTAQ/KronQ paths)')
    parser.add_argument('--bi_calibration', action=argparse.BooleanOptionalAction, default=False,
                        help='enable KronQ bi-directional calibration')
    parser.add_argument('--asym_calibrate', action=argparse.BooleanOptionalAction, default=False,
                        help='enable GPTAQ calibration')
    parser.add_argument('--grad_dir', type=str, default=None,
                        help='Directory containing pre-computed FP gradients')

    # General Quantization Arguments
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for down-projection. If set, both weights and activations of this layer will be in INT8')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='Number of bits for V-cache quantization (needs no extra rotation)')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Asymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')

    parser.add_argument('--k_bits', type=int, default=16,
                        help='Number of bits for K-cache quantization (needs an extra rotation for keys/queries)')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Asymmetric K-cache quantization')
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False,
                        help='Pre-RoPE quantization for K-cache (not supported yet)')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')

    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path.')
    parser.add_argument('--save_qmodel_path', type=str, default=None,
                        help='Save the quantized model (fp16 deploy ckpt) to the specified .pth path.')
    parser.add_argument('--save_packed', type=str, default=None,
                        help='Save the model as PACKED int4/int2 (the HF artifact) to the specified '
                             'directory. Requires --save_unrotated_kronq and w_bits in {2,4} (int3 '
                             'is not packable). Writes model.safetensors + kronq_packed_config.json.')

    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)

    # Experiments Arguments
    parser.add_argument('--save_name', type=str, default=None, help='The path to save experiment data, '
                                                                    'including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].')
    parser.add_argument('--capture_layer_io', action=argparse.BooleanOptionalAction, default=False,
                        help='Capture the input and output of the specified decoder layer and dump into a file')
    parser.add_argument('--layer_idx', type=int, default=10, help='Which decoder layer to capture')

    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "boolq"],
    )
    parser.add_argument('--lm_eval_batch_size', type=int, default=32,
                        help='Batch size for evaluating with lm eval harness.')
    parser.add_argument('--apply_chat_template', action='store_true',
                        help='Apply the model chat template to zeroshot prompts.')
    parser.add_argument('--distribute', action=argparse.BooleanOptionalAction, default=False,
                        help='Distribute the model across multiple GPUs for evaluation.')

    args = parser.parse_args()
    if args.lm_eval:
        # lm_eval >= 0.4.2 replaced initialize_tasks + tasks.ALL_TASKS with TaskManager.
        try:
            from lm_eval.tasks import TaskManager
            _tm = TaskManager()
            _all_tasks = set(_tm.all_tasks)
            for task in args.tasks:
                if task not in _all_tasks:
                    raise ValueError(f"Invalid task: {task}")
        except ImportError:
            from lm_eval import tasks
            from lm_eval import utils as lm_eval_utils
            from lm_eval.tasks import initialize_tasks
            initialize_tasks()
            for task in args.tasks:
                if task not in lm_eval_utils.MultiChoice(tasks.ALL_TASKS):
                    raise ValueError(f"Invalid task: {task}")

    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    setattr(args, 'save_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.save_name))
    os.makedirs(args.save_path, exist_ok=True)

    config_logging(os.path.join(args.save_path, f'{args.save_name}.log'))

    assert args.k_pre_rope == False, 'Pre-RoPE quantization is not supported yet!'

    if args.wandb:
        assert args.wandb_id is not None and args.wandb_project is not None, 'WandB ID/project is not provided!'

    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    return args


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def distribute_model(model) -> None:
    """Distribute the model across all available GPUs (auto-balanced)."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split_module_classes)

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    logging.debug(f"device_map: {device_map}")
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        state_dict=model.state_dict(),
    )

    cleanup_memory()