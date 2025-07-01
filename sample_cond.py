import torch
import argparse
from typing import Optional, List, Union
import os

from transformers import PreTrainedTokenizerFast

from axolotl import sampling
from axolotl.load_model import load_model
from axolotl.utils import float_list_or_testing
from axolotl.visualization import plot_sequence_logo_and_create_gif

def get_args():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="/home/kkj/axolotl/exp_local/IPR036736_90_grouped/2025.05.23/110208", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024) # remember to add 2 for prefix and suffix
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--predictor", type=str, default="analytic", choices=sampling._PREDICTORS.keys())
    parser.add_argument("--denoise", type=bool, default=True)
    parser.add_argument("--cfg_w", type=float_list_or_testing, default=1.0)
    parser.add_argument("--label", type=str, default=None, choices=['prokaryotic', 'eukaryotic', 'random'])
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--input", type=str, default="SGGHDLSDFL")
    parser.add_argument("--input_locations", type=List[int], default=[4,5])
    parser.add_argument("--output", type=str, default="samples_cond")
    parser.add_argument("--name", type=str, default="cond_sample")
    parser.add_argument("--output_x0_predictions", type=bool, default=False, help="Whether to output x0 predictions")
    parser.add_argument("--output_intermediates", type=bool, default=True, help="Whether to output intermediate samples")
    parser.add_argument("--make_x0_gif", type=bool, default=True, help="Whether to make a GIF of x0 predictions")
    args = parser.parse_args()
    return args

def sample_conditional(model_path: str,
                       tokenizer: PreTrainedTokenizerFast,
                       input: str,
                       input_locations: List[int],
                       length: int,
                       batch_size: int = 1,
                       steps: int = 1024,
                       predictor: str = "ancestral_x0",
                       denoise: bool = True,
                       cfg_w: Union[float, List[float], str] = 1.0,
                       label: str = None,
                       output: str = "samples_cond",
                       name: str = "cond_sample",
                       output_x0_predictions: bool = False,
                       output_intermediates: bool = False,
                       make_x0_gif: bool = False,
):

    # more generally commands can be defined with something like below:
    # input = "AFGKLM"
    # input_locations = [5, 6, 19, 20, 1000, 10001]

    input_ids = tokenizer(input)['input_ids'][1:-1]

    assert len(input_ids) <= length, "Input length must be less than or equal to the specified length"
    assert max(input_locations) < length, "Input locations must be less than the specified length"

    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(batch_size, 1)
    batch_dims = (batch_size, length)

    def proj_fun(x):
        x[:, input_locations] = input_ids
        return x

    device = torch.device('cuda')
    model, graph, noise = load_model(model_path, device)
    
    output = output + ".txt"

    sampling_fn = sampling.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims, 
        predictor=predictor,
        steps=steps, 
        denoise=denoise,
        device=device,
        proj_fun=proj_fun,
        cfg=cfg_w,
        label=label,
        num_labels=model.num_labels,
        use_tqdm=True,
        prediction_type="x0",
        return_x0=output_x0_predictions,
        return_intermediates=output_intermediates,
    )

    samples, sampling_label, sampling_cfg_w, x0_predictions, intermediates = sampling_fn(model)
    samples = proj_fun(samples)
    sequences = tokenizer.batch_decode(samples)
    
    sampling.write_samples(output=output, sequences=sequences, sampling_label=sampling_label, sampling_cfg_w=sampling_cfg_w, name=name, steps=steps)
    
    if x0_predictions is not None:
        x0_predictions_outfolder = output.replace(".txt", "_x0_predictions")
        os.makedirs(x0_predictions_outfolder, exist_ok=True)
        for i, x0_seq in enumerate(x0_predictions):
            x0_seq = tokenizer.batch_decode(x0_seq.argmax(dim=-1), skip_special_tokens=True)
            sampling.write_samples(output=f"{x0_predictions_outfolder}/{i}.fasta", sequences=x0_seq, sampling_label=sampling_label, sampling_cfg_w=sampling_cfg_w, name=name + "_x0", steps=i, header=False, boltz_header=True)

    if make_x0_gif and x0_predictions is not None:
        x0_predictions_tensor = torch.stack(x0_predictions, dim=0).cpu().permute(1,0,3,2).numpy()
        for i, giftensor in enumerate(x0_predictions_tensor):
            plot_sequence_logo_and_create_gif(giftensor, positions_per_line=64, ylim=(0, 1), dpi=100, output_gif_path=f"{output.replace('.txt', f'_x0_predictions_{i}.gif')}", png_dir="sequence_logo_pngs", num_processes=10)

    if intermediates is not None:
        intermediates_outfile = output.replace(".txt", "_intermediates.txt")
        for i, intermediate in enumerate(intermediates):
            intermediate_seq = tokenizer.batch_decode(intermediate)
            sampling.write_samples(output=intermediates_outfile, sequences=intermediate_seq, sampling_label=sampling_label, sampling_cfg_w=sampling_cfg_w, name=name + "_intermediate", steps=i, header=False)


def mask_sequence(
    sequence, # e.g "MGLSDGEWQLVLNVWGKVEADVAGHGQ",
    masked_locs, # e.g [(0,5), (10,15), 21, 23],
    add_bos: bool = True,
    add_eos: bool = True,
):
    if not isinstance(masked_locs, list):
        raise TypeError("masked_locs must be a list of integers or tuples")
    if not all(isinstance(il, (int, tuple)) for il in masked_locs):
        raise TypeError("masked_locs must be a list of integers or tuples")
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")

    for il in masked_locs:
        if type(il) == tuple:
            masked_sequence = masked_sequence[:il[0]] + "_"*(il[1]-il[0]) + masked_sequence[il[1]:]
        elif type(il) == int:
            masked_sequence = masked_sequence[:il] + "_" + masked_sequence[il+1:]
    if add_bos:
        masked_sequence = "[" + masked_sequence
        sequence = "[" + sequence
    if add_eos:
        masked_sequence = masked_sequence + "]"
        sequence = sequence + "]"
    print("sequence before and after masking:")
    print(sequence)
    print(masked_sequence)


def preprocess_masked_sequence(
    masked_sequence: str,
) -> tuple[List[int], str, int]:
    """
    Preprocess the masked sequence to create a list of input positions and string of the amino tokens to put at these positions.
    Args:
        masked_sequence (str): The sequence with masked positions represented by underscores.
    Returns:
        tuple: A tuple containing:
            - input_locs (List[int]): A list of indices where the sequence is not masked (i.e., where the character is not "_").
            - input_sequence (str): The sequence with only the non-masked characters.
            - length (int): The length of the input sequence
    """
        
    # preprocess to get the correct input format
    input_locs = []
    length = len(masked_sequence)

    for i, c in enumerate(masked_sequence):
        if c != "_":
            input_locs.append(i)
    input_sequence = "".join([masked_sequence[i] for i in input_locs])

    return input_locs, input_sequence, length

if __name__=="__main__":
    args = get_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/axolotl/tokenizer/tokenizer_absorb')

    if args.input is not None: # If input is provided, use it instead of prefix and suffix
        assert args.input_locations is not None, "If input is provided, input_locations must be provided as well"
        print("Using input and input_locations instead of prefix and suffix")
        input_sequence = args.input
        input_locs = args.input_locations
    else: # If input is not provided, use prefix and suffix for conditional sampling
        assert args.prefix is not None or args.suffix is not None, "If input is not provided, prefix and/or suffix must be provided"
        input_sequence = args.prefix + args.suffix
        input_locs = list(range(len(args.prefix))) + list(range(args.length-len(args.suffix), args.length))

    sample_conditional(
        model_path=args.model_path,
        tokenizer=tokenizer,
        input=input_sequence,
        input_locations=input_locs,
        length=args.length,
        batch_size=args.batch_size,
        steps=args.steps,
        predictor=args.predictor,
        denoise=args.denoise,
        cfg_w=args.cfg_w,
        label=args.label,
        output=args.output,
        name=args.name,
        output_x0_predictions=args.output_x0_predictions,
        output_intermediates=args.output_intermediates,
        make_x0_gif=args.make_x0_gif,
    )