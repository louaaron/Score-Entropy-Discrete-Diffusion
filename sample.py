import torch
import argparse

from transformers import PreTrainedTokenizerFast
import torch.nn.functional as F

from axolotl import sampling
from axolotl.load_model import load_model
from axolotl.utils import float_list_or_testing
from axolotl.visualization import plot_sequence_logo_and_create_gif

from typing import List, Union
import os

def get_args():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="/home/kkj/axolotl/exp_local/IPR036736_90_grouped/2025.04.05/144904", type=str) #/home/kkj/axolotl/exp_local/UniRef50_grouped/2025.04.30/205511", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--predictor", type=str, default="ancestral_x0", choices=sampling._PREDICTORS.keys())
    parser.add_argument("--denoise", type=bool, default=True)
    parser.add_argument("--cfg_w", type=float_list_or_testing, default=1.0)
    parser.add_argument("--label", type=str, default='random', choices=['prokaryotic', 'eukaryotic', 'random'])
    parser.add_argument("--output", type=str, default="axolotl_samples_ACP-like")
    parser.add_argument("--name", type=str, default="sample")
    parser.add_argument("--output_x0_predictions", type=bool, default=True, help="Whether to output x0 predictions")
    parser.add_argument("--output_intermediates", type=bool, default=True, help="Whether to output intermediate samples")
    parser.add_argument("--make_x0_gif", type=bool, default=True, help="Whether to make a GIF of x0 predictions")
    args = parser.parse_args()
    return args

def sample(model_path: str,
           tokenizer: PreTrainedTokenizerFast,
           batch_size: int,
           length: int,
           steps: int,
           predictor: str,
           denoise: bool,
           cfg_w: Union[float, List[float], str],
           label: str,
           output: str,
           name: str,
           output_x0_predictions: bool = False,
           output_intermediates: bool = False,
           make_x0_gif: bool = False,
):
    if make_x0_gif and not output_x0_predictions:
        raise ValueError("If you want to make a GIF of x0 predictions, you need to set output_x0_predictions=True")
    
    device = torch.device('cuda')
    model, graph, noise = load_model(model_path, device)
    
    output = output + ".txt"

    sampling_fn = sampling.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(batch_size, length), 
        predictor=predictor,
        steps=steps, 
        denoise=denoise,
        cfg=cfg_w,
        label=label,
        num_labels=model.num_labels,
        device=device,
        use_tqdm=True,
        prediction_type="x0",
        return_x0=output_x0_predictions,
        return_intermediates=output_intermediates,
    )

    samples, sampling_label, sampling_cfg_w, x0_predictions, intermediates = sampling_fn(model)
    sequences = tokenizer.batch_decode(samples)

    sampling.write_samples(output=output, sequences=sequences, sampling_label=sampling_label, sampling_cfg_w=sampling_cfg_w, steps=steps, name=name)

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

if __name__=="__main__":
    args = get_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/axolotl/tokenizer/tokenizer_absorb')

    sample(model_path=args.model_path,
           tokenizer=tokenizer,
           batch_size=args.batch_size,
           length=args.length,
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