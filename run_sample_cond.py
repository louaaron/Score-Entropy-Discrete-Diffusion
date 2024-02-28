import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import sampling


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--prefix", type=str, default="Hi, my name is")
    parser.add_argument("--suffix", type=str, default=" and that's why I'm late.")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    prefix_ids = tokenizer(args.prefix).input_ids
    suffix_ids = tokenizer(args.suffix).input_ids
    input_ids = prefix_ids + suffix_ids
    input_locs = list(range(len(prefix_ids))) + list(range(1024-len(suffix_ids), 1024))

    # more generaly commands can be defined with something like below:
    # input_ids = [0, 1, 512, 8080, 50256, 20000]
    # input_locs = [5, 6, 19, 20, 1000, 10001]


    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device, proj_fun=proj_fun
    )

    samples = proj_fun(sampling_fn(model))

    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        print(i)
        print("=================================================")

if __name__=="__main__":
    main()