"""
Script to sample from a pretrained NICE model.
"""
import argparse
from pathlib import Path

import torch
import torchvision

from sde_sampler.distr import nice
from sde_sampler.distr.base import DATA_DIR


def prepare_data(x, mean=None, reverse=False):
    """Prepares data for NICE.

    In training mode, flatten and dequantize the input.
    In inference mode, reshape tensor into image size.

    Args:
        x: input minibatch.
        mean: center of original dataset.
        reverse: True if in inference mode, False if in training mode.
    Returns:
        transformed data.
    """
    if reverse:
        width = int(x.shape[-1] ** 0.5)
        assert width * width == x.shape[-1]
        x += mean
        x = x.reshape((x.shape[0], 1, width, width))
    else:
        assert x.shape[-1] == x.shape[-2]
        x = x.reshape(x.shape[0], -1)
        x -= mean
    return x


def sample_nice(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the checkpoint
    checkpoint = torch.load(DATA_DIR / "nice.pt", map_location=device)
    
    # Extract model parameters from checkpoint
    batch_size = checkpoint["batch_size"]
    latent = checkpoint["latent"]
    coupling = checkpoint["coupling"]
    mid_dim = checkpoint["mid_dim"]
    hidden = checkpoint["hidden"]
    mask_config = checkpoint["mask_config"]
    
    # Determine full dimension from model state dict
    model_state = checkpoint["model_state_dict"]
    full_dim = 14 * 14
    resize = 14
    MNIST_SIZE = 28
    
    # Load and resize the mean the same way as in train_nice.py
    mean = torch.load(DATA_DIR / "mnist_mean.pt").reshape((1, MNIST_SIZE, MNIST_SIZE))
    mean = torchvision.transforms.Resize(size=(resize, resize), antialias=True)(mean).reshape(
        (1, full_dim)
    )
    
    # Create the prior distribution based on the saved config
    if latent == "normal":
        prior = torch.distributions.Normal(
            torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)
        )
    elif latent == "logistic":
        prior = nice.StandardLogistic()
    
    # Create the model
    flow = nice.NiceModel(
        prior=prior,
        coupling=coupling,
        in_out_dim=full_dim,
        mid_dim=mid_dim,
        hidden=hidden,
        mask_config=mask_config,
    ).to(device)
    
    # Load the model state dict
    flow.load_state_dict(checkpoint["model_state_dict"])
    flow.eval()
    
    # Generate samples
    with torch.no_grad():
        samples = flow.sample(args.n_sample).cpu()
        
        # Save raw samples
        torch.save(samples, DATA_DIR / f'nice_mnist_sample_{args.n_sample}.pt')
        
        # Always save as images for visualization
        samples_img = prepare_data(samples, mean=mean, reverse=True)
        output_dir = Path(DATA_DIR).parent / "logs" / "nice_samples"
        output_dir.mkdir(exist_ok=True, parents=True)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(samples_img),
            output_dir / f"nice_samples_{args.n_sample}.png",
        )
        print(f"Samples saved as image at {output_dir}/nice_samples_{args.n_sample}.png")
    
    print(f"Generated {args.n_sample} samples from NICE model")
    print(f"Saved to {DATA_DIR}/nice_mnist_sample_{args.n_sample}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sample from pretrained NICE model")
    parser.add_argument(
        "--n_sample", help="number of samples to generate", type=int, default=1000
    )
    parser.add_argument(
        "--save_images", help="save samples as images", action="store_true"
    )
    args = parser.parse_args()
    sample_nice(args)
