import argparse
from clt_tracing.feature_extractor import train_autoencoder

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acts_path",   type=str, required=True)
    p.add_argument("--layers",      type=int, nargs="+", required=True)
    p.add_argument("--input_dim",   type=int,         required=True)
    p.add_argument("--feature_dim", type=int,         required=True)
    p.add_argument("--device",      type=str, default="cpu")
    p.add_argument("--out",         type=str, required=True)
    args = p.parse_args()

    train_autoencoder(
        args.acts_path, args.layers,
        args.input_dim, args.feature_dim,
        args.device, args.out
    )
