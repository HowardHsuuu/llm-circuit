import argparse
from clt_tracing.clt_model import train_clt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feat",        type=str, required=True)
    p.add_argument("--logits",      type=str, required=True)
    p.add_argument("--layers",      type=int, nargs="+", required=True)
    p.add_argument("--feature_dim", type=int,         required=True)
    p.add_argument("--vocab_size",  type=int,         required=True)
    p.add_argument("--device",      type=str, default="cpu")
    p.add_argument("--out",         type=str, required=True)
    args = p.parse_args()

    train_clt(
        args.feat, args.logits, args.layers,
        args.feature_dim, args.vocab_size,
        args.device, args.out
    )
