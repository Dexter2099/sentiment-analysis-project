# main.py

import argparse
from src.train import train
from src.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis on IMDb Reviews")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help="Mode to run the script: 'train' or 'eval'")
    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting training...")
        train()
    elif args.mode == 'eval':
        print("Starting evaluation...")
        evaluate()

if __name__ == "__main__":
    main()
