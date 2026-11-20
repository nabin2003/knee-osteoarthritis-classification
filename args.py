
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(
        description="Knee X-ray classification training"
    )

    parser.add_argument(
        '--csv_dir',
        type=str,
        default='data',
        help='Folder containing the fold CSV files'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='outputs',
        help='Folder for saving models and plots'
    )

    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help='Batch size'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )

    parser.add_argument(
        '--wd',
        type=float,
        default=0.0001,
        help='Weight decay'
    )

    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet18',
        help='Model backbone'
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    return args
