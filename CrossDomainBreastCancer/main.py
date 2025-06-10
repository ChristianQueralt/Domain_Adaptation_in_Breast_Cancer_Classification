# main.py
import argparse

def main():
    parser = argparse.ArgumentParser(description="Cross-domain Breast Cancer Training")
    parser.add_argument('--experiment', type=str, required=True,
                        choices=[
                            'ddsm_bcdr',
                            'ddsm10_bcdr',
                            'bcdr_ddsm',
                            'bcdr_percent_transfer',
                            'bcdr_percent_transfer_shade'
                        ],
                        help='Experiment to run: ddsm_bcdr | ddsm10_bcdr | bcdr_ddsm | bcdr_percent_transfer | bcdr_percent_transfer_shade')
    args = parser.parse_args()

    if args.experiment == 'ddsm_bcdr':
        from experiments.train_ddsm_test_bcdr import run_experiment
        run_experiment()
    elif args.experiment == 'ddsm10_bcdr':
        from experiments.train_ddsm10_bcdr_test_bcdr import run_experiment
        run_experiment()
    elif args.experiment == 'bcdr_ddsm':
        from experiments.train_bcdr_test_ddsm import run_experiment
        run_experiment()
    elif args.experiment == 'bcdr_percent_transfer':
        from experiments.transfer_learning_BCDR_percentages import run_experiment
        run_experiment()
    elif args.experiment == 'bcdr_percent_transfer_shade':
        from experiments.transfer_learning_BCDR_percentages_shade import run_experiment
        run_experiment()
    else:
        raise ValueError("Unknown experiment selected")

if __name__ == "__main__":
    main()
