import sys

from cfb_model import print_report
from cfb_analysis import print_monte_carlo, print_pca_summary

if __name__ == "__main__":
    print_report()
    if "--montecarlo" in sys.argv:
        print()
        print_monte_carlo()
    if "--pca" in sys.argv:
        print()
        print_pca_summary()
