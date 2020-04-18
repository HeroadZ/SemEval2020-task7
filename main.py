from util import util
import argparse


def main():
    # for reproducing
    util.setup_seed(66) 

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="turn on the train function", action="store_true")
    parser.add_argument("-a", "--showA", help="show the result of task A", action="store_true")
    parser.add_argument("-b", "--showB", help="show the result of task B", action="store_true")
    parser.add_argument("-s", "--show", help="show the results", action="store_true")
    args = parser.parse_args()
    if args.train:
        util.train()
    if args.showA:
        util.show_A_res()
    if args.showB:
        util.show_B_res()
    if args.show:
        util.show_res()


if __name__ == '__main__':
    main()
