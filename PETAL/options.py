import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
    def initialize_parser(self):
        self.parser.add_argument('--gpu_ids', type=str, default="0", help="the GPUs to use")
        self.parser.add_argument('--target_model', type=str, default="pythia-6.9b", help="the model to attack")
        self.parser.add_argument('--surrogate_model', type=str, default="gpt2-xl")
        self.parser.add_argument('--embedding_model', type=str, default="all-MiniLM-L6-v2", help="the model used to compute semantic similarity")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument('--data', type=str, default="WikiMIA", help="the dataset to evaluate")
        self.parser.add_argument('--length', type=int, default=32, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
        self.parser.add_argument('--decoding', type=str, default="greedy", help="the decoding strategy to use")




