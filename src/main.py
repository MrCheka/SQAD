import argparse

from nn import NNModel


def create_parser():
    parser = argparse.ArgumentParser(prog='sqad', description='Quastion Answering with Stanford Question Answering Dataset', add_help=False)

    parser.add_argument('--context', '-c', required=True, help='Context or input for test model')
    parser.add_argument('--question', '-q', required=True, help='Question to context')
    parser.add_argument('--help', '-h', action='help', help='Help')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args()

    model = NNModel()
    model.train()
    model.test(namespace.context, namespace.question)
