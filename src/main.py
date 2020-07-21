import argparse

from nn import NNModel


def create_parser():
    parser = argparse.ArgumentParser(prog='sqad', description='Quastion Answering with Stanford Question Answering Dataset', add_help=False)

    parent_group = parser.add_argument_group(title='Parameters')
    parent_group.add_argument('--help', '-h', help='Help')

    subparsers = parser.add_subparsers(dest='mode', title='Possible commands')

    train_parser = subparsers.add_parser('train', add_help=False, help='Train new model')
    train_parser.add_argument('--model', '-m', required=True, help='Name of model')

    test_parser = subparsers.add_parser('test', add_help=False, help='Test existing model')
    test_parser.add_argument('--model', '-m', required=True, help='Name of model')
    test_parser.add_argument('--context', '-c', required=True, help='Context or input for test model')
    test_parser.add_argument('--question', '-q', required=True, help='Question to context')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args()

    model = NNModel()

    if namespace.mode == 'train':
        model.train(namespace.model)
    elif namespace.mode == 'test':
        model.load_model(namespace.model)
        model.test(namespace.context, namespace.question)
    else:
        print('Incorrect mode')
