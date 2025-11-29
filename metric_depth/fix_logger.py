import logging

from util.utils import init_log


def main():
    print('', flush=True)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    logger.info('test')
    print('', flush=True)

if __name__ == '__main__':
    main()

