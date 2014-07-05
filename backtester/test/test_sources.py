import os

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(WORKING_DIR, 'data')
SYMBOL_LIST_FILE = os.path.join(DATA_DIR, 'stocklist.txt')
SYMBOL_LIST = ['TEST', 'TEST2', 'TEST3']
TEST_CSV_FILE = os.path.join(DATA_DIR, 'TEST.csv')
TEST_REVERSED_CSV_FILE = os.path.join(DATA_DIR, 'TEST_reversed.csv')
