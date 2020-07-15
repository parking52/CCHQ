# usage generate.py test_generate

import sys
from eval import predict
import tokenize
import nltk
nltk.download('punkt')

try:
    sys.argv[1]
except IndexError:
    print('No file argument provided')


def process_file(file_path):

    print('---------')
    with open(file_path, 'r') as my_file:

        for line in my_file:
            nltk_tokens = nltk.word_tokenize(line.strip('\n'))
            ## TODO Something with the punctuation

            predicted_tags = predict(nltk_tokens)

            print(nltk_tokens)
            print(predicted_tags)
            print('---------')


if __name__ == "__main__":
    process_file(sys.argv[1])