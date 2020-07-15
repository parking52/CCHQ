import sys
from eval import predict
from library_viterby import preprocess
from eval import read_training_artifacts

if __name__ == "__main__":

    try:
        data_file = sys.argv[1]
    except IndexError:
        print('using default file')
        data_file = "test_generate.txt"

    word_file_path = 'artifacts/source/temp_generate.txt'

    words = []
    with open(word_file_path, 'w') as output_file:
        with open(data_file, 'r') as input_file:
            for line in input_file:
                for word in eval(line):
                    words.append(word)
                    output_file.write(f'{word}\n')
    input_file.close()
    output_file.close()

    A, B, vocab, tag_counts, states = read_training_artifacts()
    _, processed_text_corpus = preprocess(vocab, word_file_path)
    predictions = predict(states, A, B, vocab, tag_counts, processed_text_corpus)

    print(words)
    print(predictions)
