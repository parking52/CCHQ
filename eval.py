from io import open
from conllu import parse_incr
from numpy import loadtxt
import json
import sys

from library_viterby import initialize, \
    viterbi_forward, viterbi_backward, preprocess


def tokenlist_to_sentence(tokenlist):
    sentence = []
    tags = []
    for word in tokenlist:
        sentence.append(word['lemma'])
        tags.append(word['upos'])
    return sentence, tags


def predict(states, A, B, vocab, tag_counts, processed_text_corpus):

    best_probs, best_paths = initialize(states, tag_counts, A, B, processed_text_corpus, vocab)
    best_probs, best_paths = viterbi_forward(A, B, processed_text_corpus, best_probs, best_paths, vocab)
    predictions = viterbi_backward(best_probs, best_paths, processed_text_corpus, states)

    return predictions


def read_training_artifacts():
    A = loadtxt('artifacts/model/transition_matrix.csv', delimiter=',')
    B = loadtxt('artifacts/model/emission_matrix.csv', delimiter=',')
    vocab = json.load(open("artifacts/model/vocab.json"))
    tag_counts = json.load(open("artifacts/model/tag_counts.json"))
    with open('artifacts/model/states.txt', 'r') as filehandle:
        states = json.load(filehandle)

    return A, B, vocab, tag_counts, states


def create_test_words_file(data_file, word_file_path):
    with open(word_file_path, 'w') as the_file:
        for tokenlist in parse_incr(open(data_file, "r", encoding="utf-8")):
            sentence, tags = tokenlist_to_sentence(tokenlist)
            for word in sentence:
                the_file.write(f'{word}\n')


def compare_predictions(data_file, predictions):
    count = 0
    current_count = 0
    count_correct = 0
    for tokenlist in parse_incr(open(data_file, "r", encoding="utf-8")):
        read_sentence, true_tags = tokenlist_to_sentence(tokenlist)
        for word_idx in range(len(read_sentence)):
            if true_tags[word_idx] == predictions[word_idx + current_count]:
                count_correct += 1
            count += 1
        current_count = count
    print(f'{count} words have been evaluated')
    print(f'{count_correct} words were accurate')
    accuracy = count_correct / count
    print(f'accuracy is {accuracy}')


if __name__ == "__main__":

    try:
        data_file = sys.argv[1]
    except IndexError:
        print('using default file')
        data_file = "artifacts/source/en_gum-ud-test.conllu"

    word_file_path = 'artifacts/source/gum_test_words.txt'

    create_test_words_file(data_file, word_file_path)

    A, B, vocab, tag_counts, states = read_training_artifacts()
    _, processed_text_corpus = preprocess(vocab, "artifacts/source/gum_test_words.txt")
    predictions = predict(states, A, B, vocab, tag_counts, processed_text_corpus)

    compare_predictions(data_file, predictions)



