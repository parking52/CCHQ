from io import open
from conllu import parse_incr
from numpy import loadtxt

from library_viterby import create_dictionaries, create_transition_matrix, create_emission_matrix, initialize, \
    viterbi_forward, viterbi_backward, preprocess

def tokenlist_to_sentence(tokenlist):
    sentence = []
    tags = []
    for word in tokenlist:
        sentence.append(word['lemma'])
        tags.append(word['upos'])
    # sentence.append('\n')
    # tags.append('--n--"')
    return sentence, tags


def predict(states, A, B, vocab, tag_counts, processed_text_corpus):

    #A = loadtxt('artifacts/model/transition_matrix.csv', delimiter=',')
    #B = loadtxt('artifacts/model/emission_matrix.csv', delimiter=',')

    best_probs, best_paths = initialize(states, tag_counts, A, B, processed_text_corpus, vocab)
    best_probs, best_paths = viterbi_forward(A, B, processed_text_corpus, best_probs, best_paths, vocab)
    predictions = viterbi_backward(best_probs, best_paths, processed_text_corpus, states)

    return predictions

if __name__ == "__main__":

    print('Checking accuracy')

    data_file = open("artifacts/source/en_gum-ud-test.conllu", "r", encoding="utf-8")
    word_file_path = 'artifacts/source/gum_test_words.txt'
    with open(word_file_path, 'a') as the_file:

        for tokenlist in parse_incr(data_file):
            sentence, tags = tokenlist_to_sentence(tokenlist)
            for word in sentence:
                the_file.write(f'{word}\n')

    data_file = open("artifacts/source/en_gum-ud-test.conllu", "r", encoding="utf-8")
    for tokenlist in parse_incr(data_file):
        count = 0
        count_correct = 0
        sentence, tags = tokenlist_to_sentence(tokenlist)
        predicted_tags = predict(sentence)
        for word_idx in range(len(sentence)):
            if tags[word_idx] == predicted_tags[word_idx]:
                count_correct += 1
            count += 1


    accuracy = count_correct / count

    print(f'accuracy is {accuracy}')




