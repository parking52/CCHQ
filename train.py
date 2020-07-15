from library_viterby import create_dictionaries, create_transition_matrix, create_emission_matrix
from eval import tokenlist_to_sentence
from conllu import parse_incr
from numpy import savetxt
import json
import sys


all_words = []
training_corpus = []
count = 0

try:
    data_file = sys.argv[1]
except IndexError:
    print('using default file')
    data_file = "artifacts/source/en_gum-ud-train.conllu"

for tokenlist in parse_incr(open(data_file, "r", encoding="utf-8")):
    count += 1
    sentence, tags = tokenlist_to_sentence(tokenlist)
    for word, tag in zip(sentence, tags):
        all_words.append(word)
        training_corpus.append(f"{word}\t{tag}\n")
    training_corpus.append("\t\n")

# load in the training corpus


voc_l = list(set(all_words))
voc_l.append('--n--')
voc_l.append('--unk--')
voc_l.append('--unk_adj--')
voc_l.append('--unk_adv--')
voc_l.append('--unk_digit--')
voc_l.append('--unk_noun--')
voc_l.append('--unk_punct--')
voc_l.append('--unk_upper--')
voc_l.append('--unk_verb--')


vocab = {}
# Get the index of the corresponding words.
for i, word in enumerate(sorted(voc_l)):
    vocab[word] = i


alpha = 0.001

emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)

states = sorted(tag_counts.keys())


A = create_transition_matrix(alpha, tag_counts, transition_counts)
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))


def save_artifacts(A, B, states, vocab, tag_counts):
    savetxt('artifacts/model/transition_matrix.csv', A, delimiter=',')
    savetxt('artifacts/model/emission_matrix.csv', B, delimiter=',')
    with open('artifacts/model/states.txt', 'w') as filehandle:
        json.dump(states, filehandle)
    json.dump(vocab, open("artifacts/model/vocab.json", 'w'))
    json.dump(tag_counts, open("artifacts/model/tag_counts.json", 'w'))


save_artifacts(A, B, states, vocab, tag_counts)
