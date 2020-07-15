from library_viterby import create_dictionaries, create_transition_matrix, create_emission_matrix, initialize, \
    viterbi_forward, viterbi_backward, preprocess
from eval import tokenlist_to_sentence
from conllu import parse_incr
from numpy import savetxt


data_file = open("artifacts/source/en_gum-ud-train.conllu", "r", encoding="utf-8")

all_words = []
training_corpus = []

count = 0

for tokenlist in parse_incr(data_file):
    count +=1
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

savetxt('artifacts/model/transition_matrix.csv', A)
savetxt('artifacts/model/emission_matrix.csv', A)


_, processed_text_corpus = preprocess(vocab, "artifacts/source/gum_test_words.txt")


from eval import predict
pred = predict(states, A, B, vocab, tag_counts, processed_text_corpus)

count = 0
current_count = 0
count_correct = 0

data_file = open("artifacts/source/en_gum-ud-test.conllu", "r", encoding="utf-8")

for tokenlist in parse_incr(data_file):
    read_sentence, true_tags = tokenlist_to_sentence(tokenlist)
    print(tokenlist)
    #predicted_tags = predict(sentence)
    for word_idx in range(len(read_sentence)):
        if true_tags[word_idx] == pred[word_idx+current_count]:
            count_correct += 1
        count += 1
    current_count = count

print(f'{count} words have been evaluated')
print(f'{count_correct} words were accurate')
accuracy = count_correct / count
print(f'accuracy is {accuracy}')


