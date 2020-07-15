# CCHQ

### Assumptions
1. 

### Modelling decisions
I went with the latest algorithm I just discovered: Viterbi.
As I am yet unfamiliar with NLP with Neural Nets, I went with an easier algorithm.
Most of the library comes from the Coursera class. From my point of view, I would use libraries anyway, so I chose to use one I coded.

### Testing strategy
I looked at how many words were correctly predicted. All words are treated equally.
I am not aware of clever adjustement in the accuracy on the POS tagging problem. 
I could imagine that some tags could be ignored (SYM, PUNCT, X).
The accuracy is 0.85.

### Trade-offs
1. Simple algorithm - but quick to train.
2. Juggling around data format, almost lost as much time that I saved.
3. The eval framework should be independent of model assumptions - I would work on that next.


### Time taken
I took around 1 hour per day for 4 days. 
An embarassing amount of time was spent juggling around the unfamiliar data formats. 
I also took a coursera class as the NLP class happened to be under free trial.
