import pickle
import re

with open('test/names.pkl', 'rb') as f:
    test_names = pickle.load(f)
with open('test/elements.pkl', 'rb') as f:
    test_elements = pickle.load(f)

with open('train/names.pkl', 'rb') as f:
    train_names = pickle.load(f)
with open('train/elements.pkl', 'rb') as f:
    train_elements = pickle.load(f)



def padding(max_length, list):
    for s in list:
        s.extend(['<pad>']* (max_length - len(s)))
    return max_length, list

def helper(max_length, splitted_list):
    max_names, padded_names = padding(max_length, splitted_list)
    unique_name_tokens = sorted(list(set([item for sublist in padded_names for item in sublist])))
    return unique_name_tokens, padded_names
## NAMES

test_splitted_names = [re.findall(r'[\s,-]|[^,\s-]+', x) for x in test_names]
train_splitted_names = [re.findall(r'[\s,-]|[^,\s-]+', x) for x in train_names]

test_max_length = max([len(s) for s in test_splitted_names])
train_max_length = max([len(s) for s in train_splitted_names])

total_max = max(test_max_length, train_max_length)

test_name_tokens, test_padded_names = helper(total_max, test_splitted_names)
train_name_tokens, train_padded_names = helper(total_max, train_splitted_names)

unique_names = sorted(list(set(test_name_tokens + train_name_tokens)))
names_vocabulary = {w:i for i,w in enumerate(unique_names)}
print(f"Names Vocabulary: Length = {len(names_vocabulary)}")
print(names_vocabulary)

test_tokenized_names = [[names_vocabulary.get(item, item) for item in sublist] for sublist in test_padded_names]
train_tokenized_names = [[names_vocabulary.get(item, item) for item in sublist] for sublist in train_padded_names]

print(test_tokenized_names[0])
print(train_tokenized_names[0])

with open('test_names.pkl', 'wb') as f:
    pickle.dump(test_tokenized_names, f)
with open('train_names.pkl', 'wb') as f:
    pickle.dump(train_tokenized_names, f)
with open('names_vocabulary.pkl', 'wb') as f:
    pickle.dump(names_vocabulary, f)

## ELEMENTS

test_max_length = max([len(s) for s in test_elements])
train_max_length = max([len(s) for s in train_elements])

total_max = max(test_max_length, train_max_length)

test_elements_tokens, test_padded_elements = helper(total_max, test_elements)
train_elements_tokens, train_padded_elements = helper(total_max, train_elements)

unique_elements = sorted(list(set(test_elements_tokens + train_elements_tokens)))
elements_vocabulary = {w:i for i,w in enumerate(unique_elements)}
print(f"Elements Vocabulary: Length = {len(elements_vocabulary)}")
print(elements_vocabulary)

test_tokenized_elements = [[elements_vocabulary.get(item, item) for item in sublist] for sublist in test_padded_elements]
train_tokenized_elements = [[elements_vocabulary.get(item, item) for item in sublist] for sublist in train_padded_elements]

print(test_tokenized_elements[0])
print(train_tokenized_elements[0])

with open('test_elements.pkl', 'wb') as f:
    pickle.dump(test_tokenized_elements, f)
with open('train_elements.pkl', 'wb') as f:
    pickle.dump(train_tokenized_elements, f)
with open('elements_vocabulary.pkl', 'wb') as f:
    pickle.dump(elements_vocabulary, f)




