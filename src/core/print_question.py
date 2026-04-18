import pickle

with open('data/data/wikidata_big/questions/valid.pickle', 'rb') as f:
    questions = pickle.load(f)
    
print(questions[0])