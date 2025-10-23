import pickle, dill

with open('artifacts/preprocessor.pkl', 'rb') as f:
    pre = pickle.load(f)
print('Preprocessor:', type(pre))

with open('artifacts/model.pkl', 'rb') as f:
    model = dill.load(f)
print('Model:', type(model))