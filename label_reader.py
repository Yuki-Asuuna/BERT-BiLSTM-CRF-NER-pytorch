import pickle
with open("output\label2id.pkl", "rb") as f:
    label2id = pickle.load(f)
    print(label2id)

with open("output\label_list.pkl", "rb") as f:
    label2id = pickle.load(f)
    print(label2id)