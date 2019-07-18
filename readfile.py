import pickle

filename = "pre_compute/autoalmostSurePolicy.pkl"
with open(filename, "rb") as f1:
    almostpolicy = pickle.load(f1)

print(almostpolicy[1])