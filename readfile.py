import pickle

filename = "all_policies(1).pkl"
with open(filename, "rb") as f1:
    Policy = pickle.load(f1)

print(Policy[0])