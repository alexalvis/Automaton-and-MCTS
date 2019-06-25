import pickle

filename = "../pre_compute/filename_P_s1_a_s2.pkl"
with open(filename, "rb") as f1:
    P = pickle.load(f1)

print(P[((9,9),(9,8))])