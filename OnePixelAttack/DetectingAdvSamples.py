import pickle

with open("train.pkl","rb") as file:
    new_x_train = pickle.load(file)
    new_y_train = pickle.load(file)

with open("test.pkl", "wb") as file:
    new_x_test = pickle.dump(file)
    new_y_test = pickle.dump(file)
