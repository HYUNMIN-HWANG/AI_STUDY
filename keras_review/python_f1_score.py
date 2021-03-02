from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

print(accuracy_score(labels, guesses))      # 0.3
print(recall_score(labels, guesses))        # 0.42857142857142855
print(precision_score(labels, guesses))     # 0.5
print(f1_score(labels, guesses))            # 0.4615384615384615 
