from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt
breast_cancer_data = load_breast_cancer()

print(breast_cancer_data.data[0])

print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)

print(breast_cancer_data.target_names)
training_data, validation_data, training_labels,validation_labels = train_test_split(breast_cancer_data.data,breast_cancer_data.target,test_size=0.2,random_state=100)

#one label for every piece of data!
print(len(training_data) == len(training_labels))

print("Accuracy of sklearns' Classifier")
accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  acc = classifier.score(validation_data, validation_labels)
  accuracies.append(acc)
  print(acc)

k_list = range(1,101)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()


def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def min_max_normalize(lst):
  minimum = min(lst)
  maximum = max(lst)
  normalized = []
  for i in range(len(lst)):
    normalized.append((lst[i] - minimum)/(maximum - minimum))
  return normalized

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for point, label in zip(dataset,labels):
    distance_to_point = euclidean_distance(point, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, label])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_positive = 0
  num_negative = 0
  for neighbor,label in neighbors:
    if label == 1:
    	num_positive += 1
    else:
    	num_negative += 1
  return 1 if num_positive > num_negative else 0


def find_validation_accuracy(training_set,training_labels,validation_set,validation_labels,k):
  num_correct = 0.0
  for point in range(len(validation_set)):
    guess = classify(validation_set[point],training_set,training_labels,k)
    if guess == validation_labels[point]:
      num_correct+=1
  return num_correct/len(validation_set)

print("Accuracy of My Classifier")
acc_of_my_knn = []
for k in range(1,101):
	acc = find_validation_accuracy(training_data,training_labels,validation_data,validation_labels,k)
	acc_of_my_knn.append(acc)
	print(acc)

plt.plot(k_list, acc_of_my_knn)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()