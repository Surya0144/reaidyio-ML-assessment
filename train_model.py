import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

print("Training model...")
iris = load_iris()
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(iris.data, iris.target)
joblib.dump(clf, "model.pkl")
print("Success: model.pkl created.")