
from scripts.make_dataset import create_image_and_labels, create_train_and_test_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

images, labels = create_image_and_labels(['./data/processed/training10_0/training10_0.tfrecords'])
x_train, x_test, y_train, y_test = create_train_and_test_data(images, labels)

x_train_reshaped = x_train.reshape((x_train.shape[0], -1))
print(x_train_reshaped.shape)
x_test_reshaped = x_test.reshape((x_test.shape[0], -1))
print(x_test_reshaped.shape)

rf_model = RandomForestClassifier()
rf_model.fit(x_train_reshaped, y_train)

test_preds = rf_model.predict(x_test_reshaped)
print('traditional ML accuracy is: ', accuracy_score(test_preds,y_test))
print(classification_report(test_preds,y_test))
