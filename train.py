import os
import numpy as np
from PIL import Image
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs("static/graphs", exist_ok=True)


# CONFIGURATION

IMG_SIZE = 128
BATCH_SIZE = 32

# SAMPLE IMAGE VISUALIZATION

cat_path = "dataset/animals/cat"
dog_path = "dataset/animals/dog"

cat_img = os.path.join(cat_path, os.listdir(cat_path)[0])
dog_img = os.path.join(dog_path, os.listdir(dog_path)[0])

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(Image.open(cat_img))
plt.title("Cat Sample Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(Image.open(dog_img))
plt.title("Dog Sample Image")
plt.axis("off")

plt.show()

cat_count = len(os.listdir(cat_path))
dog_count = len(os.listdir(dog_path))

print(f"Total Cat Images : {cat_count}")
print(f"Total Dog Images : {dog_count}")


# MODERN DATASET DISTRIBUTION

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(7,7))

fig.patch.set_facecolor("#07152f")
ax.set_facecolor("#07152f")

sizes = [cat_count, dog_count]

colors = ["#9D4EDD", "#3B82F6"]

wedges, texts = ax.pie(
    sizes,
    startangle=90,
    colors=colors,
    wedgeprops=dict(
        width=0.42,
        edgecolor="#07152f"
    )
)

# Center Text
ax.text(
    0,
    0.08,
    f"{cat_count+dog_count}",
    ha="center",
    color="white",
    fontsize=24,
    fontweight="bold"
)

ax.text(
    0,
    -0.12,
    "Total",
    ha="center",
    color="#cbd5e1",
    fontsize=14
)

plt.savefig(
    "static/graphs/dataset_distribution.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="#07152f"
)

plt.close()


# DATA PREPROCESSING

datagen = ImageDataGenerator(
rescale=1./255,
validation_split=0.2
)

train_data = datagen.flow_from_directory(
"dataset/animals",
target_size=(IMG_SIZE, IMG_SIZE),
batch_size=BATCH_SIZE,
class_mode='binary',
subset='training'
)

val_data = datagen.flow_from_directory(
"dataset/animals",
target_size=(IMG_SIZE, IMG_SIZE),
batch_size=BATCH_SIZE,
class_mode='binary',
subset='validation'
)

# CNN MODEL

model = Sequential()

model.add(
Conv2D(
32,
(3,3),
activation='relu',
input_shape=(128,128,3)
)
)

model.add(MaxPooling2D(2,2))

model.add(
Conv2D(
64,
(3,3),
activation='relu'
)
)

model.add(MaxPooling2D(2,2))

model.add(
Conv2D(
128,
(3,3),
activation='relu'
)
)

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(
128,
activation='relu'
))

model.add(
Dropout(0.5)
)

model.add(
Dense(
1,
activation='sigmoid'
)
)

model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy']
)

model.summary()

# TRAIN MODEL

history = model.fit(
train_data,
validation_data=val_data,
epochs=10
)

# SAVE MODEL

if not os.path.exists("model"):
    os.makedirs("model")

model.save("model/cnn_model.h5")

print("\nModel trained and saved successfully!")


# MODERN ACCURACY & LOSS GRAPH


plt.style.use("dark_background")

fig, ax = plt.subplots(1,2, figsize=(14,5))

bg = "#07152f"

fig.patch.set_facecolor(bg)

for a in ax:
    a.set_facecolor(bg)

# Accuracy

ax[0].plot(
    history.history["accuracy"],
    color="#3B82F6",
    linewidth=3,
    label="Training Accuracy"
)

ax[0].plot(
    history.history["val_accuracy"],
    color="#F59E0B",
    linewidth=3,
    label="Validation Accuracy"
)

ax[0].set_title(
    "Accuracy",
    color="white",
    fontsize=16,
    fontweight="bold"
)

ax[0].grid(
    color="white",
    alpha=0.08
)

ax[0].legend()

# Loss

ax[1].plot(
    history.history["loss"],
    color="#3B82F6",
    linewidth=3,
    label="Training Loss"
)

ax[1].plot(
    history.history["val_loss"],
    color="#F59E0B",
    linewidth=3,
    label="Validation Loss"
)

ax[1].set_title(
    "Loss",
    color="white",
    fontsize=16,
    fontweight="bold"
)

ax[1].grid(
    color="white",
    alpha=0.08
)

ax[1].legend()

plt.tight_layout()

plt.savefig(
    "static/graphs/accuracy_loss.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=bg
)

plt.close()

# FINAL ACCURACY PRINT

loss, accuracy = model.evaluate(val_data)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Validation Accuracy : {accuracy*100:.2f}%")
print(f"Validation Loss : {loss:.4f}")


# PREDICTIONS FOR EVALUATION

val_data.reset()

predictions = model.predict(val_data)

y_pred = (predictions > 0.5).astype(int)

y_true = val_data.classes

cm = confusion_matrix(
    y_true,
    y_pred
)

# MODERN CONFUSION MATRIX

plt.figure(figsize=(8,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=["Cat","Dog"],
    yticklabels=["Cat","Dog"]
)

plt.xlabel(
    "Predicted",
    fontsize=12,
    color="white"
)

plt.ylabel(
    "Actual",
    fontsize=12,
    color="white"
)

plt.xticks(color="white")
plt.yticks(color="white")

plt.savefig(
    "static/graphs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="#07152f"
)

plt.close()


# CLASSIFICATION REPORT

report = classification_report(
    y_true,
    y_pred,
    target_names=["Cat","Dog"]
)

print(report)

with open(
    "static/classification_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)