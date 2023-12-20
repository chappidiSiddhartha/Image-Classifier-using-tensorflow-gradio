This project demonstrates a simple image classification system using Gradio, TensorFlow, and the MNIST dataset. The goal is to allow users to draw digit on a sketchpad, and the system will predict the digit using a pre-trained neural network.



Model Architecture
The image classification model is a neural network with multiple layers:

Flatten layer
Dense layer (512 units, ReLU activation)
Dropout layer (20% dropout)
Dense layer (256 units, ReLU activation)
Dropout layer (20% dropout)
Dense layer (128 units, ReLU activation)
Dropout layer (20% dropout)
Dense layer (10 units, no activation specified)


The model is trained on the MNIST dataset for 8 epochs using the Adam optimizer and Sparse Categorical Crossentropy loss.

The model has achieved an accuracy of 98.47%
