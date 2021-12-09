Face Mask appliance is important in the time of COVID-19.

Transfer learning.
OpenCV, PyTorch, Matplotlib.

Use transfer learning approach to build a deep learning model on top of an existing powerful image model RestNet18
then load and preprocess dataset images with OpenCV and feed them into the model to train the model. The model is built and trained with powerful PyTorch, with final accuracy up to 98.9%.
Use OpenCV to capture video frames and predict whether the human in the video is face masked or not.

1. preprocess the dataset cropping face using OpenCV's facial detection feature.
2. build transfer learning model based of RestNet18 and train it.
3. Capture video with OpenCV and crop human face in the video frame, and predict whether masked or not.
