# cnn-doa-app

Python application for real-time estimation of audio signal DOA (Direction of Arrival).
This code is a part of my Master of Engineering thesis titled "Real-time Deep Neural Networks for Microphone Array Direction of Arrival Estimation".

The application runs on Raspberry Pi with a MiniDSP UMA-8 microphone array connected to it. It is capable of estimating DOA for up to 2 sources active at the same time. This estimation is done using TensorFlow Lite models trained on synthetic signals simulated using [pyroomacoustics](https://github.com/LCAV/pyroomacoustics). Jupyter Notebooks used to simulate audio, preprocess it and train the models are available in my [MEng-project](https://github.com/thomson008/MEng-project) repository.

Screenshots from the application in single-source and multi-source mode respectively are attached below:
<p float="left">
  <img src="/single_app_active.png" width="400" />
  <img src="/multi_app_active.png" width="400" />
</p>
