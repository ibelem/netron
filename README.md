# WebNN Netron

## Enhanced WebNN Netron Tool for Web Developers

1. WebNN API Support Status in Chromium
2. Enhanced WebNN Netron Tool for Web Developers. Provides the following features to streamline WebNN application development using Vanilla JavaScript
3. Weight and Bias Validation: Includes a built-in reader to verify data correctness in exported .bin files

This tool is based on [Netron](https://github.com/lutzroeder/netron). [Netron](https://github.com/lutzroeder/netron) is a viewer for neural network, deep learning and machine learning models developed by [Lutz Roeder](https://github.com/lutzroeder).

## Getting Started

- Visit [https://ibelem.github.io/netron/](https://ibelem.github.io/netron) page, load a model
- Visit [https://ibelem.github.io/netron/?url=model_link](https://ibelem.github.io/netron/?url=https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite)
- Visit [https://ibelem.github.io/netron/reader.html?json=json_url&bin=bin_url](https://ibelem.github.io/netron/reader.html?json=https://ibelem.github.io/netron/webnn/model_20250429081302.json&bin=https://ibelem.github.io/netron/webnn/model_20250429081302.bin) to view the model weights and biases data

<div align="center">
<img width="400px" height="100px" src="https://github.com/lutzroeder/netron/raw/main/.github/logo-light.svg#gh-light-mode-only">
<img width="400px" height="100px" src="https://github.com/lutzroeder/netron/raw/main/.github/logo-dark.svg#gh-dark-mode-only">
</div>

Netron is a viewer for neural network, deep learning and machine learning models.

Netron supports ONNX, TensorFlow Lite, Core ML, Keras, Caffe, Darknet, PyTorch, TensorFlow.js, Safetensors and NumPy.

Netron has experimental support for TorchScript, torch.export, ExecuTorch, TensorFlow, OpenVINO, RKNN, ncnn, MNN, PaddlePaddle, GGUF and scikit-learn.

<p align='center'><a href='https://www.lutzroeder.com/ai'><img src='.github/screenshot.png' width='800'></a></p>

## Install

**macOS**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.dmg` file or run `brew install --cask netron`

**Linux**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.AppImage` file or run `snap install netron`

**Windows**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.exe` installer or run `winget install -s winget netron`

**Browser**: [**Start**](https://netron.app) the browser version.

**Python**: `pip install netron`, then run `netron [FILE]` or `netron.start('[FILE]')`.

## Models

Sample model files to download or open using the browser version:

 * **ONNX**: [squeezenet](https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-3.onnx) [[open](https://netron.app?url=https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-3.onnx)]
 * **TorchScript**: [traced_online_pred_layer](https://github.com/ApolloAuto/apollo/raw/master/modules/prediction/data/traced_online_pred_layer.pt) [[open](https://netron.app?url=https://github.com/ApolloAuto/apollo/raw/master/modules/prediction/data/traced_online_pred_layer.pt)]
 * **TensorFlow Lite**: [yamnet](https://huggingface.co/thelou1s/yamnet/resolve/main/lite-model_yamnet_tflite_1.tflite) [[open](https://netron.app?url=https://huggingface.co/thelou1s/yamnet/blob/main/lite-model_yamnet_tflite_1.tflite)]
 * **TensorFlow**: [chessbot](https://github.com/srom/chessbot/raw/master/model/chessbot.pb) [[open](https://netron.app?url=https://github.com/srom/chessbot/raw/master/model/chessbot.pb)]
 * **Keras**: [mobilenet](https://github.com/aio-libs/aiohttp-demos/raw/master/demos/imagetagger/tests/data/mobilenet.h5) [[open](https://netron.app?url=https://github.com/aio-libs/aiohttp-demos/raw/master/demos/imagetagger/tests/data/mobilenet.h5)]
 * **Core ML**: [exermote](https://github.com/Lausbert/Exermote/raw/master/ExermoteInference/ExermoteCoreML/ExermoteCoreML/Model/Exermote.mlmodel) [[open](https://netron.app?url=https://github.com/Lausbert/Exermote/raw/master/ExermoteInference/ExermoteCoreML/ExermoteCoreML/Model/Exermote.mlmodel)]
 * **Darknet**: [yolo](https://github.com/AlexeyAB/darknet/raw/master/cfg/yolo.cfg) [[open](https://netron.app?url=https://github.com/AlexeyAB/darknet/raw/master/cfg/yolo.cfg)]
