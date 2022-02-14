# Hello-ML

A basic reimplementation of MNIST digits classification with a multi-layer-perceptron using pure Rust (no PyTorch/TensorFlow). Uses no libraries other than [ndarray](https://github.com/rust-ndarray/ndarray) and autograd.

## Usage

First, run `download_mnist.sh`. Then, compile:

```sh
cargo build --release
```

Finally, if you're planning to run for debug mode, then:

```sh
ML_DEBUG_TRUE=0 ./target/release/hello-ml
```

Note that debug mode is **up to 50x slower** than standard mode, so debug mode is not recommended during any time other than development.

If you want to run in release mode, then:

```sh
./target/release/hello-ml
```

## What this includes

- Train model
- Make predictions from model

## What this _does not_ include

- Save or load model (since autograd doesn't support typical Serde serialization)
- GPU-acceleration (this will be pure CPU)
- ONNX saving
- Any sort of graphical user interface
- Any sort of argument parsing or user-provided parameters