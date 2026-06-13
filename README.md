# simpleflow.js

simpleflow.js is a lightweight browser-first neural network utility for training, prediction, and IndexedDB model persistence.

## WebGPU Build

Load [simpleflow.webgpu.js](simpleflow.webgpu.js) when you want the same model format and builder API with optional WebGPU acceleration.

```html
<script src="./simpleflow.webgpu.js"></script>
```

The WebGPU build preserves the synchronous CPU methods:

- `build()`
- `predict(x)`
- `train(X, Y, epochs)`

It also adds async GPU-oriented helpers:

- `await model.initializeGPU()`
- `await model.predictGPU(x)`
- `await model.trainGPU(X, Y, epochs)`
- `model.getGPUStatus()`
- `await new FlexibleNNBuilder().withLayerSizes([...]).buildGPU()`

If WebGPU is unavailable, the GPU helpers fall back to the CPU path and keep the same saved-model format used by [simpleflow.js](simpleflow.js).

## Quick Start

Include [simpleflow.js](simpleflow.js) in a page, then create a model with `FlexibleNNBuilder`.

### Regression (unbounded output)

```javascript
const nn = new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 1])
  .withLearningRate(0.01)
  .withActivation('relu')
  .build(); // output is linear by default
```

### Regression (bounded 0..1)

```javascript
const nn = new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 1])
  .withLearningRate(0.01)
  .withActivation('relu')
  .withOutputActivation('sigmoid')
  .build();
```

### Binary classification

```javascript
const nn = new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 1])
  .withLearningRate(0.01)
  .withActivation('relu')
  .withOutputActivation('sigmoid')
  .build();

const probability = nn.predict([0.3, -1.2, 0.7])[0];
const predictedLabel = probability >= 0.5 ? 1 : 0;
```

### Multi-class classification (argmax over linear logits)

```javascript
const nn = new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 3])
  .withLearningRate(0.01)
  .withActivation('relu')
  .build(); // linear output layer

const logits = nn.predict([0.3, -1.2, 0.7]);
const predictedClass = logits.indexOf(Math.max(...logits));
```

> Note: output activation is currently applied element-wise. Full vector softmax is not built into the forward pass.

## Training

```javascript
const X = [
  [0, 0, 0],
  [0, 1, 0],
  [1, 0, 1],
  [1, 1, 1],
];

const Y = [
  [0],
  [0],
  [1],
  [1],
];

nn.train(X, Y, 200);
```

## Save and Load Models

### Save a trained model to IndexedDB

```javascript
await nn.saveModel('my-model');
```

### Load a model from IndexedDB

```javascript
const loaded = await FlexibleNN.loadModel('my-model');
const out = loaded.predict([0.1, 0.2, 0.3]);
```

### Import a model file from URL into IndexedDB

```javascript
await FlexibleNN.loadBinModelToIndexedDB('/models/foo.bin', 'foo');
const model = await FlexibleNN.loadModel('foo');
```

### Export a model from IndexedDB as .bin

```javascript
await FlexibleNN.exportModelToBinFile({
  key: 'foo',
  quantized: true,
  bits: 8,
});
```

## WebGPU Example

```javascript
const nn = await new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 1])
  .withLearningRate(0.01)
  .withActivation('relu')
  .withOutputActivation('sigmoid')
  .buildGPU();

const out = await nn.predictGPU([0.1, 0.2, 0.3]);
console.log(nn.getGPUStatus(), out);
```

Current note: the WebGPU path accelerates forward execution. Parameter updates still reuse the same JavaScript backprop logic so model behavior stays aligned with the original build.

## Full API Specification

See [SIMPLEFLOW_SPEC.md](SIMPLEFLOW_SPEC.md) for a complete API and data format reference.
