# simpleflow.js Specification

This document describes the current runtime behavior and public API exposed by [simpleflow.js](simpleflow.js).

The repository also includes [simpleflow.webgpu.js](simpleflow.webgpu.js), which keeps the same saved-model format and core builder surface while adding optional async WebGPU execution helpers.

## 1. Global Exports

When loaded in a browser page, [simpleflow.js](simpleflow.js) exposes:

- `window.FlexibleNN`
- `window.FlexibleNNBuilder`

When loaded instead, [simpleflow.webgpu.js](simpleflow.webgpu.js) exposes the same globals plus aliases:

- `window.FlexibleNNWebGPU`
- `window.FlexibleNNWebGPUBuilder`

## 2. Model Shape and Layer Rules

- A network is fully connected (dense) and feed-forward.
- `layerSizes` is required and must be an array with length >= 2.
- Example: `[3, 10, 1]`
  - input dimension = 3
  - one hidden layer of size 10
  - output dimension = 1

## 3. Activation Functions

### Hidden-layer activation names

Supported names for `withActivation(name, cap?)`:

- `relu`
- `leakyRelu`
- `clippedRelu` (uses `cap`, default `1.0`)
- `sigmoid`
- `tanh`

### Output activation

Configured with `withOutputActivation(name, cap?)`.

- If omitted, output is linear (`f(x) = x`).
- Supports same named activations as hidden layer.
- `clippedRelu` uses output cap (default `1.0`).
- Output activation is applied element-wise to each output neuron.

> Important: vector softmax is not built into the output layer execution path.

## 4. Builder API

### `new FlexibleNNBuilder()`
Creates a chainable builder.

### `withLayerSizes(sizes: number[])`
Sets required network layer dimensions.

### `withLearningRate(lr: number)`
Sets SGD step size. Default if omitted in final model: `0.01`.

### `withClipValue(clipVal: number)`
Sets clipping limit for:

- pre-activation values (`z`)
- parameter updates
- final weight/bias values

Default: `Number.MAX_SAFE_INTEGER / 2`.

### `withActivation(name: string, cap = 1.0)`
Sets hidden layer activation by name.

### `withCustomActivation(fn, dFn)`
Stores custom hidden activation in builder config.

Current note: constructor primarily resolves hidden activation via named lookup from `activationName`.

### `withOutputActivation(name: string, cap = 1.0)`
Sets output activation by name.

### `withCustomOutputActivation(fn, dFn)`
Sets custom output activation function and derivative.

### `build(): FlexibleNN`
Builds the network.

Throws if `layerSizes` is missing.

### `buildGPU(): Promise<FlexibleNN>`
Available in [simpleflow.webgpu.js](simpleflow.webgpu.js).

- Builds the network.
- Attempts GPU initialization before resolving.
- Still resolves with a usable model if WebGPU is unavailable.

## 5. FlexibleNN Instance API

### `forward(x: number[]): number[]`
Runs inference and returns output vector.

Also caches:

- `this.zs` (pre-activations by layer)
- `this.as` (activations by layer, including input)

### `backward(y: number[]): void`
Performs one backprop update using MSE gradient style:

- output delta initialized as `2 * (a - y)`
- hidden deltas use hidden derivative
- parameter update uses configured learning rate and clipping

### `train(X: number[][], Y: number[][], epochs = 100): void`
Trains sample-by-sample (online SGD) over `epochs`.

- Loss printed each epoch as mean MSE across samples.
- Optional fourth argument: `{ logEvery?: number }`
- `logEvery: 0` disables epoch logging.

### `predict(x: number[]): number[]`
Alias for `forward(x)`.

### `initializeGPU(): Promise<boolean>`
Available in [simpleflow.webgpu.js](simpleflow.webgpu.js).

- Returns `true` when a WebGPU device and per-layer pipelines are ready.
- Returns `false` when WebGPU is unavailable or when the model uses custom activations that cannot be compiled to WGSL.

### `forwardGPU(x: number[]): Promise<number[]>`
Available in [simpleflow.webgpu.js](simpleflow.webgpu.js).

- Runs the forward pass on WebGPU when initialized.
- Falls back to CPU `forward()` when WebGPU is unavailable.

### `predictGPU(x: number[]): Promise<number[]>`
Async alias for `forwardGPU(x)`.

### `trainGPU(X: number[][], Y: number[][], epochs = 100): Promise<void>`
Available in [simpleflow.webgpu.js](simpleflow.webgpu.js).

- Uses the GPU forward path when available.
- Reuses the same JavaScript backprop/update logic to preserve training behavior.
- Default execution mode is `auto`, which prefers CPU forward passes during training to avoid GPU-to-CPU synchronization on every sample.
- Optional override: `trainGPU(X, Y, epochs, { execution: 'cpu' | 'gpu' | 'auto' })`.
- Optional logging control: `trainGPU(X, Y, epochs, { logEvery?: number })`.
- `logEvery: 0` disables epoch logging.

### `withTrainingExecution(mode: 'auto' | 'cpu' | 'gpu')`
Available in [simpleflow.webgpu.js](simpleflow.webgpu.js).

- Sets the default training execution mode used by `trainGPU()`.

### `getGPUStatus(): { supported: boolean, initialized: boolean, reason: string | null }`
Available in [simpleflow.webgpu.js](simpleflow.webgpu.js).

- Reports whether WebGPU exists in the environment.
- Reports whether the model finished GPU initialization.
- Includes the latest fallback reason when GPU execution is unavailable.

### `saveModel(key: string): Promise<boolean>`
Stores model metadata + parameters in IndexedDB database `SimpleNN_DB`, store `models`.

## 6. FlexibleNN Static API

### `FlexibleNN.loadModel(key: string): Promise<FlexibleNN>`
Loads a saved model from IndexedDB and reconstructs a new network with saved config + params.

### `FlexibleNN.loadBinModelToIndexedDB(url, key, options?)`
Fetches model file from URL and stores in IndexedDB.

Options:

- `quantized` (default `false`)
- `bits` (default `8`)

Behavior:

- If `quantized === false` or `bits === 32`: treats file as JSON content.
- If quantized and `bits === 8`: parses custom binary format and dequantizes to float arrays.

### `FlexibleNN.importModelFromFile(file, key)`
Reads a local file as text JSON and stores model by key.

### `FlexibleNN.exportModelToBinFile({ key, quantized = false, bits = 8 })`
Exports model from IndexedDB and triggers browser download.

- Non-quantized (or `bits === 32`): JSON blob with `.bin` extension.
- Quantized `8-bit`: custom packed binary with metadata header + Int8 arrays.

### `FlexibleNN.import8bitBinModelToIndexedDB(file, key)`
Imports local 8-bit quantized binary file, dequantizes, and stores model.

## 7. IndexedDB Details

- DB name: `SimpleNN_DB`
- Object store: `models`
- Key: user-supplied model key string

## 8. Saved Model Object Shape

A typical stored model includes:

- `layerSizes: number[]`
- `learningRate: number`
- `weights: number[][][]`
- `biases: number[][]`
- `activationName: string`
- `activationCap: number`
- `outputActivationName: string | null`
- `outputActivationCap: number`

## 9. Quantized 8-bit Binary Layout

Serialized file layout:

1. `uint32` little-endian metadata byte length
2. UTF-8 JSON metadata string
3. For each layer transition `l`:
   - Int8 flattened weight matrix bytes (`outSize * inSize`)
   - Int8 bias vector bytes (`outSize`)

Metadata includes:

- `layerSizes`
- activation settings
- `weightsMinMax` per layer
- `biasesMinMax` per layer

Dequantization maps each Int8 value back into original min/max range.

## 10. Practical Usage Notes

- Use `sigmoid` output for binary classification.
- For multi-class classification, output logits and compute `argmax` externally.
- Training is plain SGD without minibatches, momentum, or regularization.
- Console logging occurs every epoch in `train()`.
