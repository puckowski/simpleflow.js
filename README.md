# simpleflow.js

JavaScript library for training and deploying ML models

Regression (unbounded):

```javascript
const nn = new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 1])
  .withActivation('relu')
  .withOutputActivation() // linear by default
  .build();
Regression (bounded 0–1):
```

```javascript
const nn = new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 1])
  .withActivation('relu')
  .withOutputActivation('sigmoid')
  .build();
Classification (2-class):
```

```javascript
const nn = new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 1])
  .withActivation('relu')
  .withOutputActivation('sigmoid')
  .build();
Classification (multi-class, 3 classes):
```

```javascript
const nn = new FlexibleNNBuilder()
  .withLayerSizes([3, 10, 3])
  .withActivation('relu')
  .withOutputActivation((arr) => softmax(arr), null) // softmax over whole output
  .build();
```

Note: For softmax, you’ll need to handle it as a vector function, not per-neuron!

# Loading Pre-trained Models

```javascript
// Load foo.bin into IndexedDB under the key "foo"
await loadBinModelToIndexedDB('/models/foo.bin', 'foo');
// Now you can use FlexibleNN.loadModel('foo') as usual
const model = await FlexibleNN.loadModel('foo');
```
