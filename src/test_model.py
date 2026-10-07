import tensorflow as tf

from model import build_songnet


INPUT_SHAPE = (1293, 128)


model = build_songnet(INPUT_SHAPE)

model.summary()

dummy_input = tf.random.normal(
    shape=(1, INPUT_SHAPE[0], INPUT_SHAPE[1])
)

output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)