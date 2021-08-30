import tensorflow as tf
print(tf.__version__)   # 2.3.1

print(len(tf.config.list_physical_devices('GPU')))  # 2

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

'''
StreamExecutor device (1): NVIDIA GeForce GTX 1080 Ti, Compute Capability 6.1
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 11439809592699129853
, name: "/device:XLA_CPU:0"
device_type: "XLA_CPU"
memory_limit: 17179869184
locality {
}
incarnation: 1970089409332256833
physical_device_desc: "device: XLA_CPU device"
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 9100193139
locality {
  bus_id: 1
  links {
  }
}
incarnation: 11907811264421897223
physical_device_desc: "device: 0, name: NVIDIA GeForce GTX 1080 Ti, pci bus id:
0000:01:00.0, compute capability: 6.1"
, name: "/device:GPU:1"
device_type: "GPU"
memory_limit: 9100193139
locality {
  bus_id: 1
  links {
  }
}
incarnation: 1827177917615850502
physical_device_desc: "device: 1, name: NVIDIA GeForce GTX 1080 Ti, pci bus id:
0000:03:00.0, compute capability: 6.1"
, name: "/device:XLA_GPU:0"
device_type: "XLA_GPU"
memory_limit: 17179869184
locality {
}
incarnation: 1541530280801690521
physical_device_desc: "device: XLA_GPU device"
, name: "/device:XLA_GPU:1"
device_type: "XLA_GPU"
memory_limit: 17179869184
locality {
}
incarnation: 17922899426689132715
physical_device_desc: "device: XLA_GPU device"
]
'''