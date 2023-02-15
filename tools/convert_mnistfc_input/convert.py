import onnx
import onnx.helper as h

m1 = onnx.load("mnist_3x50.onnx")
m2 = onnx.load("mnist-net_256x6.onnx")

onnx.checker.check_model(m1)
onnx.checker.check_model(m2)

g1 = m1.graph
g2 = m2.graph

n1 = g1.node
n2 = g2.node

"""
print(type(n1))

print("1.1---------")
print(n1[0])
print("1.2---------")
print(n1[1])
print("1.3---------")
print(n1[2])
print("---------")


print("2.1---------")
print(n2[0])
print("2.2---------")
print(n2[1])
print("2.3---------")
print(n2[2])
print("---------")
"""

# n2.remove(n2[0])
# n2.insert(0, n1[2])
# n2.insert(0, n1[1])
# n2.insert(0, n1[0])

# n2[0]
# print(n2[0])
# print(n2[1])
# print(n2[2])
# exit()

# n2[2].output[0] = "9"
# n2[3].input[0] = "9"

g = h.make_graph(
    n2, "torch-jit-export", m1.graph.input, m2.graph.output, m2.graph.initializer
)
m3 = h.make_model(g)
onnx.checker.check_model(m3)

onnx.save(m3, "modified.onnx")

m3 = onnx.load("modified.onnx")
onnx.checker.check_model(m3)

print(m3.graph.input)
for n in m3.graph.node[:5]:
    print("--------------")
    print(n)
