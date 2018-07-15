import numpy as np
import my_tensorflow as mtf
from os import system

"""
Y = MX + B

M = 10
B = 5
X = 10

""" 
system('clear')
graph = mtf.Graph()
graph.set_as_default()

m = mtf.Variable(10)
b = mtf.Variable(-5)

x = mtf.Placeholder()

z = mtf.multiply(m,x)
y = mtf.add(z,b)


sess = mtf.Session()
ans = sess.run(y,feed_dict={x:10})

print(ans)

a = mtf.Variable(np.array([1,2,3]))
b = mtf.Variable(np.array([[1],[2],[3]]))

c = mtf.matmul(a,b)
ans = sess.run(c)
print(ans)