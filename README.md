# My Tensorflow

This is a small project which I started with the intention of understanding how Google's [Tensorflow](https://github.com/tensorflow/tensorflow) works.This project is written in Python3.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites

You need to have the following packages pre-installed.

```
pip install numpy
```

### Installing

How to get started with My Tensorflow.

It is fairly simple you just need to clone the project.

```
git clone https://github.com/Jeetu95/My-Tensorflow.git
```

Include it in your project by adding this line in imports.

```
import my_tensorflow as mtf
```

Run the **example.py** to see if it is working or not.

## Using My Tesorflow
Example Code.

```
import my_tensorflow as mtf
a = mtf.Variable(10)
b = mtf.Variable(20)
c = mtf.add(a,b)

sess = mtf.Session()
print(sess.run(c))
```

Output
```
30
```
## Built With

* [Python](https://www.python.org/) - The language used for writting this project.
* [Numpy](http://www.numpy.org/) - The package used for fast computing of array object.

## Authors

* **Subhajit Das** - *Initial work* - [LinkedIn](https://www.linkedin.com/in/subhajit-das-764742142/)   [Twitter](https://twitter.com/Subhajit_Das_95) 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

