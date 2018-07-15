import numpy as np


class Operation(object):
    """Base Operation class from which other higher level operations will inherit"""

    def __init__(self, input_nodes=[]):
        """ Constructior.
        After object creation we append it to respective list of _default_graph.

        Args : 
            input_nodes - List of all input nodes 
        """
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.inputs = []
        self.output= None

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        """Placeholder method to be overwritten by the inherited class."""
        pass


class add(Operation):
    """Add operation """

    def __init__(self, x, y):
        """ Constructior.

        Args : 
            x, y - Actual data to perform add operation on
        """
        super(add, self).__init__([x, y])

    def compute(self, x_var, y_var):
        """ Actual addition is done here.

        Args : 
            x_var, y_var - Data to perform add operation on.

        Return:
            summation of args .
        """
        self.inputs = [x_var, y_var]
        return x_var+y_var


class multiply(Operation):
    """Multiply operation """

    def __init__(self, x, y):
        """ Constructior.

        Args : 
            x, y - Actuall data to perform multiply operation on
        """
        super(multiply, self).__init__([x, y])

    def compute(self, x_var, y_var):
        """ Actual multiply is done here.

        Args : 
            x_var, y_var - Data to perform multiply operation on.

        Return:
            Product of args .
        """
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):
    """Matrix - Multiplication operation """

    def __init__(self, x, y):
        """ Constructior.

        Args : 
            x, y - Actuall data to perform operation on
        """
        super(matmul, self).__init__([x, y])

    def compute(self, x_var, y_var):
        """ Actua operation is done here.

        Args : 
            x_var, y_var - Data to perform operation on.

        Return:
            dot product of args .
        """
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


class Variable():
    """Implementation of tensorflow's placeholder class"""

    def __init__(self, initial_value=None):
        """Constructor.
        After this object creation we append it to respective list of _default_graph."""

        self.value = initial_value
        self.output_nodes = []
        self.output = None
        # _default_graph - It is the default graph object connecting Placeholders and Variables to Operations.

        _default_graph.variables.append(self)


class Placeholder():
    """Implementation of tensorflow's placeholder class"""

    def __init__(self):
        """Constructor.
        After this object creation we append it to respective list of _default_graph."""

        self.output_nodes = []
        self.output = None
        # _default_graph - It is the default graph object connecting Placeholders and Variables to Operations.

        _default_graph.placeholders.append(self)


class Session():
    """ Implementation of tensorflow's Session class."""

    def traverse_postorder(self, operation):
        """PostOrder Traversal of nodes.
        This function makes sure that all operations are done in right order.

        Args:
            operation : the operation whose postorder form is required.

        Returns:
            List of operations in postorder form
        """
        nodes_postorder = []

        def recurse(node):
            if isinstance(node, type(operation)) or isinstance(node,Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)

        recurse(operation)

        return nodes_postorder

    def run(self, operation, feed_dict={}):
        """Running the session to produce output

        Args:
            operation - The operation to compute. 
            feed_dict - A dictionary to map values to placeholders. 

        Return:
            output of the operation.
        """
        nodes_postorder = self.traverse_postorder(operation)
        
        for node in nodes_postorder:

            if type(node) == Placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value

            else:  # Operation
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


class Graph():
    """The main thing connecting every operation placeholder and variable."""

    def __init__(self):
        """ Constructor."""

        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        """We call this method to set the current graph as the default graph.
        In this way we can have multiple computation graph """

        global _default_graph
        _default_graph = self
