class Node:
    def __init__(self, history, parent=None):
        """
        Initialization
        :param B: The history.
        :param parent: The parent node.py of the current node.py.
        """
        self.visits = 1
        self.V = 0.0  # we can reshape the reward function
        self.history = history# which includes the states and actions
        self.children = []
        self.parent = parent

    def add_child(self, belief):
        """
        Create a child node.py and add it to the child list
        :param belief: the child node.py of the current node.py.
        :return: NULL
        """
        child = Node(belief, self)
        self.children.append(child)

    def update(self, reward):
        """
        Update the reward and add 1 to the visitation.
        :param reward: the new immediate reward
        :return: NULL
        """
        self.V += reward
        self.visits += 1  # if we visit that node, the visit is plus 1

    def fully_expanded(self):
        """
        Determine if the current node.py is fully expanded
        :return: NULL
        """
        if len(self.children) == self.history.num_moves:
            return True
        return False

    def __eq__(self, other):
        if self.history == other.history:
            return True
        return False

    def __repr__(self):
        s = "Node; children: %d; visits: %d; V: %f" % (len(self.children), self.visits, self.V)
        return s
