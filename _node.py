# node.py (Modificado)

"""
Defines the Node class, representing a component within a RuleTree.
Now includes feature type information for leaf nodes.
"""

# Usa ALL_COMPARISON_OPERATORS para checagem inicial genérica em is_leaf
from constants import ALL_COMPARISON_OPERATORS, LOGICAL_OPERATORS

class Node:
    """
    Represents a node in the rule tree. Includes feature type for leaves.
    """
    def __init__(self, attribute=None, operator=None, value=None, left=None, right=None, feature_type=None): # <<< Added feature_type
        """
        Initializes a Node.

        Args:
            attribute (str, optional): Attribute name for leaf nodes. Defaults to None.
            operator (str, optional): Comparison or logical operator. Defaults to None.
            value (any, optional): Threshold/category value for leaf nodes. Defaults to None.
            left (Node, optional): Left child node. Defaults to None.
            right (Node, optional): Right child node. Defaults to None.
            feature_type (str, optional): 'numeric' or 'categorical' for leaf nodes. Defaults to None.
        """
        self.attribute = attribute
        self.operator = operator
        self.value = value
        self.left = left
        self.right = right
        self.feature_type = feature_type # <<< Stores the type ('numeric' or 'categorical')

    def is_leaf(self):
        """Checks if the node is a leaf node (represents a condition)."""
        # A leaf has an attribute and a comparison operator. Feature type should also be set.
        return (self.attribute is not None and
                self.operator in ALL_COMPARISON_OPERATORS and # Check against all possible comparisons
                self.left is None and self.right is None)
                # self.feature_type is not None) # Could add this check for stricter validation

    def is_internal(self):
        """Checks if the node is an internal node (logical operator)."""
        # An internal node has a logical operator and children
        return (self.operator in LOGICAL_OPERATORS and
                self.left is not None and self.right is not None)

    def __str__(self):
        """Provides a basic string representation for debugging."""
        if self.is_leaf():
            # Optionally include type in debug string: f"({self.attribute}({self.feature_type}) {self.operator} {self.value})"
            return f"({self.attribute} {self.operator} {self.value})"
        elif self.is_internal():
            return f"({self.operator})"
        else:
            # Could be an incomplete node during construction/modification
             return f"(Incomplete/Invalid Node: Op={self.operator}, Attr={self.attribute})"