"""
Parameter filtering module.

This module provides functionality to filter parameter combinations before computation,
allowing users to reduce the number of combinations by applying constraints.

Supported filter operations:
- Comparison operators: <, <=, >, >=, ==, !=
- Logical operators: and, or, not
- Parameter references: Use parameter names directly

Examples:
    # Simple comparison
    filter_expr = "P_aux < 1e6"
    
    # Multiple conditions
    filter_expr = "P_aux_DT_eq < P_aux and T_i > 10"
    
    # Complex expression
    filter_expr = "(V_plasma > 100 and n_tot < 1e20) or (T_i > 15 and T_i < 25)"
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
import ast
import operator


class FilterError(Exception):
    """Exception raised when filter expression is invalid."""
    pass


class ParameterFilter:
    """
    Filter for parameter combinations based on user-defined expressions.
    
    Attributes:
        expression: String expression defining the filter
        param_names: List of parameter names available for filtering
    """
    
    # Supported operators
    OPERATORS = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Not: operator.not_,
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    
    def __init__(self, expression: str, param_names: List[str]):
        """
        Initialize filter with expression and available parameters.
        
        Args:
            expression: Filter expression string (e.g., "P_aux < 1e6")
            param_names: List of available parameter names
            
        Raises:
            FilterError: If expression is invalid or uses unknown parameters
        """
        self.expression = expression
        self.param_names = param_names
        
        # Parse and validate expression
        try:
            self.ast_tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise FilterError(f"Invalid filter expression syntax: {e}")
        
        # Validate that all names in expression are valid parameters
        self._validate_expression()
    
    def _validate_expression(self) -> None:
        """Validate that expression only uses known parameter names."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Name):
                if node.id not in self.param_names:
                    raise FilterError(
                        f"Unknown parameter '{node.id}' in filter expression.\n"
                        f"Available parameters: {', '.join(self.param_names)}"
                    )
    
    def _eval_node(self, node: ast.AST, param_values: Dict[str, float]) -> Any:
        """
        Recursively evaluate AST node with parameter values.
        
        Args:
            node: AST node to evaluate
            param_values: Dictionary mapping parameter names to values
            
        Returns:
            Evaluated value (float or bool)
            
        Raises:
            FilterError: If node type is not supported
        """
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, param_values)
        
        elif isinstance(node, ast.Constant):
            return node.value
        
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        
        elif isinstance(node, ast.Name):
            return param_values[node.id]
        
        elif isinstance(node, ast.UnaryOp):
            op_func = self.OPERATORS.get(type(node.op))
            if op_func is None:
                raise FilterError(f"Unsupported unary operator: {type(node.op).__name__}")
            operand = self._eval_node(node.operand, param_values)
            return op_func(operand)
        
        elif isinstance(node, ast.BinOp):
            op_func = self.OPERATORS.get(type(node.op))
            if op_func is None:
                raise FilterError(f"Unsupported binary operator: {type(node.op).__name__}")
            left = self._eval_node(node.left, param_values)
            right = self._eval_node(node.right, param_values)
            return op_func(left, right)
        
        elif isinstance(node, ast.Compare):
            # Handle comparison chains (e.g., 1 < x < 10)
            left = self._eval_node(node.left, param_values)
            result = True
            
            for op, comparator in zip(node.ops, node.comparators):
                op_func = self.OPERATORS.get(type(op))
                if op_func is None:
                    raise FilterError(f"Unsupported comparison operator: {type(op).__name__}")
                
                right = self._eval_node(comparator, param_values)
                result = result and op_func(left, right)
                left = right
            
            return result
        
        elif isinstance(node, ast.BoolOp):
            op_func = self.OPERATORS.get(type(node.op))
            if op_func is None:
                raise FilterError(f"Unsupported boolean operator: {type(node.op).__name__}")
            
            # Evaluate all values
            values = [self._eval_node(val, param_values) for val in node.values]
            
            # Apply operator
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
            else:
                raise FilterError(f"Unsupported boolean operator: {type(node.op).__name__}")
        
        else:
            raise FilterError(f"Unsupported node type in expression: {type(node).__name__}")
    
    def evaluate(self, param_values: Dict[str, float]) -> bool:
        """
        Evaluate filter expression for given parameter values.
        
        Args:
            param_values: Dictionary mapping parameter names to values
            
        Returns:
            True if combination passes filter, False otherwise
        """
        try:
            result = self._eval_node(self.ast_tree, param_values)
            return bool(result)
        except Exception as e:
            raise FilterError(f"Error evaluating filter expression: {e}")


def apply_filter_to_combinations(
    input_data: Dict[str, np.ndarray],
    filter_expr: Optional[str],
    verbose: bool = True
) -> tuple[Dict[str, np.ndarray], np.ndarray, int]:
    """
    Apply filter to parameter combinations and return valid indices.
    
    This function evaluates the filter expression for all parameter combinations
    and returns only the combinations that pass the filter, along with their
    indices in the full parameter space.
    
    Args:
        input_data: Dictionary of parameter arrays
        filter_expr: Filter expression string (None = no filtering)
        verbose: Whether to print filtering statistics
        
    Returns:
        Tuple of (filtered_input_data, valid_indices, original_n_combinations)
        - filtered_input_data: Dict with flattened arrays of valid combinations
        - valid_indices: Array of indices in original parameter space
        - original_n_combinations: Total combinations before filtering
        
    Example:
        >>> input_data = {'P_aux': np.array([1e5, 1e6, 1e7]), 'T_i': np.array([10, 20])}
        >>> filtered, indices, total = apply_filter_to_combinations(
        ...     input_data, "P_aux < 1e6 and T_i > 15"
        ... )
        >>> # Returns only combinations where P_aux < 1e6 AND T_i > 15
    """
    if filter_expr is None or filter_expr.strip() == "":
        # No filtering - return all combinations
        param_shapes = [arr.shape[0] for arr in input_data.values()]
        n_combinations = int(np.prod(param_shapes))
        return input_data, np.arange(n_combinations), n_combinations
    
    # Create filter
    param_names = list(input_data.keys())
    try:
        param_filter = ParameterFilter(filter_expr, param_names)
    except FilterError as e:
        raise ValueError(f"Invalid filter expression: {e}")
    
    # Get parameter shapes and total combinations
    param_shapes = [arr.shape[0] for arr in input_data.values()]
    n_combinations = int(np.prod(param_shapes))
    
    if verbose:
        print(f"\n{'='*60}")
        print("APPLYING PARAMETER FILTER")
        print(f"{'='*60}")
        print(f"Filter expression: {filter_expr}")
        print(f"Total combinations before filtering: {n_combinations:,}")
    
    # Create meshgrid for all parameter combinations
    param_arrays = [np.asarray(arr) for arr in input_data.values()]
    meshgrid = np.meshgrid(*param_arrays, indexing='ij')
    
    # Flatten meshgrid arrays
    flattened_arrays = [grid.ravel() for grid in meshgrid]
    
    # Evaluate filter for each combination
    valid_mask = np.ones(n_combinations, dtype=bool)
    
    for i in range(n_combinations):
        # Get parameter values for this combination
        param_values = {name: flattened_arrays[j][i] 
                       for j, name in enumerate(param_names)}
        
        # Evaluate filter
        try:
            valid_mask[i] = param_filter.evaluate(param_values)
        except Exception as e:
            if verbose:
                print(f"Warning: Error evaluating filter for combination {i}: {e}")
            valid_mask[i] = False
    
    # Get valid indices
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    # Create filtered input data (flattened arrays with only valid combinations)
    filtered_input_data = {}
    for j, name in enumerate(param_names):
        filtered_input_data[name] = flattened_arrays[j][valid_mask]
    
    if verbose:
        print(f"Valid combinations after filtering: {n_valid:,}")
        print(f"Filter efficiency: {n_valid/n_combinations*100:.2f}%")
        print(f"Combinations excluded: {n_combinations - n_valid:,}")
        print(f"{'='*60}\n")
    
    return filtered_input_data, valid_indices, n_combinations


