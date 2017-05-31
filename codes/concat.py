from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops

def concat(concat_dim, values,  name="concat"):
    if not isinstance(values, (list, tuple)):
        values = [values]
    if len(values) == 1:  # Degenerate case of one tensor.
        with ops.name_scope(name) as scope:
            ops.convert_to_tensor(concat_dim,
                                  name="concat_dim",
                                  dtype=dtypes.float32).get_shape(
                                  ).assert_is_compatible_with(tensor_shape.scalar())
            return identity(values[0], name=scope)
    return gen_array_ops._concat(concat_dim=concat_dim,
                                 values=values,
                                 name=name)
