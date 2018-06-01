
# from tensorflow.ops.rnn_cell_impl import _zero_state_tensors
def _zero_state_tensors(state_size, batch_size, dtype):
    """Create tensors of zeros based on state_size, batch_size, and dtype."""
    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = _concat(batch_size, s)
        size = array_ops.zeros(c, dtype=dtype)
        if not context.executing_eagerly():
            c_static = _concat(batch_size, s, static=True)
            size.set_shape(c_static)
        return size
    return nest.map_structure(get_state_shape, state_size)