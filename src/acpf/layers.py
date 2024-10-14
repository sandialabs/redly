#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#
#  Contains a modification of the tensorflow keras Dense layer:
#  https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/core/dense.py#L31-L264
#  Used in accorance with Apache License-2.0
#  ___________________________________________________________________________


from tensorflow.keras import activations
from tensorflow.keras import backend
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
import tensorflow.compat.v2 as tf


class MDense(Layer):
    '''
    Modification of tf.keras.layers.Dense to support masking
    https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/core/dense.py#L31-L264
    '''
    def __init__(self, 
                 units,
                 activation=None,
                 use_bias=True,
                 mask=None,
                 init_weights=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MDense, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Received an invalid value for `units`, expected '
                             f'a positive integer. Received: units={units}')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.mask = mask
        self.init_weights = init_weights
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with a floating-point '
                            f'dtype. Received: dtype={dtype}')

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to a Dense layer '
                             'should be defined. Found None. '
                             f'Full input shape received: {input_shape}')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        # modification: change initializer to use init_weights if specified
        if self.init_weights is not None:
            def my_init(shape, dtype=self.dtype):
                return tf.convert_to_tensor(self.init_weights, dtype)
            self.kernel_initializer = my_init

        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        
        # modification:  initialize mask
        if self.mask is not None:
            def my_mask_init(shape, dtype=self.dtype):
                return tf.convert_to_tensor(self.mask, dtype)
            self.mask_kernel = self.add_weight(
                'mask',
                shape=[last_dim, self.units],
                initializer=my_mask_init,
                dtype=self.dtype,
                trainable=False)
                
        self.built = True

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        # modification:  adjust kernel with mask if necessary
        if self.mask is not None:
            kernel = tf.math.multiply(self.kernel, self.mask_kernel)
        else:
            kernel = self.kernel
            
        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension (last
            # dimension not ragged), we can flatten the input and restore the ragged
            # dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError('Dense layer only supports RaggedTensors when the '
                                  'innermost dimension is non-ragged. Received: '
                                  f'inputs.shape={inputs.shape}.')
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:
                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                    inputs = inputs.to_tensor()
                    original_inputs = tf.RaggedTensor.from_nested_row_splits(
                        inputs, original_inputs.nested_row_splits[:-1])

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul operation for
            # large sparse input tensors. The op will result in a sparse gradient, as
            # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(kernel, ids, weights, combiner='sum')
            else:
                outputs = tf.matmul(a=inputs, b=kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the input shape of a Dense layer '
                             'should be defined. Found None. '
                             f'Received: input_shape={input_shape}')
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return config