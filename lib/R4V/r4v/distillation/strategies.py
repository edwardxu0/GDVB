from functools import partial


def drop_layer(model, layer_id, layer_type=None):
    if layer_type is not None:
        model.drop_layer(layer_id, layer_type=layer_type)
    else:
        model.drop_layer(layer_id)


def drop_operation(model, layer_id, op_type, layer_type=None):
    if layer_type is not None:
        model.drop_operation(layer_id, op_type, layer_type=layer_type)
    else:
        model.drop_operation(layer_id, op_type)


def forall(model, layer_type, strategy, excluding=None, **s_config):
    action = partial(getattr(model, strategy), **s_config)
    model.forall(layer_type, action, excluding=excluding)


def linearize(model, layer_id, layer_type=None):
    if layer_type is not None:
        model.linearize(layer_id, layer_type=layer_type)
    else:
        model.linearize(layer_id)


def scale_input(model, factor):
    model.scale_input(factor)


def scale_layer(model, layer_id, factor, layer_type=None):
    if layer_type is not None:
        model.scale_layer(layer_id, factor, layer_type=layer_type)
    else:
        model.scale_layer(layer_id, factor)


def scale_convolution_stride(model, layer_id, factor, layer_type=None):
    if layer_type is not None:
        model.scale_convolution_stride(layer_id, factor, layer_type=layer_type)
    else:
        model.scale_convolution_stride(layer_id, factor)


def replace_convolution_padding(model, layer_id, padding, layer_type=None):
    if layer_type is not None:
        model.replace_convolution_padding(layer_id, padding, layer_type=layer_type)
    else:
        model.replace_convolution_padding(layer_id, padding)
