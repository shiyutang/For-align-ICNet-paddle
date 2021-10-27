import paddle

def sum_count(bool_tensor):
    # print(bool_tensor.dtype)
    assert bool_tensor.dtype == paddle.bool
    return paddle.fluid.layers.where(bool_tensor).shape[0]

# x = paddle.to_tensor([1,2,3])
# y = (x>0)
# print(sum_count(y))