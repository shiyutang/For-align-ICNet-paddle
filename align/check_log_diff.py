import numpy as np
from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()

    # forward
    # f0d = diff_helper.load_info("./forward/forward0_paddle.npy")
    # f0t = diff_helper.load_info("./forward/forward0_torch.npy")
    # f1d = diff_helper.load_info("./forward/forward1_paddle.npy")
    # f1t = diff_helper.load_info("./forward/forward1_torch.npy")
    # f2d = diff_helper.load_info("./forward/forward2_paddle.npy")
    # f2t = diff_helper.load_info("./forward/forward2_torch.npy")
    # f3d = diff_helper.load_info("./forward/forward3_paddle.npy")
    # f3t = diff_helper.load_info("./forward/forward3_torch.npy")
    #
    # diff_helper.compare_info(f3d, f3t)
    # diff_helper.report(
    #     diff_method="mean", diff_threshold=1e-6, path="./forward_diff.txt")


    # backward
    l0d = diff_helper.load_info("./backward/lr_backward_0_paddle.npy")
    l0t = diff_helper.load_info("./backward/lr_backward_0_torch.npy")
    l1d = diff_helper.load_info("./backward/lr_backward_1_paddle.npy")
    l1t = diff_helper.load_info("./backward/lr_backward_1_torch.npy")
    l2d = diff_helper.load_info("./backward/lr_backward_2_paddle.npy")
    l2t = diff_helper.load_info("./backward/lr_backward_2_torch.npy")
    l3d = diff_helper.load_info("./backward/lr_backward_3_paddle.npy")
    l3t = diff_helper.load_info("./backward/lr_backward_3_torch.npy")
    l4d = diff_helper.load_info("./backward/lr_backward_4_paddle.npy")
    l4t = diff_helper.load_info("./backward/lr_backward_4_torch.npy")

    diff_helper.compare_info(l4d, l4t)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-5, path="./lr_backward_diff.txt")

    s0d = diff_helper.load_info("./backward/loss_backward_0_paddle.npy")
    s0t = diff_helper.load_info("./backward/loss_backward_0_torch.npy")
    s1d = diff_helper.load_info("./backward/loss_backward_1_paddle.npy")
    s1t = diff_helper.load_info("./backward/loss_backward_1_torch.npy")
    s2d = diff_helper.load_info("./backward/loss_backward_2_paddle.npy")
    s2t = diff_helper.load_info("./backward/loss_backward_2_torch.npy")
    s3d = diff_helper.load_info("./backward/loss_backward_3_paddle.npy")
    s3t = diff_helper.load_info("./backward/loss_backward_3_torch.npy")
    s4d = diff_helper.load_info("./backward/loss_backward_4_paddle.npy")
    s4t = diff_helper.load_info("./backward/loss_backward_4_torch.npy")

    diff_helper.compare_info(s4d, s4t)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-5, path="./loss_backward_diff.txt")
