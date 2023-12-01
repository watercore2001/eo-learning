import torch
from einops import rearrange

def add_tensor(x: torch.Tensor | None, y: torch.Tensor | None):
    if x is None and y is None:
        return None
    if x is None:
        return y
    if y is None:
        return x
    return x + y


def window_partition(x: torch.Tensor, window_size: int):
    # the order of (window_num_in_h window_size_in_h) is important: window_num_in_h pieces of window
    return rearrange(x,
                     pattern="b c (window_num_in_h window_size_in_h) (window_num_in_w window_size_in_w) -> "
                     "(b window_num_in_h window_num_in_w) (window_size_in_h window_size_in_w) c",
                     window_size_in_h=window_size, window_size_in_w=window_size)


def window_reverse(x: torch.Tensor, window_num_in_h: int, window_num_in_w: int, window_size: int):
    return rearrange(x,
                     pattern="(b window_num_in_h window_num_in_w) (window_size_in_h window_size_in_w) c -> "
                     "b c (window_num_in_h window_size_in_h) (window_num_in_w window_size_in_w)",
                     window_num_in_h=window_num_in_h, window_num_in_w=window_num_in_w,
                     window_size_in_h=window_size, window_size_in_w=window_size)


class WindowShifter:
    def __init__(self, window_size: int, shift_size: int):
        self.window_size = window_size
        self.shift_size = shift_size

    def shift(self, x: torch.Tensor):
        """
        Args:
            x: b c h w
        Returns:
            shifted x: b c h w
            shift mask: (window_num, m*m, m*m) m is window size
        """
        _, _, h, w = x.size()
        # different roll order is equivalent
        x = x.roll(shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        img_mask = torch.zeros((h, w), device=x.device)

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        # split the input feature into 9 area. each are have different value
        mask_value = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[h_slice, w_slice] = mask_value
                mask_value += 1

        # img must have four dimension, so the mask can be data by self.window_partition
        mask_windows = window_partition(img_mask[None, None, :, :], window_size=self.window_size)

        mask_windows = rearrange(mask_windows, pattern="window_num mm 1 -> window_num mm")
        shift_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        shift_mask = shift_mask.masked_fill(shift_mask != 0, -float('inf')).masked_fill(shift_mask == 0, 0)
        # (window_num, m * m, m * m)
        return x, shift_mask

    def reverse(self, x: torch.Tensor):
        return x.roll(shifts=(self.shift_size, self.shift_size), dims=(2, 3))