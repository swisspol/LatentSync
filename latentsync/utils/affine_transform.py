# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch
from einops import rearrange
import kornia


class AlignRestore(object):
    def __init__(self, align_points=3, resolution=256, device="cpu", dtype=torch.float32):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = resolution / 256 * 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
            self.device = device
            self.dtype = dtype
            self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=dtype)
            self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype)

    def align_warp_face(self, img, lmks3, smooth=True):
        affine_matrix, self.p_bias = self.transformation_from_points(lmks3, self.face_template, smooth, self.p_bias)

        img_tensor = rearrange(
            torch.from_numpy(img).to(device=self.device, dtype=self.dtype), "h w c -> c h w"
        ).unsqueeze(0)
        affine_matrix_tensor = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        crop_tensor = kornia.geometry.transform.warp_affine(
            img_tensor,
            affine_matrix_tensor,
            (self.face_size[1], self.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )
        cropped_face = rearrange(crop_tensor.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
        return cropped_face, affine_matrix

    def restore_img(self, h, w, face, torch_affine_matrix):
        if isinstance(torch_affine_matrix, np.ndarray):
            torch_affine_matrix = (
                torch.from_numpy(torch_affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)
            )

        torch_inverse_affine = kornia.geometry.transform.invert_affine_transform(torch_affine_matrix)
        face_tensor = rearrange(torch.from_numpy(face).to(device=self.device, dtype=self.dtype), "h w c -> c h w")

        inv_restored = kornia.geometry.transform.warp_affine(
            face_tensor.unsqueeze(0),
            torch_inverse_affine,
            (h, w),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        ).squeeze(0)

        inv_mask = kornia.geometry.transform.warp_affine(
            self.mask, torch_inverse_affine, (h, w), padding_mode="zeros"
        )  # (1, 1, h_up, w_up)

        out_face = rearrange(inv_restored / 255, "c h w -> h w c")
        out_mask = inv_mask.squeeze(0).squeeze(0)  # (h_up, w_up)
        return (out_face, out_mask)

    def transformation_from_points(self, points1: torch.Tensor, points0: torch.Tensor, smooth=True, p_bias=None):
        if isinstance(points0, np.ndarray):
            points2 = torch.tensor(points0, device=self.device, dtype=torch.float32)
        else:
            points2 = points0.clone()

        if isinstance(points1, np.ndarray):
            points1_tensor = torch.tensor(points1, device=self.device, dtype=torch.float32)
        else:
            points1_tensor = points1.clone()

        c1 = torch.mean(points1_tensor, dim=0)
        c2 = torch.mean(points2, dim=0)

        points1_centered = points1_tensor - c1
        points2_centered = points2 - c2

        s1 = torch.std(points1_centered)
        s2 = torch.std(points2_centered)

        points1_normalized = points1_centered / s1
        points2_normalized = points2_centered / s2

        covariance = torch.matmul(points1_normalized.T, points2_normalized)
        U, S, V = torch.svd(covariance)

        R = torch.matmul(V, U.T)

        det = torch.det(R)
        if det < 0:
            V[:, -1] = -V[:, -1]
            R = torch.matmul(V, U.T)

        sR = (s2 / s1) * R
        T = c2.reshape(2, 1) - (s2 / s1) * torch.matmul(R, c1.reshape(2, 1))

        M = torch.cat((sR, T), dim=1)

        if smooth:
            bias = points2_normalized[2] - points1_normalized[2]
            if p_bias is None:
                p_bias = bias
            else:
                bias = p_bias * 0.2 + bias * 0.8
            p_bias = bias
            M[:, 2] = M[:, 2] + bias

        return M.cpu().numpy(), p_bias
