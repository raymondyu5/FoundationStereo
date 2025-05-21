#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# TensorRT demo for FoundationStereo (.engine built in FP16, TRT 10.3)
# ---------------------------------------------------------------------------

import os, sys, argparse, logging, cv2, imageio.v2 as imageio, numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda; import pycuda.autoinit  # noqa: F401
import open3d as o3d
import time

# repo-local utils
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(code_dir, ".."))
from Utils import vis_disparity, depth2xyzmap, toOpen3dCloud   # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# -------------------------------------------------------------------- helpers
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        return rt.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    """Return addr dict {tensor_name: device_ptr}, plus host buffers & stream."""
    addr, host_buffers = {}, {}
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        name   = engine.get_tensor_name(i)
        shape  = engine.get_tensor_shape(name)
        dtype  = trt.nptype(engine.get_tensor_dtype(name))
        host   = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
        dev    = cuda.mem_alloc(host.nbytes)
        addr[name] = int(dev)
        host_buffers[name] = (host, dev, shape)
    return addr, host_buffers, stream


def execute(ctx, addr, stream):
    """Register every tensor address, then launch with the API that exists."""
    for name, dev in addr.items():
        ctx.set_tensor_address(name, dev)
    start = time.time()
    if hasattr(ctx, "execute_async_v3"):        # TensorRT 10+
        ctx.execute_async_v3(stream.handle)
    else:                                       # TRT ≤ 9
        ctx.execute_async_v2(list(addr.values()), stream.handle)

    stream.synchronize()
    end = time.time()
    print(f"Inference only (TRT kernel) time: {(end - start)*1000:.2f} ms")

# ---------------------------------------------------------------------------


def img_to_fp16(img):                 # HWC uint8 → 1×3×H×W FP16 in [0,1]
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    return t.half()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--left",   default="assets/left.png")
    ap.add_argument("--right",  default="assets/right.png")
    ap.add_argument("--K",      default="assets/K.txt")
    ap.add_argument("--out",    default="output/")
    ap.add_argument("--z_far",  type=float, default=10.0)
    ap.add_argument("--denoise",action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---------------- TensorRT ------------------------------------------------
    engine = load_engine(args.engine)
    ctx    = engine.create_execution_context()
    addr, host_bufs, stream = allocate_buffers(engine)

    # engine expects 1×3×480×640 for both inputs (left/right)
    H, W = 480, 640

    # ---------------- read + resize ------------------------------------------
    imgL = cv2.resize(imageio.imread(args.left),  (W, H), cv2.INTER_LINEAR)
    imgR = cv2.resize(imageio.imread(args.right), (W, H), cv2.INTER_LINEAR)

    tL = img_to_fp16(imgL)
    tR = img_to_fp16(imgR)

    # ---------------- copy to host buffers -----------------------------------
    host_bufs["left"][0][:]  = tL.flatten()
    host_bufs["right"][0][:] = tR.flatten()

    # copy host→device
    for name, (hbuf, dptr, _) in host_bufs.items():
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cuda.memcpy_htod_async(dptr, hbuf, stream)

    # ---------------- inference ----------------------------------------------
    execute(ctx, addr, stream)

    # copy output device→host
    for name, (hbuf, dptr, _) in host_bufs.items():
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cuda.memcpy_dtoh_async(hbuf, dptr, stream)
    stream.synchronize()

    disp = host_bufs["disp"][0].reshape(host_bufs["disp"][2])[0, 0]  # H×W

    # ---------------- visualisation ------------------------------------------
    vis = vis_disparity(disp)
    imageio.imwrite(os.path.join(args.out, "vis.png"),
                    np.concatenate([imgL, vis], axis=1))
    logging.info("Saved vis.png")

    # ---------------- depth + cloud ------------------------------------------
    with open(args.K) as f:
        K = np.array(list(map(float, f.readline().split()))).reshape(3, 3)
        baseline = float(f.readline())

    depth = K[0, 0] * baseline / disp
    breakpoint()
    np.save(os.path.join(args.out, "depth_meter.npy"), depth)
    logging.info("Saved depth_meter.npy")

    xyz  = depth2xyzmap(depth, K)
    breakpoint()
    pcd  = toOpen3dCloud(xyz.reshape(-1, 3), imgL.reshape(-1, 3))
    mask = (xyz[:, :, 2] > 0) & (xyz[:, :, 2] <= args.z_far)
    pcd  = pcd.select_by_index(np.flatnonzero(mask.flatten()))
    if args.denoise:
        pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
    o3d.io.write_point_cloud(os.path.join(args.out, "cloud.ply"), pcd)
    logging.info("Saved cloud.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    breakpoint()
    print(f"Point cloud has {points.shape[0]} points.")
    print(f"Example points:\n{points[:5]}")



if __name__ == "__main__":
    main()
