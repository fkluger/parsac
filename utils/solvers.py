import torch
import utils.residual_functions


def vanishing_point(X):
    l1 = X[..., 0, 6:9]
    l2 = X[..., 1, 6:9]

    vp = torch.cross(l1, l2, dim=-1)

    vp_norm = torch.norm(vp, keepdim=True, dim=-1) + 1e-8
    vp = vp / vp_norm

    return vp


def homography(X):

    dims = X.size()
    A_dims = tuple(list(dims)[:-2] + [9, 9])

    A = torch.zeros(A_dims, device=X.device)

    A[..., 0:8:2, 5] = -1
    A[..., 0:8:2, 3:5] = -X[..., :, 0:2]
    A[..., 0:8:2, 6:8] = X[..., :, 0:2]
    A[..., 0:8:2, 6] *= X[..., :, 3]
    A[..., 0:8:2, 7] *= X[..., :, 3]
    A[..., 0:8:2, 8] = X[..., :, 3]
    A[..., 1:8:2, 2] = 1
    A[..., 1:8:2, 0:2] = X[..., :, 0:2]
    A[..., 1:8:2, 6:8] = -X[..., :, 0:2]
    A[..., 1:8:2, 6] *= X[..., :, 2]
    A[..., 1:8:2, 7] *= X[..., :, 2]
    A[..., 1:8:2, 8] = -X[..., :, 2]

    try:
        u, s, v = torch.svd(A)
        h = v[..., -1].to(X.device)
    except:
        h = torch.zeros(list(dims)[:-2] + [9], device=X.device)

    return h


def fundamental_7point(X):

    # adapted from https://imkaywu.github.io/blog/2017/06/fundamental-matrix/

    dims = X.size()
    A_dims = tuple(list(dims)[:-2] + [7, 9])

    A = torch.ones(A_dims, device=X.device)

    A[..., 0:2] = X[..., 0:2]
    A[..., 0:2] *= X[..., 2][..., None]
    A[..., 3:5] = X[..., 0:2]
    A[..., 3:5] *= X[..., 3][..., None]
    A[..., 6:8] = X[..., 0:2]

    B = A.cpu()
    u, s, v = torch.svd(B)

    degenerate = s[..., -3].to(X.device).abs() < 1e-5

    fs = v[..., -2:].transpose(-1, -2).to(X.device)
    fs = fs / torch.clamp(torch.norm(fs, dim=-1, keepdim=True), min=1e-9)
    Fs = fs.contiguous().view(list(dims)[:-2] + [2, 3, 3])

    D = torch.zeros(list(dims)[:-2] + [2, 2, 2], device=Fs.device)
    D_tmp = torch.zeros(list(dims)[:-2] + [3, 3], device=D.device)
    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                D_tmp[..., 0] = Fs[..., i1, :, 0]
                D_tmp[..., 1] = Fs[..., i2, :, 1]
                D_tmp[..., 2] = Fs[..., i3, :, 2]
                D[..., i1, i2, i3] = torch.det(D_tmp)

    poly_coeffs = torch.zeros(list(dims)[:-2] + [4], device=X.device)
    poly_coeffs[..., 3] = -D[..., 1, 0, 0]+D[..., 0, 1, 1]+D[..., 0, 0, 0]+D[..., 1, 1, 0]+D[..., 1, 0, 1]-D[..., 0, 1, 0]-D[..., 0, 0, 1]-D[..., 1, 1, 1]
    poly_coeffs[..., 2] = D[..., 0, 0, 1]-2*D[..., 0, 1, 1]-2*D[..., 1, 0, 1]+D[..., 1, 0, 0]-2*D[..., 1, 1, 0]+D[..., 0, 1, 0]+3*D[..., 1, 1, 1]
    poly_coeffs[..., 1] = D[..., 1, 1, 0]+D[..., 0, 1, 1]+D[..., 1, 0, 1]-3*D[..., 1, 1, 1]
    poly_coeffs[..., 0] = D[..., 1, 1, 1]
    poly_coeffs = poly_coeffs / torch.clamp(torch.abs(poly_coeffs[..., 3][..., None]), min=1e-9)
    poly_coeffs = poly_coeffs * torch.sign(poly_coeffs[..., 3][..., None])

    # find roots of polynomial det(a*F1 + (1-a)*F2)
    # https://en.wikipedia.org/wiki/Companion_matrix
    companion = torch.diag_embed(torch.ones(2, dtype=torch.float32, device=X.device), offset=-1)
    companion = companion.view([1]*(len(dims)-2) + [3, 3]).expand(list(dims[:-2]) + [3, 3])
    companion = torch.clone(companion)
    companion[..., 0, -1] -= poly_coeffs[..., 0]
    companion[..., 1, -1] -= poly_coeffs[..., 1]
    companion[..., 2, -1] -= poly_coeffs[..., 2]

    eigvals_complex, _ = torch.linalg.eig(companion)
    eigvals = eigvals_complex.real.to(X.device)

    isnotreal = torch.logical_not(torch.isreal(eigvals_complex.to(X.device)))

    # compute residuals for all three solutions and select the best
    f = fs[..., 0, None, :] * eigvals[..., None] + fs[..., 1, None, :] * (1 - eigvals[..., None])
    X_ = X[..., None, :, :].expand(list(dims)[:-2] + [3] + list(dims)[-2:])
    res = utils.residual_functions.sampson_distance(X_, f)
    mean_res = res.mean(-1) + isnotreal.float() * res.max() * 10
    best_fs = torch.argmin(mean_res, dim=-1, keepdim=True)[..., None].expand(list(dims)[:-2] + [1, 9])
    h = torch.gather(f, -2, best_fs).squeeze(-2)

    h = h * torch.logical_not(degenerate).float()[..., None]

    return h


minimal_solver = {"vp": vanishing_point, "homography": homography, "fundamental": fundamental_7point}