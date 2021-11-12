import torch

def sqrt_matrix(mtx):
    size = mtx.size()
    u, e, v = torch.svd(mtx, some=False)
    k_c = size[0]
    for i in range(size[0]):
        if e[i] < 0.00001:
            k_c = i
            break
    d = e[:k_c].pow(0.5)
    m_step1 = torch.mm(v[:, :k_c], torch.diag(d))
    m = torch.mm(m_step1, v[:, :k_c].t())
    return m

def sqrt_inv_matrix(mtx):
    size = mtx.size()
    u, e, v = torch.svd(mtx, some=False)
    k_c = size[0]
    for i in range(size[0]):
        if e[i] < 0.00001:
            k_c = i
            break
    d = e[:k_c].pow(-0.5)
    m_step1 = torch.mm(v[:, :k_c], torch.diag(d))
    m = torch.mm(m_step1, v[:, :k_c].t())
    return m
    

def feature_transform(content_feature, style_feature, alpha=1.0):
    content_feature = content_feature.type(dtype=torch.float64)
    style_feature = style_feature.type(dtype=torch.float64)
    
    content_feature1 = content_feature.squeeze(0)
    cDim = content_feature1.size()
    content_feature1 = content_feature1.reshape(cDim[0], -1)
    c_mean = torch.mean(content_feature1, 1, keepdim=True)
    content_feature1 = content_feature1 - c_mean
    content_cov = torch.mm(content_feature1, content_feature1.t()).div(cDim[1]*cDim[2]-1)
    
    style_feature1 = style_feature.squeeze(0)
    sDim = style_feature1.size()
    style_feature1 = style_feature1.reshape(sDim[0], -1)
    s_mean = torch.mean(style_feature1, 1, keepdim=True)
    style_feature1 = style_feature1 - s_mean
    style_cov = torch.mm(style_feature1, style_feature1.t()).div(sDim[1]*sDim[2]-1)
    
    sqrtInvU = sqrt_inv_matrix(content_cov)
    sqrtU = sqrt_matrix(content_cov)
    C = torch.mm(torch.mm(sqrtU, style_cov), sqrtU)
    sqrtC = sqrt_matrix(C)
    T = torch.mm(torch.mm(sqrtInvU, sqrtC), sqrtInvU)
    target_feature = torch.mm(T, content_feature1)
    target_feature = target_feature + s_mean
    res_feature = target_feature.reshape(cDim[0], cDim[1], cDim[2]).unsqueeze(0).float()
    
    res_feature = alpha * res_feature + (1.0 - alpha) * content_feature
    return res_feature.type(dtype=torch.float32)