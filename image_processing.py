from scipy import ndimage as ndi


def gaussian_filter(chromato):
    return ndi.gaussian_filter(chromato, sigma=5)

def gauss_laplace(chromato):
    return ndi.gaussian_laplace(chromato, sigma=5)


def gauss_multi_deriv(chromato):
    return ndi.gaussian_gradient_magnitude(chromato, sigma=5)

def prewitt(chromato):
    return ndi.prewitt(chromato)

def sobel(chromato):
    return ndi.sobel(chromato)