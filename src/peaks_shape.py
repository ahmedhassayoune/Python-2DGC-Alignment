import cv2

def calculate_moment(img, width, height, p, q) -> int:
    r"""Renvoie le moment d'ordre (p + q) de l'image f de taille (width * height).

    Parameters
    ----------
    img: np.array
        L'image binaire
    width * height : int * int
        Les dimensions de l'image
    p : int
    q : int
    
    Returns
    -------
    moment : int
        Le moment d'ordre (p + q).
    """
    moment = 0
    for x in range(width):
        for y in range(height):
            moment += img[x][y] * pow(x, p) * pow(y, q)

    return moment

def centre_of_gravity(img, width, height):
    """
    Renvoie les coordonnées du centre de gravité (x, y) de la forme de l'image f de taille (width * height).

    Parameters
    ----------
    img: np.array
        L'image binaire
    width * height : int * int
        Les dimensions de l'image
        
    Returns
    -------
    x, y : int * int
        Les coordonnées du centre de gravité.
    """
    moment_00 = calculate_moment(img, width, height, 0, 0)
    x_centre = calculate_moment(img, width, height, 1, 0) / moment_00
    y_centre = calculate_moment(img, width, height, 0, 1) / moment_00
    return x_centre, y_centre

def central_moment(img, width, height, p, q) -> int:
    """
    Renvoie le moment central d'ordre (p + q) de l'image f de taille (width * height) centré sur le centre de gravité de l'image.
    Le moment central est insensible aux translations.

    Parameters
    ----------
    img: np.array
        L'image binaire
    width * height : int * int
        Les dimensions de l'image
    p : int
    q : int
    
    Returns
    -------
    moment : int
        Le moment central d'ordre (p + q).
    """
    x_centre, y_centre = centre_of_gravity(img, width, height)

    central_moment = 0
    for x in range(width):
        for y in range(height):
            central_moment += img[x][y] * pow(x - x_centre, p) * pow(y - y_centre, q)
    return central_moment

def scale_moment(img, width, height, mu_00):
    """
    Renvoie une séléction de moments insensibles aux translations d'ordres prédéfinis de l'image f de taille (width * height) centré sur le centre de gravité de l'image.

    Parameters
    ----------
    img: np.array
        L'image binaire
    width * height : int * int
        Les dimensions de l'image
    mu_00 : int
        Le moment central d'ordre 0, on le passe en argument pour des soucis de performances
        
    Returns
    -------
    moment : int * int * int * int * int * int * int
        Les moments insensibles aux translations d'ordres prédéfinis centré.
    """
    x_centre, y_centre = centre_of_gravity(img, width, height)

    m_20 = 0
    m_02 = 0
    m_11 = 0
    m_30 = 0
    m_12 = 0
    m_21 = 0
    m_03 = 0

    for x in range(width):
        for y in range(height):
            px = img[x][y]
            m_20 += px * pow(x - x_centre, 2) * pow(y - y_centre, 0)
            m_02 += px * pow(x - x_centre, 0) * pow(y - y_centre, 2)
            m_11 += px * pow(x - x_centre, 1) * pow(y - y_centre, 1)
            m_30 += px * pow(x - x_centre, 3) * pow(y - y_centre, 0)
            m_12 += px * pow(x - x_centre, 1) * pow(y - y_centre, 2)
            m_21 += px * pow(x - x_centre, 2) * pow(y - y_centre, 1)
            m_03 += px * pow(x - x_centre, 0) * pow(y - y_centre, 3)
            

    v_20 = m_20 / pow(mu_00, 1 + (2 + 0)/2)
    v_02 = m_02 / pow(mu_00, 1 + (0 + 2)/2)
    v_11 = m_11 / pow(mu_00, 1 + (1 + 1)/2)
    v_30 = m_30 / pow(mu_00, 1 + (3 + 0)/2)
    v_12 = m_12 / pow(mu_00, 1 + (1 + 2)/2)
    v_21 = m_21 / pow(mu_00, 1 + (2 + 1)/2)
    v_03 = m_03 / pow(mu_00, 1 + (0 + 3)/2)
    return v_20, v_02, v_11, v_30, v_12, v_21, v_03

def rotation_moment(img, width, height):
    """
    Renvoie les 7 moments insensibles aux rotations et à l'échelle de l'image f de taille (width * height) centré sur le centre de gravité de l'image.

    Parameters
    ----------
    img: np.array
        L'image binaire
    width * height : int * int
        Les dimensions de l'image
        
    Returns
    -------
    moment : int * int * int * int * int * int * int
        Les 7 moments insensibles aux rotations d'ordre prédéfinis centré.
    """
    mu_00 = central_moment(img, width, height, 0, 0)
    v_20, v_02, v_11, v_30, v_12, v_21, v_03 = scale_moment(img, width, height, mu_00)

    phi_1 = v_20 + v_02
    phi_2 = pow(v_20 - v_02, 2) + pow(2 * v_11, 2)
    phi_3 = pow(v_30 - 3 * v_12, 2) + pow(3 * v_21 - v_03, 2)
    phi_4 = pow(v_30 + v_12, 2) + pow(v_21 + v_03, 2)
    phi_5 = (v_30 - 3 * v_12) * (v_30 + v_12) * (pow(v_30 + v_12, 2) - 3 * pow(v_21 + v_03, 2)) + (3 * v_21 - v_03) * (v_21 + v_03) * (3 * pow(v_30 + v_12, 2) - pow(v_21 + v_03, 2) )
    phi_6 = (v_20 - v_02) * (pow(v_30 + v_12, 2) - pow(v_21 + v_03, 2)) + 4 * v_11 * (v_30 + v_12) * (v_21 + v_03)
    phi_7 = (3 * v_21 - v_03) * (v_30 + v_12) * (pow(v_30 + v_12, 2) - 3 * pow(v_21 + v_03, 2)) - (v_30 - 3 * v_12) * (v_21 + v_03) * (3 * pow(v_30 + v_12, 2) - pow(v_21 + v_03, 2) )

    return phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7

'''def get_final_moment(img):
    """
    Redimensionne et binarise l'image pour ne garder que les pixels ayant une valeurs et obtenir les moments insensibles aux rotations, a l'échelle et centré de l'image.

    Parameters
    ----------
    img: np.array
        L'image en BGR

    Returns
    -------
    moment : int * int * int * int * int * int * int
        Les 7 moments insensibles aux rotations d'ordre prédéfinis centré.
    """
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 254, 255, 0)
    width, height, _ = img.shape
    mask = (thresh[...] == 255)

    binary_img = thresh.copy()
    binary_img[~mask] = 1
    binary_img[mask] = 0
    return rotation_moment(binary_img, width, height)'''