import numpy as np

from constants import *


def kepler(M, e, tol=1e-12):
    """
        Solver for Kepler's equation:
            E - e sinE = M
        Using the Newton-Raphson method:
            f(E) = E - e sinE - M = 0
            f'(E) = 1 - e cosE
            x_(n+1) = x_n - f(x_n)/f'(x_n)
    :param M: mean anomaly, in radians
    :param e: eccentricity
    :param tol: tolerance. Default=1e-12
    :return: tuple(E, iters): E, the eccentric anomaly (in radians); iters, number of iterations
    """
    # we use M for the first guess of E
    E = M
    iters = 1
    while True:
        diff = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= diff
        if np.abs(diff) < tol:
            return E, iters
        iters += 1


def eccentric_anomaly_to_theta(E, e):
    """
        Calculates the true anomaly, given an eccentric anomaly and the orbit's eccentricity,
        from the formula:
            tan(E/2) = sqrt((1-e)/(1+e))*tan(theta/2)
            => theta = 2*arctan(tan(E/2)*sqrt((1+e)/(1-e)))
    :param E: eccentric anomaly, in radians
    :param e: eccentricity
    :return: the true anomaly, in radians
    """
    return 2 * np.arctan(np.tan(E / 2.0) * np.sqrt((1.0 + e) / (1.0 - e)))


def theta_to_r_distance(theta, e, a):
    """
        Calculates radial distance for a certain theta, e and a
        from the formula:
            r = a * (1-e²) / (1 + e cos(theta))
    :param theta: true anomaly, in radians
    :param e: eccentricity
    :param a: semi-major axis, in km
    :return:
    """
    return a * (1.0 - e*e) / (1.0 + e * np.cos(theta))


def thetas_to_alts(theta, e, a):
    """
        Calculates altitude from theta, for a particular orbit (e and a)
    :param theta: true anomaly, in radians
    :param e: eccentricity
    :param a: semi-major axis, in km
    :return: altitude, in km
    """
    return theta_to_r_distance(theta, e, a) - R_EARTH


def ts_to_alts(ts, e, a):
    """
        Calculates altitudes for a list of timestamps for a given orbit (e and a), assuming t0 is at perigee
        M = n * t
        E calculates from M using Newton-Raphson
        E -> theta
        theta -> altitude
    :param ts: array of timestamps, in seconds
    :param e: eccentricity
    :param a: semi-major axis, in km
    :return: tuple of two arrays: altitudes and thetas, both with the same size as ts parameter
    """
    Ms = np.sqrt(G_EARTH / np.power(a, 3)) * ts  # n * t
    Es = np.empty_like(Ms)
    # this one can't be vectorized, as we're running Newton-Raphson on each individual entry
    for i in range(len(Es)):
        Es[i], _ = kepler(Ms[i], e)
    thetas = eccentric_anomaly_to_theta(Es, e)
    hs = thetas_to_alts(thetas, e, a)
    return hs, thetas


def theta_h_to_x_y(thetas, heights):
    """
        Convert (theta, height) pairs to (x, y) pairs
    :param thetas: array of thetas (true anomalies) in radians
    :param heights: array of altitudes, in km
    :return: tuple of two arrays: xs and ys, in kms, both with same size as thetas
    """
    return np.multiply(np.cos(thetas), R_EARTH + heights), np.multiply(np.sin(thetas), R_EARTH + heights)


def theta_to_x_y(theta, e, a):
    """
        Calculate x,y position for a given theta in a given orbit (e and a)
    :param theta: true anomaly, in radians
    :param e: eccentricity
    :param a: semi-major axis, in km
    :return: tuple of x,y, in km
    """
    return theta_h_to_x_y(theta, thetas_to_alts(theta, e, a))


def theta_to_eccentric_anomaly(theta, e):
    """
        Calculate the eccentric anomaly from true anomaly
            Uses the formula:
            tan(E/2) = sqrt((1-e)/(1+e))*tan(theta/2)
    :param theta: true anomaly, in radians
    :param e: eccentricity
    :return: eccentric anomaly, in radians
    """
    return 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta / 2.0))


def eccentric_anomaly_to_delta_t(E, e, a):
    """
        Calculate (t-t_0) for a given eccentric anomaly, for a given orbit (e and a)
            Uses formula:
            E - e.sin(E) = n.(t-t_0)
    :param E: eccentric anomaly, in radians
    :param e: eccentricity
    :param a: semi-major axis, in km
    :return: t-t0, in seconds
    """
    return (E - e * np.sin(E)) / np.sqrt(G_EARTH / np.power(a, 3))


def delta_theta_to_delta_t(theta1, theta2, e, a):
    """
        Calculates time difference between satellite position theta1 and theta2, for a given orbit (e and a)
            = (t-t0) of theta2 - (t-t0) of theta1
    :param theta1: true anomaly in radians - first point
    :param theta2: true anomaly in radians - second point
    :param e: eccentricity
    :param a: semi-major axis, in km
    :return: time difference between passage at theta1 and theta2, in seconds
    """

    E1 = theta_to_eccentric_anomaly(theta1, e)
    delta_t_1 = eccentric_anomaly_to_delta_t(E1, e, a)

    E2 = theta_to_eccentric_anomaly(theta2, e)
    delta_t_2 = eccentric_anomaly_to_delta_t(E2, e, a)

    return delta_t_2 - delta_t_1