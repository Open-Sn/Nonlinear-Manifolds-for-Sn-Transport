# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:07:14 2023

@author: jean.ragusa 
"""
import numpy as np


class AQ:
    def __init__(self, snorder):
        if type(snorder) != int:
            raise Exception("AQ::snorder should be an integer")
        if snorder % 2 != 0:
            raise Exception("AQ::snorder should be even")
        # get quadrature
        mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
        # normalize weights to sum to 1
        w_q /= np.sum(w_q)

        # save in object
        self.ndir = snorder
        self.mu_q = np.copy(mu_q)
        self.w_q = np.copy(w_q)


if __name__ == "__main__":
    print("Running AQ as the main code:")

    myAQ = AQ(8)
    print("ndir = ", myAQ.ndir)
    print("directions = ", myAQ.mu_q)
