
import numpy.random as npr

import pandas as pd
import numpy as np

from src.scm import *


# Model page 63

def f_ux(n): return lambda: np.floor(npr.random(n) * 0)
def f_uy(n): return lambda: np.floor(npr.random(n) * 0)
def f_ue(n): return lambda: np.floor(npr.random(n) * 10)
def f_uz(n): return lambda: np.floor(npr.random(n) * 0)
def f_ua(n): return lambda: np.floor(npr.random(n) * 10)



def f_z(e, a, uz): return np.floor((0.85 * e + 1.15 * a + uz))

def f_x(e, z, ux): return np.floor((1.25 * e + 1.05 * z + ux))

def f_y(x, z, a, uy): return np.floor((0.95 * x + 0.75 * z + 0.15 * a + uy))


def mk_scm(ns):
    return {
        "e": wrap_0(f_ue(ns)),
        "a": wrap_0(f_ua(ns)),
        "ux": wrap_0(f_ux(ns)),
        "uy": wrap_0(f_uy(ns)),
        "uz": wrap_0(f_uz(ns)),

        "z": wrap_3(f_z, "e", "a", "uz"),

        "x": wrap_3(f_x, "e", "z", "ux"),
        "y": wrap_4(f_y, "x", "z", "a", "uy"),

    }



def mk_scm_do(ns):
    return {
        "e": wrap_0(f_ue(ns)),
        "a": wrap_0(f_ua(ns)),
        "ux": wrap_0(f_ux(ns)),
        "uy": wrap_0(f_uy(ns)),
        "uz": wrap_0(f_uz(ns)),

        "z": wrap_3(f_z, "e", "a", "uz"),

        "x": wrap_0(lambda : 5),
        # "x": wrap_3(f_x, "e", "z", "ux"),
        "y": wrap_4(f_y, "x", "z", "a", "uy"),

    }

# "y": wrap_4(f_y, "x", "z", "a", "uy"),


x0 = 5
y0 = 9

scm = mk_scm_do(10000)

ctxs = val_all(scm)

pdf=pd.DataFrame(ctxs)

prob(pdf, x0, y0)

a = \
    pdf[(pdf.x==x0) & (pdf.y==y0)]

pdf[(pdf.x==x0) ]



scm = mk_scm(200000)

ctxs = val_all(scm)

pdf=pd.DataFrame(ctxs)

max(list(pdf.x))
min(list(pdf.x))

prob(pdf, x0, y0)
prob_do_ze(pdf,  x0, y0)
prob_do_za(pdf,  x0, y0)

prob_do_ze_slow(pdf,  x0, y0)




def prob(pdf, x0, y0):

    pdf1 = pdf[pdf.x==x0]

    p1 = pdf1[pdf1.y==y0].size / pdf1.size

    print(p1)



def prob_do_ze(pdf, x0, y0):

    acc = 0
    c = 0
    pdfxy = pdf[(pdf.x == x0) & (pdf.y == y0)]

    if pdfxy.size > 0:

        zs = set(pdfxy.z)

        for zi in zs:

            # zi = 37

            pdfxyz = pdfxy[pdfxy.z == zi]

            if pdfxyz.size > 0:

                es = set(pdfxyz.e)

                for ei in es:

                    # ei = 42

                    pdfxyze = pdfxyz[pdfxyz.e == ei]

                    if pdfxyze.size > 0:
                        pze = pdf[(pdf.z == zi) & (pdf.e == ei)].size / pdf.size

                        py = pdfxyze.size / pdf[(pdf.x == x0) & (pdf.z == zi) & (pdf.e == ei)].size

                        acc = acc + py * pze

                        c = c+1

    print(acc)
    print(c)

def prob_do_ze_slow(pdf, x0, y0):

    acc = 0
    c = 0
    ce = 0

    zs = set(pdf.z)
    es = set(pdf.e)

    print(len(zs))
    print(len(es))

    for zi in zs:
        # zi = 6

        for ei in es:
            # ei = 11

            pze = pdf[(pdf.z == zi) & (pdf.e == ei)].size / pdf.size

            pdfr = pdf[(pdf.x == x0) & (pdf.z == zi) & (pdf.e == ei)]

            if pdfr.size > 0:
                py = pdfr[(pdfr.y == y0)].size / pdfr.size

                acc = acc + py * pze

                ce = ce+1

            c = c+1

    print(acc)
    print(c, ce)

def prob_do_za(pdf, x0, y0):

    acc = 0
    c = 0
    pdfxy = pdf[(pdf.x == x0) & (pdf.y == y0)]

    if pdfxy.size > 0:

        zs = set(pdfxy.z)

        for zi in zs:

            # zi = 37

            pdfxyz = pdfxy[pdfxy.z == zi]

            if pdfxyz.size > 0:

                aas = set(pdfxyz.a)

                for ai in aas:

                    # ei = 42

                    pdfxyza = pdfxyz[pdfxyz.a == ai]

                    if pdfxyza.size > 0:
                        pza = pdf[(pdf.z == zi) & (pdf.a == ai)].size / pdf.size

                        py = pdfxyza.size / pdf[(pdf.x == x0) & (pdf.z == zi) & (pdf.a == ai)].size

                        acc = acc + py * pza

                        c = c+1

    print(acc)
    print(c)


prob_do_ze(pdf, 100, 104)

prob_do_ze_slow(pdf, 100, 104)


208
100
0.0006316938283938282
20800 431


prob_do_za(pdf)

#n = 2000000
#ctxs = [val_all(scm) for _ in range(n)]

pdf=pd.DataFrame(ctxs)

print(pdf)