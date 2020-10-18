import numpy.random as npr

import pandas as pd
import numpy as np

from src.scm import *

import statistics as s

npr.choice([i for i in range(0, 21)], n)

n = 10000

s.mean(npr.random(n) * 1000 *(1+npr.random(n) * 0.2))

p0 =  0.1
p1 =  0.6
p2 =  0.3
def f_segment(n) : return lambda: npr.choice([0, 1, 2], p=[p0, p1, p2], size=n)
#def f_segment(n) : return lambda: npr.choice([0, 1, 2], p=[0.3333, 0.3333, 0.3334], size=n)
#def f_segment(n) : return lambda: npr.choice([0, 1, 2], p=[0, 0, 1], size=n)
#def f_segment(n) : return lambda: npr.choice([2], p=[1], size=n)
def f_season(n): return lambda: npr.choice([0, 1], n)
def f_promo(n): return lambda: npr.choice([0, 1], n)


def f_promo_mult(n): return lambda: (1 + npr.random(n) * 0.2)
def f_season_mult(n): return lambda: (1 + npr.random(n))



# (1 + npr.random(promo.size) * 0.2))

def f_dep_promo(segment):
    p = np.where((segment == 0),
                 npr.choice([0, 1], p=[0.9, 0.1], size=segment.size),
                 np.where((segment == 1),
                          npr.choice([0, 1], p=[0.3, 0.7], size=segment.size),
                          npr.choice([0, 1], p=[0.5, 0.5], size=segment.size)))

    return p

def f_ca_seg_0(n): return lambda: (npr.random(n) * 50)
def f_ca_seg_1(n): return lambda: (npr.random(n) * 175 + 25)
def f_ca_seg_2(n): return lambda: (npr.random(n) * 500 + 200)

def f_ca(segment, ca_0, ca_1, ca_2):
    ca = np.where((segment == 0),
                 ca_0,
                 np.where((segment == 1),
                          ca_1,
                          ca_2))
    return ca


def f_ca_promo(ca, promo, promo_mult):
    ca = np.where((promo == 0),
                  ca,
                  ca * promo_mult)
    return ca

def f_ca_season(ca, season, season_mult):
        ca = np.where((season == 0),
                      ca,
                      ca * season_mult)
        return ca



def f_y(ca, season): return ca * (1 + season * 1)

exo = ["s", "seg", "p", "ca_0", "ca_1", "ca_2", "smult", "pmult"]

def mk_scm(ns):
    return {
        "seg": wrap_0(f_segment(ns)),
        "s": wrap_0(f_season(ns)),
        #"p": wrap_0(f_promo(ns)),
        "p": wrap_1(f_dep_promo, "seg"),

        "pmult": wrap_0(f_promo_mult(ns)),
        "smult": wrap_0(f_season_mult(ns)),

        "ca_0": wrap_0(f_ca_seg_0(ns)),
        "ca_1": wrap_0(f_ca_seg_1(ns)),
        "ca_2": wrap_0(f_ca_seg_2(ns)),

        "ca": wrap_4(f_ca, "seg", "ca_0", "ca_1", "ca_2"),
        "ca_promo": wrap_3(f_ca_promo, "ca", "p", "pmult"),
        "y": wrap_3(f_ca_season, "ca_promo", "s", "smult")

    }


def do(scm, ctxs, key, value):

    exo = exos(scm)

    scm1 = {k: wrap_0(lambda kx=k: np.array(ctxs[kx])) for k in exo}

    scm2 = {k: v for (k, v) in scm.items() if k not in exo}

    scm3 = {**scm1, **scm2}

    n = len(ctxs[exo[0]])
    scm3[key] = wrap_0(lambda: np.full(n, value))

    return scm3


def y_for_promo(scm, exo, ctxs, promo):

    scm_p = do(scm, ctxs, "p", promo)
    ctxs_p = val_all(scm_p)
    pdf = pd.DataFrame(ctxs_p)

    return pdf[(pdf.s == 1)]


def exact_uplift(scm, exo, ctxs):
    pdf_1 = y_for_promo(scm, exo, ctxs, 1)
    pdf_0 = y_for_promo(scm, exo, ctxs, 0)

    return s.mean(pdf_1.y) - s.mean(pdf_0.y)


def ri(pdf):
    return pdf.reset_index(drop=True)

n = 500000
scm = mk_scm(n)
ctxs = val_all(scm)
pdf = pd.DataFrame(ctxs)

#print(s.mean(pdf[(pdf.s==0) & (pdf.p==1)].y))

eu = exact_uplift(scm, exo, ctxs)

print("exact uplift ", eu)


part_during = pdf[(pdf.s==1) & (pdf.p==1) ].head(int(n/5)).sample(frac=1)
non_part_during = pdf[(pdf.s==1) & (pdf.p==0) ].head(int(n/5)).sample(frac=1)

part_before = pdf[(pdf.s==0) & (pdf.p==0) ].head(int(n/5)).sample(frac=1)
non_part_before = pdf[(pdf.s==0) & (pdf.p==0) ].head(int(n/5)).sample(frac=1)

uplift = s.mean(part_during.y) - s.mean(non_part_during.y)


print(uplift)


# seg 0

uplift_0 = s.mean(part_during[part_during.seg == 0].y) - s.mean(non_part_during[non_part_during.seg == 0].y)
uplift_1 = s.mean(part_during[part_during.seg == 1].y) - s.mean(non_part_during[non_part_during.seg == 1].y)
uplift_2 = s.mean(part_during[part_during.seg == 2].y) - s.mean(non_part_during[non_part_during.seg == 2].y)

uplift2 = uplift_0 * p0 + uplift_1 * p1 + uplift_2 * p2

print(uplift2)


exos(scm)

# emnos

k  = (s.mean(non_part_during.y) / s.mean(non_part_before.y))
print("k", k)

uplift2 = (s.mean(part_during.y)
           - s.mean(part_before.y) * k
           )

print(uplift2)


uplifti = (ri(part_during[["y"]]) -  (ri(non_part_during[["y"]]))
           )

uplift3 = (s.mean(uplifti.y)
           )

print(uplift3)


s.mean(pdf.y)

non_part_before[non_part_before.ca == 0]


s.mean(pdf.y.head(2000))

print(part_during.size)
print(non_part_during.size)
print(part_before.size)
print(non_part_before.size)


a = (ri(non_part_during[["ca"]]) / ri(non_part_before[["ca"]])

a[pd.isnull(a.ca)]