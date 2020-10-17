

import numpy.random as npr

import pandas as pd
import numpy as np
import statistics as s
import math


from src.scm import *


# Model page 63




def prob_x_y(pdf, x0, y0):

    pdf1 = pdf[(pdf.x==x0) & (pdf.y==y0)]

    p1 = pdf1.size / pdf.size

    return p1


def prob_z(pdf, z0):
    return pdf[pdf.z == z0].size / pdf.size


def prob_y_when_x_z(pdf, y0, x0, z0):

    pdf1 = pdf[(pdf.x==x0) & (pdf.z == z0)]

    p1 = pdf1[pdf1.y==y0].size / pdf1.size if pdf1.size > 1 else 0

    return p1


def prob_y_when_x(pdf, y0, x0):

    pdf1 = pdf[(pdf.x==x0)]

    p1 = pdf1[pdf1.y==y0].size / pdf1.size if pdf1.size > 1 else 0

    return p1


def prob_z_when_x(pdf, z0, x0):

    pdf1 = pdf[(pdf.x==x0)]

    p1 = pdf1[pdf1.z==z0].size / pdf1.size if pdf1.size > 0 else 0

    return p1


def prob_y_when_x_cond_z(pdf, y0, x0):

    zs = set(pdf.z)

    acc = 0
    cz = 0

    for zi in zs:
        py =   prob_y_when_x_z(pdf, y0, x0, zi)
        if py > 0:
            cz = cz+1
        pz =   prob_z_when_x(pdf, zi, x0)
#        pz =   prob_z(pdf, zi)
        acc = acc + py * pz

    return (acc, cz)



def adjust_y_when_x_cond_z(pdf, y0, x0):

    zs = set(pdf.z)

    acc = 0

    for zi in zs:
        py_xz = prob_y_when_x_z(pdf, y0, x0, zi)
        pz = prob_z(pdf, zi)
        acc = acc + py_xz * pz

    return acc




def mk_pdf(n):

    scm = mk_scm(n)
    ctxs = val_all(scm)
    pdf = pd.DataFrame(ctxs)

    return pdf


def test(n, pdf, x0, y0):

    scm_do = mk_scm_do(n, x0, np.array(pdf.ux),  np.array(pdf.uy),  np.array(pdf.z))
#    scm_do = mk_scm_do(n, x0, np.array(pdf.ux), np.array(pdf.uy),  np.array(pdf.z))

    ctxs_do = val_all(scm_do)
    pdf_do = pd.DataFrame(ctxs_do)

    p_0 = prob_y_when_x(pdf, y0, x0)
    p_1 = adjust_y_when_x_cond_z(pdf, y0, x0)

    p_do_1 = prob_y_when_x(pdf_do, y0, x0)
    p_do_2, cz = prob_y_when_x_cond_z(pdf_do, y0, x0)

    print(pdf_do[(pdf_do.x == x0) & (pdf_do.y == y0)].size / pdf_do.size)

    return (p_0, p_1, p_do_1, p_do_2, cz, pdf_do)


def test_range(n, pdf, x0, y0):

    scm_do = mk_scm_do(n, x0, np.array(pdf.ux),  np.array(pdf.uy),  np.array(pdf.z))
    ctxs_do = val_all(scm_do)
    pdf_do = pd.DataFrame(ctxs_do)

    p_0 = prob_y_when_x(pdf, y0, x0)
    p_1 = adjust_y_when_x_cond_z(pdf, y0, x0)

    p_do_1 = prob_y_when_x(pdf_do, y0, x0)
    p_do_2 = prob_y_when_x_cond_z(pdf_do, y0, x0)

    return (p_0, p_1, p_do_1, p_do_2)


def test_z(n, x0, y0):

    scm = mk_scm(n)
    ctxs = val_all(scm)
    pdf = pd.DataFrame(ctxs)

    p_1 = prob(pdf, x0, y0)
    p_2 = prob_x_y(pdf, x0, y0)

    scm_do = mk_scm_do(n, x0, np.array(pdf.ux),  np.array(pdf.uy),  np.array(pdf.z))


    scm_do = mk_scm_do(n, x0, np.array(pdf.ux), np.array(pdf.uy), np.array(pdf.z))


    #scm_do = mk_scm_do(n, x0, np.array(pdf.ux), np.array(pdf.uy), np.array(pdf.z))
    scm_do = mk_scm_do(n, x0, f_ux(n)(), f_uy(n)(), f_z(n)())

    ctxs_do = val_all(scm_do)

    pdf_do = pd.DataFrame(ctxs_do)

    zs = set(pdf.z)

    r =[(zi, prob_y_when_x_z(pdf, x0, zi, y0), prob_y_when_x_z(pdf_do, x0, zi, y0),
         prob_z_when_x(pdf, z0=zi, x0=x0), prob_z_when_x(pdf_do, z0=zi, x0=x0))
         for zi in zs]

    return r


'''

s.mean(list(np.floor(npr.normal(loc=0.0, scale=5, size=n))+0.5))

s.mean(list(f_uy(n)()))

np.floor(npr.normal(loc=0.0, scale=0.00001, size=15) * 5) / 5
'''

n = 2000
n=10

#def f_ux(n): return lambda: np.floor(npr.normal(loc=0.0, scale=10, size=n)*5)/5


def f_ux(n): return lambda: npr.random(n)*10-5
def f_uy(n): return lambda: npr.random(n)*10-5

def f_z(n): return lambda: npr.choice([i for i in range(0, 21)], n)


def f_x(z, ux): return np.where((z+ux) > 10, 1, 0)


def f_y(x, z, uy): return np.where(z < uy*5, x, x * (-1))


def mk_scm(ns):
    return {
        "ux": wrap_0(f_ux(ns)),
        "uy": wrap_0(f_uy(ns)),
        "z": wrap_0(f_z(ns)),

        "x": wrap_2(f_x, "z", "ux"),

        "y": wrap_3(f_y, "x", "z", "uy"),

    }

def mk_scm_do(ns, x0, ux=None, uy=None, z=None):
    return {
        "ux": wrap_0(f_ux(ns)),
        "uy": wrap_0(f_uy(ns)),
        "z": wrap_0(f_z(ns)),

        "x": wrap_0(lambda: x0),

        "y": wrap_3(f_y, "x", "z", "uy"),

    }



n=10000

(pdf) = mk_pdf(n)

pdfxy = pdf[["x", "y"]].drop_duplicates()

pdf.ux.drop_duplicates()
pdf.uy.drop_duplicates()

print(pdfxy.size)

#test(n, pdf, pdfxy.loc[1672].x, pdfxy.loc[1672].y)

#pdfxy.loc[0]

acc = {
    "pa": [],
    "pdo": [],
    "pdo2": [],
    "e": [],
    "cz": [],
    }

#for index, r in pdfxy[pdfxy.index == 0].iterrows():
for index, r in pdfxy.iterrows():
#    print("=================")
#    print(r.x, r.y)
    (po, pa, pdo, pdo2, cz, pdf_do) = (test(n, pdf, r.x, r.y))

    e = abs(pa-pdo)
    if pdo > 0:
        acc["pa"] =  acc["pa"] + [pa*100]
        acc["pdo"] = acc["pdo"] + [pdo*100]
        acc["pdo2"] = acc["pdo2"] + [pdo2*100]
        acc["e"] = acc["e"] + [e*100]
        acc["cz"] = acc["cz"] + [cz]

print(acc)

a = pdf_do[pdf_do.y==57]
a.size/pdf_do.size

b = pdf[(pdf.x==40) & (pdf.y==57)]
b.size/pdf.size

adjust_y_when_x_cond_z(pdf, 57, 40)
prob_y_when_x(pdf, 57, 40)

prob_y_when_x(pdf_do, 57, 40)


def mk_scm_do_bis(ns, x0):
    return {
        "ux": wrap_0(f_ux(ns)),
        "uy": wrap_0(f_uy(ns)),
        "z": wrap_0(f_z(ns)),

        "x": wrap_0(lambda : x0),

        "yr": wrap_3(f_y, "x", "z", "uy"),
        "y": wrap_1(f_ry, "yr"),

    }


scm_do = mk_scm_do_bis(5000, 40)

ctxs_do = val_all(scm_do)
pdf_do = pd.DataFrame(ctxs_do)



print(adjust_y_when_x_cond_z(pdf, 60, 40))
print(prob_y_when_x(pdf_do, 60, 40))

print(prob_y_when_x(pdf, 60, 40))

len(set(pdf_do.y))

set(pdf[pdf.x==40].y)

set(pdf[(pdf.x==40) & (pdf.y==60) & (pdf.z==20)].z)

pdf[(pdf.x==40) & (pdf.z==20)]

pdf[(pdf.x==40) & (pdf.z==20)]

pdf[(pdf.z==20)].size / pdf.size

pdf.size

len(set(pdf.z))
print("fini")
print(q.pearsonr(acc["pa"], acc["pdo"]))


plt.close()
plt.scatter(acc["cz"], acc["e"] )

plt.show()


print(s.mean(acc["e"]))
print(s.median(acc["e"]))

print(s.stdev(acc["e"]))
print("=========")

print(s.mean(acc["pdo"]))
print(s.median(acc["pdo"]))

print(s.stdev(acc["pdo"]))

from scipy.stats.stats import pearsonr


import scipy.stats.stats as q

q.pearsonr(accpa, accpa)

import matplotlib.pyplot as plt
import numpy as np

plt.close()
plt.hist(accpdo, density=True, bins=100)  # `density=False` would make counts
plt.show()

plt.close()
plt.hist(accpa, density=True, bins=100)  # `density=False` would make counts
plt.show()

plt.close()
plt.hist(acc2, density=True, bins=100)  # `density=False` would make counts
plt.show()

max(acc2)


import numpy as np
import matplotlib.pyplot as plt

# Create data
N = 60
g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N))
g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N))
g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N))

data = (g1, g2, g3)
colors = ("red", "green", "blue")
groups = ("coffee", "tea", "water")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

for data, color, group in zip(data, colors, groups):
x, y = data
ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()


print(max(acc2))
print(min(acc2))

l1 = list(filter(lambda x: x < 20, acc2))
print(len(l1))


l = list(pdf.ux)

min(l)

import matplotlib.pyplot as plt

x = [value1, value2, value3,....]
plt.hist(x, bins = number of bins)
plt.show()

for index, r in pdfxy.iterrows():
    print("=================")
    print(r.x, r.y)
    print(test(n, pdf, r.x, r.y))


test(n, 1, 3)


prob(pdf, 8, 15)
prob_do_z(pdf, 8, 15)


acc =0
for index, r in pdfxy.iterrows():
    print("=================")
    print(r.x, r.y)
    print(prob(pdf, r.x, r.y))
    print(prob_do_z(pdf, r.x, r.y))

    p = prob(pdf, r.x, r.y)
    acc = acc + p * prob_x(pdf, r.x, r.y)


x0 = 5

n=10000

x0 = 2
scm_do = mk_scm_do(n, x0, np.array(pdf.ux),  np.array(pdf.uy),  np.array(pdf.z))

ctxs_do = val_all(scm_do)

pdf_do = pd.DataFrame(ctxs_do)

# a =

pdf_do[(pdf_do.x==2) & (pdf_do.y==3)].size

prob_y_when_x(pdf_do, 3, 2)

pdf1 =

test(n, 2, 4)

x0=2
y0=3

pdfxy_do = pdf_do[["x", "y"]].drop_duplicates()
pdfxy_do.size

zs = set(pdf_do.z)
xs = set(pdfxy_do.x)

acc=0
for zi in zs:
    for xi in xs:
        p1 = prob_z(pdf_do, zi)
        p2 = prob_z_when_x(pdf_do, zi, xi)

        assert abs(p1-p2) < 10e-3

        print(p1, p2)


print(acc)



acc=0
for (index, r) in pdfxy_do.iterrows():
    print("========")
    print(r.x, r.y)
    print(prob_y_when_x(pdf_do, r.y, r.x),
          prob_y_when_x_cond_z(pdf_do, r.y, r.x))


    py_x = prob_y_when_x(pdf_do, r.y, r.x)
    px = pdf_do[(pdf_do.x==r.x) ].size/pdf_do.size

    # acc = acc + prob_x_y(pdf_do, r.x, r.y)  # py_x * px
    acc = acc +  py_x * px

print(acc)


pdf1 = pdf_do[pdf_do.x==x0]

p = pdf1[pdf1.y==13].size/pdf1.size

for zi in zs:
    p1 = prob_z(pdf_do, zi)
    p2 = prob_z_when_x(pdf_do, zi, x0)
    print(p1, p2)


for index, r in pdfxy.iterrows():
    print("=================")
    print(r.x, r.y)
    print(test(n, r.x, r.y))



for index, r in pdfxy.iterrows():
    print("=================")
    print(r.x, r.y)
    print(test_z(n, r.x, r.y))

pdf = pdf_do
x0 = 14
y0=21

n = 50000



prob_x_y




z0 = 9

x0 = 27
y0 = 36


scm_do = mk_scm_do(n)

ctxs = val_all(scm_do)

pdf_do=pd.DataFrame(ctxs)


p1 = prob_x_y(pdf_do, x0, y0)

print("p1", p1)

prob(pdf_do, x0, y0)
prob_do_z(pdf_do,  x0, y0)

prob_z(pdf_do, x0, z0, y0)

print(pdf_do[pdf_do.z==5].size/pdf_do.size)

pdf1=pdf[pdf.x==x0]
print(pdf1[pdf1.z==5].size/pdf1.size)


a = \
    pdf[(pdf.x==x0) & (pdf.y==y0)]

pdf[(pdf.x==x0) ]

set(pdf.z)


scm = mk_scm(n)

ctxs = val_all(scm)

pdf=pd.DataFrame(ctxs)


p2 = prob_x_y(pdf, x0, y0)

print("p2", p2)


for zi in set(pdf.z):
    print("=====  ", zi)
    print(prob_x_y(pdf_do, x0, y0), prob_x_y(pdf, x0, y0)
          )
    print(prob_y_when_x_z(pdf_do, x0, zi, y0), prob_y_when_x_z(pdf, x0, zi, y0)
      )

    print(  prob_z(pdf_do, zi), prob_z(pdf, zi)
            )


print(pdf[pdf.z==7].size/pdf.size)


print(max(list(pdf.x)))
print(min(list(pdf.x)))

prob_do_z(pdf,  x0, y0)

prob_z(pdf, x0, z0, y0)


prob(pdf, x0, y0)

prob_do_z_slow(pdf,  x0, y0)



pdf = pdf_do
x0 = 14
y0=21


def f (a, b):
    return a + b


f(*[1, 2])