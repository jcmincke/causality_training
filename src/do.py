

import numpy.random as npr

import pandas as pd
import numpy as np
import statistics as s
import math
import matplotlib.pyplot as plt


from src.scm import *
from src.probxyz import *

# Model page 63


def mk_pdf_old(n, x0):

    scm = mk_scm(n)

    if x0 is None:
        ctxs = val_all(scm)
        pdf = pd.DataFrame(ctxs)

        return pdf
    else:
        c = 0
        acc_pdf = None
        while c < n:
            ctxs = val_all(scm)
            pdf = pd.DataFrame(ctxs)
            pdf = pdf[pdf.x == x0]
            if acc_pdf is None:
                acc_pdf = pdf
            else:
                acc_pdf = acc_pdf.append(pdf)

            c = acc_pdf.size / len(ctxs.keys())

        return acc_pdf


def mk_pdf(n, x0):

    scm = mk_scm(n)

    if x0 is None:
        ctxs = val_all(scm)
        pdf = pd.DataFrame(ctxs)

        return pdf
    else:
        c = 0
        acc_pdf = None
        while c < n:
            ctxs = val_all(scm)
            pdf = pd.DataFrame(ctxs)
            if acc_pdf is None:
                acc_pdf = pdf
            else:
                acc_pdf = acc_pdf.append(pdf)

            pdfx = acc_pdf[acc_pdf.x == x0]

            c = pdfx.size / len(ctxs.keys())

            print(c, acc_pdf.size)

        return acc_pdf

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


def test_e(n_do, pdf, x0):


    scm_do = mk_scm_do(n_do, x0, None, None,  np.array(pdf.z))

    ctxs_do = val_all(scm_do)
    pdf_do = pd.DataFrame(ctxs_do)

    ys = set(pdf[pdf.x == x0].y)

    ye = 0
    yea = 0

    for y0 in ys:

        # y0 = 12.0

        print(y0)

        p_0 = prob_y_when_x(pdf, y0, x0)
        #p_1 = adjust_y_when_x_cond_z(pdf, y0, x0)
        p_1 = inverse_prob_weight_y_when_x_cond_z(pdf, y0, x0)

        ye = ye + p_0 * y0
        yea = yea + p_1 * y0


    ys = set(pdf_do[pdf_do.x == x0].y)

    yea_do_1 = 0
    yea_do_2 = 0
    c=0

    print(len(ys))
    for y0 in ys:

        p_do_1 = prob_y_when_x(pdf_do, y0, x0)
        p_do_2, cz = prob_y_when_x_cond_z(pdf_do, y0, x0)

        print(c)
        c=c+1

        yea_do_1 = yea_do_1 + p_do_1 * y0
        yea_do_2 = yea_do_2 + p_do_2 * y0


    return (ye, yea, yea_do_1, yea_do_2, pdf_do)



'''

s.mean(list(np.floor(npr.normal(loc=0.0, scale=5, size=n))+0.5))

s.mean(list(f_uy(n)()))

np.floor(npr.normal(loc=0.0, scale=0.00001, size=15) * 5) / 5

z = np.round(npr.normal(loc=0.0, scale=1, size=10) * 20)
x = np.round(npr.normal(loc=z, scale=1, size=10))
y = np.round(npr.normal(loc=(x+z), scale=5, size=10))

print(x)
print(z)
print(y)

print(s.mean(x))
print(s.median(x))
'''


# def f_z(n): return lambda: np.round(npr.normal(loc=0.0, scale=0.1, size=n) * 20)

def f_z(n): return lambda: npr.choice([i for i in range(-1, 2)], n)


def f_x(z): return np.round(npr.normal(loc=z, scale=0.5))


def f_y(x, z): return np.round(npr.normal(loc=(x+z), scale=0.5))


def mk_scm(ns):
    return {
        "z": wrap_0(f_z(ns)),

        "x": wrap_1(f_x, "z"),

        "y": wrap_2(f_y, "x", "z"),

    }


def mk_scm_do(ns, x0, ux=None, uy=None, z=None):
    return {
#        "z": wrap_0(lambda: z),
        "z": wrap_0(f_z(ns)),

        "x": wrap_0(lambda : np.full(ns, x0)),

        "y": wrap_2(f_y, "x", "z"),

    }



'''
del(n)
del(x0)
'''

def run(n, x0):

    (pdf) = mk_pdf(n, x0)

#    pdf = pdf[pdf.x == x0]
    (ye, yea, yea_do_1, yea_do_2, pdf_do) = test_e(1000000, pdf, x0)

    print(ye, yea, yea_do_1, yea_do_2)

run(5000000, 2.0)

def comp(n, x0):

    (pdf) = mk_pdf(n, x0)

    zs = list(set(pdf.z))

    zs.sort()

    pdf = pdf[pdf.x == x0]
    n = 100000

    scm_do = mk_scm_do(n, x0, None, None,  np.array(pdf.z))

    ctxs_do = val_all(scm_do)
    pdf_do = pd.DataFrame(ctxs_do)


    for zi in zs:

        # zi= -91
        pdfz = pdf[(pdf.x==x0) & (pdf.z==zi)]
        pdfz_do = pdf_do[(pdf_do.x==x0) & (pdf_do.z==zi)]

#        print(zi,  pdfz_do.size / len(pdf.columns))
        print(zi, pdfz.size / len(pdf.columns), pdfz_do.size / len(pdf.columns))

comp(1000000, 1.0)

def expected_y_when_x_do(n, x0):

    scm_do = mk_scm_do(n, x0, None, None, None)

    ctxs_do = val_all(scm_do)
    pdf_do = pd.DataFrame(ctxs_do)

    ys = set(pdf_do[pdf_do.x == x0].y)

    yea_do_1 = 0
    yea_do_2 = 0
    c=0

    print(len(ys))
    for y0 in ys:

        p_do_1 = prob_y_when_x(pdf_do, y0, x0)
        p_do_2, cz = prob_y_when_x_cond_z(pdf_do, y0, x0)

        print(c)
        c=c+1

        yea_do_1 = yea_do_1 + p_do_1 * y0
        yea_do_2 = yea_do_2 + p_do_2 * y0

    return (yea_do_1, yea_do_2)

expected_y_when_x_do(100000, 1)

comp(200, 0.0)

run(1000, 0)

plt.close()
plt.scatter(acc["yea"], acc["yea_do_1"] )

plt.show()


print(acc["e"])

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
