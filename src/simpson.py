
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

    for zi in zs:
        py =   prob_y_when_x_z(pdf, y0, x0, zi)
#        pz =   prob_z_when_x(pdf, zi, x0)
        pz =   prob_z(pdf, zi)
        acc = acc + py * pz

    return acc

def adjust_y_when_x_cond_z(pdf, y0, x0):

    zs = set(pdf.z)

    acc = 0

    for zi in zs:
        py_xz = prob_y_when_x_z(pdf, y0, x0, zi)
        pz = prob_z(pdf, zi)
        acc = acc + py_xz * pz

    return acc


def prob_do_z_slow(pdf, x0, y0):

    acc = 0
    c = 0
    ce = 0

    zs = set(pdf.z)

    print(len(zs))

    for zi in zs:
        # zi = 6

            pz = pdf[(pdf.z == zi) ].size / pdf.size

            pdfr = pdf[(pdf.x == x0) & (pdf.z == zi)]

            if pdfr.size > 0:
                py = pdfr[(pdfr.y == y0)].size / pdfr.size

                acc = acc + py * pz

                ce = ce+1

            c = c+1

    print(acc)
    print(c, ce)




def mk_pdf(n):

    scm = mk_scm(n)
    ctxs = val_all(scm)
    pdf = pd.DataFrame(ctxs)

    return pdf


def test(n, pdf, x0, y0):

    scm_do = mk_scm_do(n, x0, np.array(pdf.z))
    ctxs_do = val_all(scm_do)
    pdf_do = pd.DataFrame(ctxs_do)

    p_0 = prob_y_when_x(pdf, y0, x0)
    p_1 = adjust_y_when_x_cond_z(pdf, y0, x0)

    p_do_1 = prob_y_when_x(pdf_do, y0, x0)
    p_do_2 = prob_y_when_x_cond_z(pdf_do, y0, x0)

    return (p_0, p_1, p_do_1, p_do_2)






np.floor(npr.normal(loc=0.0, scale=0.00001, size=15) * 5) / 5

n = 2000
s.mean(list(np.floor(npr.normal(loc=0.0, scale=5, size=n))+0.5))

s.mean(list(f_uy(n)()))

n = 100

z = npr.choice([0, 1], n)
x = npr.choice([0, 1], n)

z

#def f_ux(n): return lambda: np.floor(npr.normal(loc=0.0, scale=10, size=n)*5)/5

p_male = 357/(357+343)
p_female = 343/(357+343)


f_x(z)

def f_x(z):
    return np.where((z==1), npr.choice([1, 0], z.size, p=[87/357, 1-87/357]),
                            npr.choice([1, 0], z.size, p=[263/343, 1-263/343]))



def f_y(x, z):
    return np.where((z==1), np.where(x == 1, npr.choice([1, 0], z.size, p=[0.93, .07]),
                                             npr.choice([1, 0], z.size, p=[0.87, .13])),
                            np.where(x == 1, npr.choice([1, 0], z.size, p=[0.73, .27]),
                                             npr.choice([1, 0], z.size, p=[0.69, .31]))
         )

def f_z(n): return lambda: npr.choice([1, 0], n, p=[p_male, p_female])


def mk_scm(ns):
    return {
        "z": wrap_0(f_z(ns)),

        "x": wrap_1(f_x, "z"),
        "y": wrap_2(f_y, "x", "z"),
    }

def mk_scm_do(ns, x0, z=None):
    return {
        "z": wrap_0(f_z(ns)),

        "x": wrap_0(lambda : x0),
        "y": wrap_2(f_y, "x", "z"),
    }



n=1000

(pdf) = mk_pdf(n)

pdfxy = pdf[["x", "y"]].drop_duplicates()


print(pdfxy.size)

test(n, pdf, 1, 1)
test(n, pdf, 0, 1)

#test(n, pdf, pdfxy.loc[1672].x, pdfxy.loc[1672].y)

import matplotlib
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
x = np.random.normal(size=1000)
plt.hist(x, density=True, bins=30)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data');

plt.show()


#pdfxy.loc[0]




acc = []
acc2 = []
accdo=[]
for index, r in pdfxy.iterrows():
#    print("=================")
#    print(r.x, r.y)
    (po, pa, pdo, _) = (test(n, pdf, r.x, r.y))


    e = abs(pa-pdo) #/max(pa, pdo)
    if pdo > 0:
        acc = acc + [(po*100, e * 100, pdo)]
        acc2 = acc2 + [e * 100]
        accdo = accdo + [pdo * 100]

acc.sort(key=lambda t:t[1], reverse=True)
acc2.sort( reverse=True)
accdo.sort( reverse=True)


print(len(acc2))

print(s.mean(acc2))
print(s.median(acc2))
print(s.stdev(acc2))

print("=========")

print(s.mean(accdo))
print(s.median(accdo))
print(s.stdev(accdo))


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
