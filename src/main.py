

import random as rand
import pandas as pd
import numpy as np

import math as m

def rule_for(scm, var: str):
    return scm[var]

def val(scm, ctx, var: str):
    if var in ctx.keys():
        return ctx[var]
    else:
        f = rule_for(scm, var)
        v = f(scm, ctx)
        ctx[var] = v
        return v

def val_all(scm):
    ctx = {}
    for var in scm.keys():
        val(scm, ctx, var)
    return ctx

def wrap_0(f):
    def f1(scm, ctx):
        return f()
    return f1

def wrap_1(f, var1):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        return f(v1)
    return f1

def wrap_2(f, var1, var2):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        v2 = val(scm, ctx, var2)
        return f(v1, v2)
    return f1

def wrap_3(f, var1, var2, var3):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        v2 = val(scm, ctx, var2)
        v3 = val(scm, ctx, var3)
        return f(v1, v2, v3)
    return f1

def f_ux() : return rand.randrange(1,50)
def f_uy() : return rand.randrange(-5, 5)
def f_uz() : return rand.randrange(-5, 5)

#def f_uy() : return 0
#def f_uz() : return 0

def f_y(x, uy): return 2 * x + uy
def f_y_bis(y): return m.floor(y / 3) + rand.randrange(-1,1)

def f_z(y, ybis, uz): return m.floor(3 * y + ybis +  uz)


def f_x_n(x):
    return m.floor((x/100) * 10)

def f_y_n(x):
    return m.floor((x/200) * 10)

def f_z_n(x):
    return m.floor((x/600) * 10)

scm = {
    "x" : wrap_0(f_ux),
    "uy" : wrap_0(f_uy),
    "uz" : wrap_0(f_uz),

    "y": wrap_2(f_y, "x", "uy"),
    "ybis": wrap_1(f_y_bis, "y"),
    "z": wrap_3(f_z, "y", "ybis", "uz", ),

    "xn": wrap_1(f_x_n, "x"),
    "yn": wrap_1(f_y_n, "y"),
    "zn": wrap_1(f_z_n, "z"),
}

ctx={}

r = val(scm, ctx, "z")

print(r)




n = 2000000
ctxs = [val_all(scm) for _ in range(n)]

pdf=pd.DataFrame(ctxs)

pdf[pdf.index==n/2]

#r = pdf[(pdf.x==47) & (pdf.y==89) & (pdf.z==298)
#        & (pdf.xn == 0) & (pdf.yn == 0) & (pdf.zn == 0)].size / pdf.size

nr = pdf[(pdf.x==47) & (pdf.y==89) & (pdf.ybis==29) & (pdf.z==298)].size / len(scm)
print(nr)
r = pdf[(pdf.x==47) & (pdf.y==89) & (pdf.ybis==29) & (pdf.z==298)].size / pdf.size

print(r)

p1 = pdf[(pdf.x==47)].size / pdf.size

pdf2 = pdf[(pdf.x==47)]
p2 = pdf2[(pdf2.y==89)].size / pdf2.size

pdf3 = pdf[(pdf.y==89)]
p3 = pdf3[(pdf3.ybis==29)].size / pdf3.size


pdf4 = pdf[(pdf.y==89) & (pdf.ybis==29)]
p4 = pdf4[(pdf4.z==298)].size / pdf4.size


r1 = p1 * p2 * p3 * p4
print(r1)

r*r*r

pdf1 = pdf[(pdf.y==500)]  # & (pdf.y<=200)]

pdf[pdf.index==0]
list(pdf1.loc(0))

pdf1.columns

pdf1 = pdf

np.corrcoef(pdf1.x, pdf1.z)


np.corrcoef(pdf1.y, pdf1.z)


r/r1
pdf.size