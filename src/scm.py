


import functools
import statistics as s

import pandas as pd

def rule_for(scm, var: str):
    return scm[var][0]

def exos(scm):

    return [k for (k, (_, q)) in scm.items() if q == "exo"]

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
    return (f1, "exo")

def wrap_1(f, var1):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        return f(v1)
    return (f1, "endo")


def wrap_2(f, var1, var2):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        v2 = val(scm, ctx, var2)
        return f(v1, v2)
    return (f1, "endo")

def wrap_3(f, var1, var2, var3):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        v2 = val(scm, ctx, var2)
        v3 = val(scm, ctx, var3)
        return f(v1, v2, v3)
    return (f1, "endo")

def wrap_4(f, var1, var2, var3, var4):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        v2 = val(scm, ctx, var2)
        v3 = val(scm, ctx, var3)
        v4 = val(scm, ctx, var4)
        return f(v1, v2, v3, v4)
    return (f1, "endo")

def wrap_5(f, var1, var2, var3, var4, var5):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        v2 = val(scm, ctx, var2)
        v3 = val(scm, ctx, var3)
        v4 = val(scm, ctx, var4)
        v5 = val(scm, ctx, var5)
        return f(v1, v2, v3, v4, v5)
    return (f1, "endo")

def wrap_6(f, var1, var2, var3, var4, var5, var6):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        v2 = val(scm, ctx, var2)
        v3 = val(scm, ctx, var3)
        v4 = val(scm, ctx, var4)
        v5 = val(scm, ctx, var5)
        v6 = val(scm, ctx, var6)
        return f(v1, v2, v3, v4, v5, v6)

    return (f1, "endo")




def effect(pdf, x, x_val, y, rs):

    n = len(pdf.index)

    cov_prob = pdf[rs].value_counts() / n

    p0 = pdf[x] == x_val

    def weigted_term(index):

        preds = [pdf[r] == v for (r, v) in zip(rs, index)]

        p = functools.reduce(lambda acc, p: acc & p, preds, p0)

        return s.mean(pdf[p][y]) * cov_prob[index]

    ws = functools.reduce(lambda a, b: a + b, [weigted_term(index) for index in cov_prob.index])

    return (ws, cov_prob)


def effect_iwp(pdf, x, x_val, y, rs):

    pdfx = pdf[pdf[x] == x_val]

    prs = pd.DataFrame(pdf[rs].value_counts()).rename(columns={0: "nrs"}, inplace=False)
    psx = pd.DataFrame(pdfx[rs].value_counts()).rename(columns={0: "nx"}, inplace=False)


    prop_pdf = pd.merge(prs, psx, right_index=True, left_index=True)
    prop_pdf["prop"] = prop_pdf.nx / prop_pdf.nrs

    a = pd.merge(pdfx, prop_pdf, right_index=True, left_on=rs)

    a["py"] = a[y] / a.prop

    return sum(a.py) / (len(pdf[y]))

