
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
        print(v1)
        return f(v1)
    return (f1, "endo")


def wrap_2(f, var1, var2):
    def f1(scm, ctx):
        v1 = val(scm, ctx, var1)
        v2 = val(scm, ctx, var2)
        print(v1.size, v2.size)
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
