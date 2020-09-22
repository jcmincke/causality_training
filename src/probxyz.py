
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



def prob_x_when_z(pdf, x0, z0):

    pdf1 = pdf[(pdf.z==z0)]

    p1 = pdf1[pdf1.x==x0].size / pdf1.size if pdf1.size > 1 else 0

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

def inverse_prob_weight_y_when_x_cond_z(pdf, y0, x0):

    zs = set(pdf.z)

    acc = 0

    for zi in zs:
        pyxz = pdf[(pdf.x==x0) & (pdf.y==y0) & (pdf.z==zi)].size/pdf.size
        if pyxz > 0:
            p_x_when_z = prob_x_when_z(pdf, x0, zi)
            #print(zi, pyxz, p_x_when_z)


            acc = acc + pyxz / p_x_when_z

    return acc