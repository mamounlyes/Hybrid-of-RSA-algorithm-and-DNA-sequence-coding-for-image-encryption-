import tkinter as tk
from tkinter import filedialog
import hashlib
import textwrap
import cv2
import numpy as np
from scipy.integrate import odeint
from bisect import bisect_left as bsearch
import matplotlib.pyplot as plt
import dippykit as dip
def image_selector():  # returns path to selected image
    path = "NULL"
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    if path != "NULL":
        print("Image loaded!")
    else:
        print("Error Image not loaded!")
    return path
my_im = image_selector()
X = cv2.imread(my_im)

print("---------------------")
print(X[1,1,1])
print("---------------------")
# cv2_imshow(my_img)
#plt.imshow(X, cmap="gray")
#print(X.shape)
dimensions = X.shape

# height, width, number of channels in image
height = X.shape[0]
width = X.shape[1]
channels = X.shape[2]

#dip.imshow(X, cmap="gray")
#dip.show()
print(X[1,1,1])
def prime_generator(end):
    for n in range(150, 200):     # n starts from 2 to end
        for x in range(2, n):   # check if x can be divided by n
            if n % x == 0:      # if true then n is not prime
                break
        else:                   # if x is found after exhausting all values of x
            yield n             # generate the prime


gL = prime_generator(10)       # give first 10 prime numbers
pp=list(gL)
print(pp)
P=pp[0]
Q=pp[2]
e=pp[4]
print("first prime number (P) = ",P)
print("second  prime number (Q) = ",Q)
NNN = P * Q
eulerTotient = (P - 1) * (Q - 1)
print(" P * Q = N = ",NNN)
print("Value of (P - 1) * (Q - 1) = ",eulerTotient)
#find e
def GCD(a, b):
    if a == 0:
        return b;
    return GCD(b % a, a)

while GCD(e, eulerTotient) != 1:
    e = pp[1]
print("Value of e = ",e)

def power(aaaa,dddd,nnnn):
  ans=1;
  while dddd!=0:
       if dddd%2==1:

          ans=((ans%nnnn)*(aaaa%nnnn))%nnnn

       aaaa=((aaaa%nnnn)*(aaaa%nnnn))%nnnn
       dddd>>=1
  return ans;

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = egcd(b % a, a)
        return (g, y - (b // a) * x, x)


dd = egcd(e, eulerTotient)
print("Value of d = ",dd[1])
zzzz=dd[1]

row, col = X.shape[0], X.shape[1]
encrypted = [[0 for x in range(3000)] for y in range(3000)]

def image_encryption(ans, E, N):
    for i in range(0,height ):
        for j in range(0, width):
          r,g,b=X[i,j]
          C1=power(r,E,N)
          C2=power(g,E,N)
          C3=power(b,E,N)
          encrypted[i][j]=[C1,C2,C3]
          C1=C1%256
          C2=C2%256
          C3=C3%256
          X[i,j]=[C1,C2,C3]
    return ans

X=image_encryption(X,e,NNN)
#dip.im_write(X, 'encryptedRSA.jpg')
cv2.imwrite('encryptedRSA.jpg', X)

print("---------------------")
print(X[1,1,1])
print("---------------------")

a, b, c = 10, 2.667, 28
x0, y0, z0 = 0, 0, 0
# DNA-Encoding RULE #1 A = 00, T=01, G=10, C=11
dna = {}
dna["00"] = "A"
dna["01"] = "T"
dna["10"] = "G"
dna["11"] = "C"
dna["A"] = [0, 0]
dna["T"] = [0, 1]
dna["G"] = [1, 0]
dna["C"] = [1, 1]
# DNA xor
dna["AA"] = dna["TT"] = dna["GG"] = dna["CC"] = "A"
dna["AG"] = dna["GA"] = dna["TC"] = dna["CT"] = "G"
dna["AC"] = dna["CA"] = dna["GT"] = dna["TG"] = "C"
dna["AT"] = dna["TA"] = dna["CG"] = dna["GC"] = "T"
# Maximum time point and total number of time points
tmax, N = 100, 10000

def lorenz(Y, t, a, b, c):
    x, y, z = Y
    x_dot = -a * (x - y)
    y_dot = c * x - y - x * z
    z_dot = -b * z + x * y
    return x_dot, y_dot, z_dot


#---------------------------------------------------------------------------------------------------------------------


def split_into_rgb_channels(image):
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    return red, green, blue


# Secure key generation
def securekey(X):

    m, n, c = X.shape
    # Using Flatten function on array 1 to convert the multi-dimensional
    # array to 1-D array
    plainimage = X.flatten()
    key = hashlib.sha256()  # key is made a hash.sha256 object
    key.update(bytearray(plainimage))  # image data is fed to generate digest
    return key.hexdigest(), m, n


def update_lorentz(key):
    key_bin = bin(int(key, 16))[2:].zfill(256)  # covert hex key digest to binary
    k = {}  # key dictionary
    key_32_parts = textwrap.wrap(key_bin, 8)  # slicing key into 8 parts
    num = 1
    for i in key_32_parts:
        k["k{0}".format(num)] = i
        num = num + 1
    t1 = t2 = t3 = 0
    for i in range(1, 12):
        t1 = t1 ^ int(k["k{0}".format(i)], 2)
    for i in range(12, 23):
        t2 = t2 ^ int(k["k{0}".format(i)], 2)
    for i in range(23, 33):
        t3 = t3 ^ int(k["k{0}".format(i)], 2)
    global x0, y0, z0
    x0 = x0 + t1 / 256
    y0 = y0 + t2 / 256
    z0 = z0 + t3 / 256


def decompose_matrix(X):
    blue, green, red = split_into_rgb_channels(X)
    for values, channel in zip((red, green, blue), (2, 1, 0)):
        img = np.zeros((values.shape[0], values.shape[1]), dtype=np.uint8)
        img[:, :] = (values)
        if channel == 0:
            B = np.asmatrix(img)
        elif channel == 1:
            G = np.asmatrix(img)
        else:
            R = np.asmatrix(img)
    return B, G, R


def dna_encode(b1, g1, r1):
    b1 = np.unpackbits(b1, axis=1)
    g1 = np.unpackbits(g1, axis=1)
    r1 = np.unpackbits(r1, axis=1)
    m, n = b1.shape
    r_enc = np.chararray((m, int(n / 2)))
    g_enc = np.chararray((m, int(n / 2)))
    b_enc = np.chararray((m, int(n / 2)))

    for color, enc in zip((b1, g1, r1), (b_enc, g_enc, r_enc)):
        idx = 0
        for j in range(0, m):
            for i in range(0, n, 2):
                enc[j, idx] = dna["{0}{1}".format(color[j, i], color[j, i + 1])]
                idx += 1
                if (i == n - 2):
                    idx = 0
                    break

    b_enc = b_enc.astype(str)
    g_enc = g_enc.astype(str)
    r_enc = r_enc.astype(str)
    return b_enc, g_enc, r_enc


def key_matrix_encode(key, b1):
    # encoded key matrix
    b1 = np.unpackbits(b1, axis=1)
    m, n = b1.shape
    key_bin = bin(int(key, 16))[2:].zfill(256)
    Mk = np.zeros((m, n), dtype=np.uint8)
    x = 0
    for j in range(0, m):
        for i in range(0, n):
            Mk[j, i] = key_bin[x % 256]
            x += 1

    Mk_enc = np.chararray((m, int(n / 2)))
    idx = 0
    for j in range(0, m):
        for i in range(0, n, 2):
            if idx == (n / 2):
                idx = 0
            Mk_enc[j, idx] = dna["{0}{1}".format(Mk[j, i], Mk[j, i + 1])]
            idx += 1
    Mk_enc = Mk_enc.astype(str)
    return Mk_enc


def xor_operation(b1, g1, r1, mk):
    m, n = b1.shape
    bx = np.chararray((m, n))
    gx = np.chararray((m, n))
    rx = np.chararray((m, n))
    b1 = b1.astype(str)
    g1 = g1.astype(str)
    r1 = r1.astype(str)
    for i in range(0, m):
        for j in range(0, n):
            bx[i, j] = dna["{0}{1}".format(b1[i, j], mk[i, j])]
            gx[i, j] = dna["{0}{1}".format(g1[i, j], mk[i, j])]
            rx[i, j] = dna["{0}{1}".format(r1[i, j], mk[i, j])]

    bx = bx.astype(str)
    gx = gx.astype(str)
    rx = rx.astype(str)
    return bx, gx, rx


def gen_chaos_seq(m, n):
    global x0, y0, z0, a, b, c, N
    N = m * n * 4
    x = np.array((m, n * 4))
    y = np.array((m, n * 4))
    z = np.array((m, n * 4))
    t = np.linspace(0, tmax, N)
    f = odeint(lorenz, (x0, y0, z0), t, args=(a, b, c))
    x, y, z = f.T
    x = x[:(N)]
    y = y[:(N)]
    z = z[:(N)]
    return x, y, z
#??????????????????????????????


def sequence_indexing(x, y, z):
    n = len(x)
    fx = np.zeros((n), dtype=np.uint32)
    fy = np.zeros((n), dtype=np.uint32)
    fz = np.zeros((n), dtype=np.uint32)
    seq = sorted(x)
    for k1 in range(0, n):
        t = x[k1]
        k2 = bsearch(seq, t)
        fx[k1] = k2
    seq = sorted(y)
    for k1 in range(0, n):
        t = y[k1]
        k2 = bsearch(seq, t)
        fy[k1] = k2
    seq = sorted(z)
    for k1 in range(0, n):
        t = z[k1]
        k2 = bsearch(seq, t)
        fz[k1] = k2
    return fx, fy, fz


def scramble(fx, fy, fz, b1, r1, g1):
    p, q = b1.shape
    size = p * q
    bx = b1.reshape(size).astype(str)
    gx = g1.reshape(size).astype(str)
    rx = r1.reshape(size).astype(str)
    bx_s = np.chararray((size))
    gx_s = np.chararray((size))
    rx_s = np.chararray((size))

    for i in range(size):
        idx = fz[i]
        bx_s[i] = bx[idx]
    for i in range(size):
        idx = fy[i]
        gx_s[i] = gx[idx]
    for i in range(size):
        idx = fx[i]
        rx_s[i] = rx[idx]
    bx_s = bx_s.astype(str)
    gx_s = gx_s.astype(str)
    rx_s = rx_s.astype(str)

    b_s = np.chararray((p, q))
    g_s = np.chararray((p, q))
    r_s = np.chararray((p, q))

    b_s = bx_s.reshape(p, q)
    g_s = gx_s.reshape(p, q)
    r_s = rx_s.reshape(p, q)
    return b_s, g_s, r_s


def scramble_new(fx, fy, fz, b1, g1, r1):
    p, q = b1.shape
    size = p * q
    bx = b1.reshape(size)
    gx = g1.reshape(size)
    rx = r1.reshape(size)

    bx_s = b1.reshape(size)
    gx_s = g1.reshape(size)
    rx_s = r1.reshape(size)

    bx = bx.astype(str)
    gx = gx.astype(str)
    rx = rx.astype(str)
    bx_s = bx_s.astype(str)
    gx_s = gx_s.astype(str)
    rx_s = rx_s.astype(str)

    for i in range(size):
        idx = fz[i]
        bx_s[idx] = bx[i]
    for i in range(size):
        idx = fy[i]
        gx_s[idx] = gx[i]
    for i in range(size):
        idx = fx[i]
        rx_s[idx] = rx[i]

    b_s = np.chararray((p, q))
    g_s = np.chararray((p, q))
    r_s = np.chararray((p, q))

    b_s = bx_s.reshape(p, q)
    g_s = gx_s.reshape(p, q)
    r_s = rx_s.reshape(p, q)

    return b_s, g_s, r_s

def dna_decode(b1, g1, r1):
    m, n = b1.shape
    r_dec = np.ndarray((m, int(n * 2)), dtype=np.uint8)
    g_dec = np.ndarray((m, int(n * 2)), dtype=np.uint8)
    b_dec = np.ndarray((m, int(n * 2)), dtype=np.uint8)
    for color, dec in zip((b1, g1, r1), (b_dec, g_dec, r_dec)):
        for j in range(0, m):
            for i in range(0, n):
                dec[j, 2 * i] = dna["{0}".format(color[j, i])][0]
                dec[j, 2 * i + 1] = dna["{0}".format(color[j, i])][1]
    b_dec = (np.packbits(b_dec, axis=-1))
    g_dec = (np.packbits(g_dec, axis=-1))
    r_dec = (np.packbits(r_dec, axis=-1))
    return b_dec, g_dec, r_dec


def xor_operation_new(b1, g1, r1, mk):
    m, n = b1.shape
    bx = np.chararray((m, n))
    gx = np.chararray((m, n))
    rx = np.chararray((m, n))
    b1 = b1.astype(str)
    g1 = g1.astype(str)
    r1 = r1.astype(str)
    for i in range(0, m):
        for j in range(0, n):
            bx[i, j] = dna["{0}{1}".format(b1[i, j], mk[i, j])]
            gx[i, j] = dna["{0}{1}".format(g1[i, j], mk[i, j])]
            rx[i, j] = dna["{0}{1}".format(r1[i, j], mk[i, j])]

    bx = bx.astype(str)
    gx = gx.astype(str)
    rx = rx.astype(str)
    return bx, gx, rx


def recover_image(b1, g1, r1, X):
    img = X
    img[:, :, 2] = r1
    img[:, :, 1] = g1
    img[:, :, 0] = b1
   # dip.im_write(img, 'encryptedDNA.jpg')
    cv2.imwrite('encryptedDNA.jpg', img)

    print("saved ecrypted image as encryptedDNA.jpg")
    return img


def decrypt(image, fx, fy, fz, fp, Mk, bt, gt, rt):
    r1, g1, b1 = split_into_rgb_channels(image)
    p, q = rt.shape
    benc, genc, renc = dna_encode(b1, g1, r1)
    bs, gs, rs = scramble_new(fx, fy, fz, benc, genc, renc)
    bx, rx, gx = xor_operation_new(bs, gs, rs, Mk)
    blue, green, red = dna_decode(bx, gx, rx)
    green, red = red, green
    img = np.zeros((p, q, 3), dtype=np.uint8)
    img[:, :, 0] = red
    img[:, :, 1] = green
    img[:, :, 2] = blue
    #dip.im_write(img, 'decryptedDNA.jpg')
    cv2.imwrite('decryptedDNA.jpg', img)

    return img


# program exec9
key, m, n = securekey(X)
update_lorentz(key)

blue, green, red = decompose_matrix(X)
blue_e, green_e, red_e = dna_encode(blue, green, red)
Mk_e = key_matrix_encode(key, blue)
blue_final, green_final, red_final = xor_operation(blue_e, green_e, red_e, Mk_e)
x, y, z = gen_chaos_seq(m, n)
fx, fy, fz = sequence_indexing(x, y, z)
blue_scrambled, green_scrambled, red_scrambled = scramble(fx, fy, fz, blue_final, red_final, green_final)
b1, g1, r1 = dna_decode(blue_scrambled, green_scrambled, red_scrambled)
X = recover_image(b1, g1, r1, X)
print("---------------------")
print(X[1,1,1])
print("---------------------")
X=decrypt(X, fx, fy, fz, X, Mk_e, blue, green, red)
print("---------------------")
print(X[1,1,1])
print("---------------------")
#asd = dip.im_read("C:/Users/DELL/PycharmProjects/pythonProject2/decodedDNA.jpg")
#asd1= dip.float_to_im(asd)

def image_decryption(my_img, D, N):
    for i in range(0, m):
        for j in range(0, n):

              r, g, b = encrypted[i][j]
              M1 = power(r, D, N)
              M2 = power(g, D, N)
              M3 = power(b, D, N)
              my_img[i, j] = [M1, M2, M3]
    return my_img

Xdec=image_decryption(X,zzzz,NNN)
print("---------------------")
print(X[1,1,1])
print("---------------------")
#dip.im_write(X, 'decryptedRSA.jpg')
cv2.imwrite('decryptedRSA.jpg', Xdec)

print(zzzz)
