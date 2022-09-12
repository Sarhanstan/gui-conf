import numpy as np
import time
import numba
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from PIL import Image

class DV:

    def __init__(self,n):
        self.data = np.zeros(n,dtype=np.float32)
        self.size = 0

    def __len__(self):
        return self.size

    def update(self, row):
        for r in row:
            self.add(r)

    def shift(self, row, s):
        for r in row:
            self.add(r + s)

    def add(self, x):

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        data = self.data[:self.size]
        return data
class C:

    def __init__(self):
        self.data = np.zeros((100,))
        self.capacity = 100
        self.size = 0

    def __len__(self):
        return self.size

    def update(self, row):
        for r in row:
            self.add(r)

    def shift(self, row, s):
        for r in row:
            self.add(r + s)

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity,))
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        data = self.data[:self.size]
        return data
class CV:

    def __init__(self,n):
        self.data = np.zeros(n,dtype=np.int32)
        self.size = 0

    def __len__(self):
        return self.size

    def update(self, row):
        for r in row:
            self.add(r)

    def shift(self, row, s):
        for r in row:
            self.add(r + s)

    def add(self, x):

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        data = self.data[:self.size]
        return data



def load_image( infilename ) :
    img = Image.open( infilename, mode="L")
    img.load()
    img.convert("L")
    data = np.array( img, dtype=np.uint8 )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )



@numba.jit
def solve(a,b):
    return np.linalg.solve(a,b)

def Range(d):
    mn=np.min(d)
    mx=np.max(d)
    n=len(d)
    shift=(255/(mx-mn+1))
    dr=np.zeros((n,n),dtype=np.float32)
    print(mn,mx,shift)
    for i in range(n):
            for j in range(n):
                dr[i,j]=(d[i,j]-mn)*shift
    return dr


@numba.jit
def Laplas_P(i,j,n,data,p=1):
    if n==0:
        return data[i,j]
    else:
        return Laplas_P(i,j-1,n-1,data,p)+Laplas_P(i,j+1,n-1,data,p)+Laplas_P(i-1,j,n-1,data,p)+Laplas_P(i+1,j,n-1,data,p)-(4*p)*Laplas_P(i,j,n-1,data,p)

@numba.jit
def Laplas(k,smooth_r,p=1):
    n1=len(smooth_r)
    lpls=np.zeros((n1,n1),dtype=np.float32)
    exp=np.zeros((n1+k+k,n1+k+k),dtype=np.float32)
    for i in range(k,n1+k):
        for j in range(k,n1+k):
            exp[i,j]=smooth_r[i-k,j-k]
    exp=paint_border(exp,1)
    for i in range(n1):
        for j in range(n1):
            lpls[i][j]=np.abs(Laplas_P(i+k,j+k,k,exp,p))
    return lpls

@numba.jit
def Laplas_rep(n,data,p=1):
    if n==1:
        return Laplas(1,data,p)
    else:
        return Laplas_rep(n-1,Laplas(1,data,p),p)

def edge_smooth(data,ss,shift=0):
    n=len(data)-shift-shift
    ss=np.int(ss)
    ns=n//ss
    for k in range(ns):
        for j in range(1+shift,n-1-shift):
            b=data[k*ss-2+shift,j]
            a=(data[k*ss+1+shift,j]-b)/3
            data[k*ss-1+shift,j]=a+b
            data[k*ss+shift,j]=2*a+b
    for k in range(ns):
        for i in range(1+shift,n-1-shift):
            b=data[i,k*ss-2+shift]
            a=(data[i,k*ss+1+shift]-b)/3
            data[i,k*ss-1+shift]=a+b
            data[i,k*ss+shift]=2*a+b
    return data

def paint_border(data,p):
    n=len(data)
    im=data.copy()
    for j in range(0,n):
        for i in range(p):
            im[p-i-1,j]=im[p-i,j]
            im[n-p+i,j]=im[n-p+i-1,j]
    for i in range(0,n):
        for j in range(p):
            im[i,p-j-1]=im[i,p-j]
            im[i,n-p+j]=im[i,n-p+j-1]
    return im

def WhP(k,data):
    n1=len(data)
    A=np.zeros((n1,n1),dtype=np.float32)
    for i in range(n1):
        for j in range(n1):
            if data[i,j]<=k:
                A[i,j]=0
            else:
                A[i,j]=255
    return A

@numba.jit
def Disp(k,data):
    n1=len(data)
    d=np.zeros((n1,n1),dtype=np.float32)
    exp=np.zeros((n1+k+k,n1+k+k),dtype=np.float32)
    for i in range(k,n1+k):
        for j in range(k,n1+k):
            exp[i,j]=data[i-k,j-k]
    for i in range(n1):
        for j in range(n1):
            a = np.array([exp[i+1+k,j+k],exp[i-1+k,j+k],exp[i+k,j+1+k],exp[i+k,j-1+k]])
            d[i,j]=-np.std(a)
    return d

def save_m(arr1,arr2,name):
    n=len(arr1)
    A_map=np.zeros((n,n,3), dtype=np.uint8)
    arr1=Range(arr1)
    arr2=arr2.astype(bool)
    for i in range(n):
        for j in range(n):
            A_map[i][j][0]=np.uint8(arr1[i][j]*arr2[i][j]+255*( (arr2[i][j]+1)%2 ))
            A_map[i][j][1]=np.uint8(arr1[i][j]*arr2[i][j])
            A_map[i][j][2]=A_map[i][j][1]

    fmt='png'
    Pic = Image.fromarray(A_map, 'RGB')
    Pic.save('{}.{}'.format(name, fmt), fmt='png')
    Pic.close()

def CoefMap(k,p):
    n=2*k+1

    temp=np.zeros((n,n),dtype=np.float32)
    temp[k,k]=1
    for m in range(k):
        Map=np.zeros((n,n),dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if temp[i,j]:
                    Map[i-1,j]+=temp[i,j]
                    Map[i+1,j]+=temp[i,j]
                    Map[i,j+1]+=temp[i,j]
                    Map[i,j-1]+=temp[i,j]
                    Map[i,j]+=-4*p*temp[i,j]
        temp=Map
    return temp



@numba.jit
def Sign(x):
    if x<0:
        return 1
    else:
        return 0


def crossroad(i,j,pw):
    nc=i+j-pw+1
    res=0
    if nc:
        res=(nc*nc+nc)//2
    return res

def mSize(n,pw):
    npar=pw*pw+(pw+1)*(pw+1)

    res=(n-2*pw)*(n-2*pw)*npar

    for i in range(pw):
        res+=4*((npar-i*i-2*i-1)*(n-2*pw))

        for j in range(pw):
            res+=4*((npar-i*i-2*i-1)-j*j-2*j-1+crossroad(i,j,pw))

    return res



#@numba.jit
def getCoef(pos,k,CMap,pix,n):
    #C1
    ks=k+k+1
    n1=(ks)*(ks)-1-pos #обращаем нумерацию
    p=n1//(ks)  #получаем строку в карте
    q=n1%(ks) #получаем столбец
    #
    ps=p-k
    qs=q-k
    #sign = lambda x : 1 if (x < 0) else 0
    signp=Sign(ps) #sign(x): if  x>0: 0 else 1
    signq=Sign(qs)
    #
    zeromark=0
    summ=0
    up=(1-signp)*ps
    down=ks+signp*ps
    left=(1-signq)*qs
    right=ks+signq*qs

    count=0
    #/C1
    #A1
    for i in range(ks):
        for j in range(ks):
            summ+=CMap[i,j]

    for i in range(up,down):
        for j in range(left,right):
            if CMap[i,j]:
                count+=1
    #/A1
    row=np.zeros(count,dtype=np.int32)
    col=np.zeros(count,dtype=np.int32)
    dat=np.zeros(count,dtype=np.float32)
    count=0
    #A2
    for i in range(up,down):
        for j in range(left,right):
            if CMap[i,j]:
                zeromark+=CMap[i,j] #сумма захваченных коэфф
                row[count]=pix
                col[count]=pix+(i-k)*n+j-k
                dat[count]=CMap[i,j]
                if i==k and j==k:
                    mid=count
                count+=1
    #/A2

        # граничное условие
    dat[mid]+=summ-zeromark
    return dat,row,col



def CoefMatr(n, k, p=1):
    ms = k + k + 1
    m = n
    CMap = CoefMap(k, p)
    # a=getCoef(0,k,CMap,0,m)
    pix = 0
    size=mSize(n,k)
    data = DV(size)
    row = CV(size)
    col = CV(size)
    for i in range(k):
        for j in range(k):
            temp = getCoef(i * ms + j, k, CMap, pix, m)
            data.update(temp[0])
            row.update(temp[1])
            col.update(temp[2])
            pix += 1
        for j in range(n - 2 * k):
            temp = getCoef(i * ms + k, k, CMap, pix, m)
            data.update(temp[0])
            row.update(temp[1])
            col.update(temp[2])
            pix += 1
        for j in range(k):
            temp = getCoef(i * ms + k + 1 + j, k, CMap, pix, m)
            data.update(temp[0])
            row.update(temp[1])
            col.update(temp[2])
            pix += 1

    for i in range(n - 2 * k):
        for j in range(k):
            temp = getCoef(k * ms + j, k, CMap, pix, m)
            data.update(temp[0])
            row.update(temp[1])
            col.update(temp[2])
            pix += 1
        temp = getCoef(k * ms + k, k, CMap, pix, m)
        shift = pix
        for j in range(n - 2 * k):
            data.update(temp[0])
            row.shift(temp[1], pix - shift)
            col.shift(temp[2], pix - shift)
            pix += 1

        for j in range(k):
            temp = getCoef(k * ms + k + 1 + j, k, CMap, pix, m)
            data.update(temp[0])
            row.update(temp[1])
            col.update(temp[2])
            pix += 1

    for i in range(k):
        for j in range(k):
            temp = getCoef((k + 1 + i) * ms + j, k, CMap, pix, m)
            data.update(temp[0])
            row.update(temp[1])
            col.update(temp[2])
            pix += 1
        for j in range(n - 2 * k):
            temp = getCoef((k + 1 + i) * ms + k, k, CMap, pix, m)
            data.update(temp[0])
            row.update(temp[1])
            col.update(temp[2])
            pix += 1
        for j in range(k):
            temp = getCoef((k + 1 + i) * ms + k + 1 + j, k, CMap, pix, m)
            data.update(temp[0])
            row.update(temp[1])
            col.update(temp[2])
            pix += 1

    a = csr_matrix((data.finalize(), (row.finalize(), col.finalize())), shape=(m * m, m * m))
    return a



#@numba.jit
def Rev_Lap(Img, k, a=1):
    n = len(Img)
    print(Img.shape)
    VecImg = Img.reshape(n*n,)

    VecOriginImg = spsolve(CoefMatr(n, k, a), VecImg)

    OriginImg = VecOriginImg.reshape(n,n)

    return OriginImg

