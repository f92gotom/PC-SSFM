from numpy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.fftpack as fourier

#Parámetros:
beta2=5
beta3=0
alfa=0
gamma=0.05
P0=10
T0=50

L=80                                                                                                        #Longitud del mallado
m=1500                                                                                                      #Puntos del mallado (longitud)
mt=10000                                                                                                    #Puntos del mallado (tiempo)
n=5                                                                                                         #Órden del método (2,3,4,5)

T=linspace(-10*T0,10*T0,mt)                                                                                 #Mallado temporal
F=fourier.fftfreq(mt,T[1]-T[0])

#Pulsos:

u=sqrt(P0)*exp(-T**2/(2*T0**2))                                                                            #Pulso gaussiano

#u=sqrt(P0)*exp(-T**4/T0**4)                                                                                #Pulso supergaussiano

#u=sqrt(P0)*(1-(T/T0)**2)*exp(-T**2/(2*T0**2))                                                              #Pulso parabólico
#u[T>T0]=0
#u[T<-T0]=0

#u=2*sqrt(P0)/(exp(T/T0)+exp(-T/T0))                                                                        #Secante hiperbólica

#Método:
Lin=beta2*(2*pi*F)**2*1j/2-beta3*(2*pi*F)**3*1j/6-alfa/2                                                    #Operador L
h=L/m

if n==2:                                                                                                    #Coeficientes A y C según n
    A=array([3/2,-1/2])
    C=array([1/4,2/4,1/4])
elif n==3:
    A=array([23/12,-16/12,5/12])
    C=array([5/24, 13/24, 7/24, -1/24])
elif n==4:
    A=array([55/24,-59/24,37/24,-9/24])
    C=array([9/48,28/48,14/48,-4/48,1/48])
elif n==5:
    A=array([1901/720,-2774/720,2616/720,-1274/720,251/720])
    C=array([251/1440,897/1440,382/1440,-158/1440,87/1440,-19/1440])

N=zeros(shape=(n),dtype=object)
ug=zeros(shape=(m//10,mt),dtype=object)
fwhm=zeros(shape=(m//10),dtype=float)
Pmax=zeros(shape=(m//10),dtype=float)

for i in range(m):
    sumN=0
    for k in range(n-1):
        N[n-1-k]=N[n-2-k]
    N[0]=gamma*abs(u)**2*1j                                                                                 #Cálculo de N en j
    for k in range(n):
        sumN=sumN+A[k]*N[k]
    uesp=fourier.ifft(exp(Lin*h/2)*fourier.fft(exp(h*sumN)*fourier.ifft(exp(Lin*h/2)*fourier.fft(u))))      #Ecuación (28)
    Nsig=gamma*abs(uesp)**2*1j                                                                              #Cálculo de N esperado en j+1

    sumN=C[0]*Nsig
    for k in range(n):
        sumN=sumN+C[k+1]*N[k]
    u=fourier.ifft(exp(Lin*h/2)*fourier.fft(exp(h*sumN)*fourier.ifft(exp(Lin*h/2)*fourier.fft(u))))         #Ecuación (29)

    if i%10==0:
        v=abs(u)**2
        ug[i//10,:]=v                                                                                       #Guardado del valor esperado

        imax=argmax(v)                                                                                      #Cálculo del FWHM
        mitad_max=v[imax]/2

        indices_encima=where(v > mitad_max)[0]
        ind_izq=indices_encima[0]
        ind_der=indices_encima[-1]

        fwhm[i//10]=T[ind_der]-T[ind_izq]

        Pmax[i//10]=v[imax]                                                                                 #Potencia máxima

#Animación:
fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(0, 10))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = T/T0
    y = ug[i,:]
    line.set_data(x, y)
    return line,

plt.xlabel('T/To')
plt.ylabel('|u|^2 (mW)')
plt.title('')
anim = FuncAnimation(fig, animate, init_func=init,frames=m//10, interval=20, blit=False)
anim.save('animacion.gif')
plt.show()

#Gráfica FWHM:
dist=linspace(0,L,m//10)
plt.plot(dist,fwhm,'.-')
plt.xlabel('z (km)')
plt.ylabel('FWHM (ps)')
plt.title('FWHM a lo largo de z')
plt.show()

#Gráfica Pmax:
plt.plot(dist,Pmax,'.-')
plt.xlabel('z (km)')
plt.ylabel('Potencia máxima (mW)')
plt.title('Potencia máxima a lo largo de z')
plt.show()