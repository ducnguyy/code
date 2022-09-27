#gradient method of steepest ascent - ex6
from re import I

def P(x,y):
    return 0.54*(x**2)-0.02*(x**3)+1.89*(y**2)-0.09*(y**3)
def dx(x,y):
#   numerical approx: return (P(x+lr,y)-P(x,y))/(lr)
    return (0.54*2*x-0.02*3*(x**2))
def dy(x,y):
#   numerical approx: return (P(x,y+lr)-P(x,y))/(lr)
    return (1.89*2*y-0.09*3*(y**2))
def maxrate(x,y):
#   ||Maximum rate of change of P|| 
    return ((dx(x,y)**2+dy(x,y)**2)**(1/2))

x0,y0=[1.3,5.6]
multiplier=1.2
lr=0.01

def grad(x,y,lr,multiplier):
    print("Initial Position: x0=%s, y0= %s, P= %s, ||DirP||= %s, lr=%s" %(x,y,P(x,y),maxrate(x,y),lr))
    for i in range(0,30):
        k=i
        DirX=dx(x,y)
        DirY=dy(x,y)
        x+=lr*DirX 
        y+=lr*DirY
        current=P(x,y)
        rate=maxrate(x,y)
        print("STEP: %s, xk= %s, yk= %s, P= %s, ||Maximum rate of change of P||= %s, Learning Rate= %s" %(k,x,y,current,rate,lr))
        lr=lr*multiplier

grad(x0,y0,lr,multiplier)

