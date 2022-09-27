#modern flu epidemic 
#S(t)=number in the population susceptible after time t
#I(t)=number infected after time t
#R(t)=number removed after time t

transmission_coefficient=a=0.7
removal_rate_per_week=c=0.6
b=0.002
n=1000
input=[2,5,10,100]
#solve system when: I(0)=2,5,10,100
#solve system when: I(0)=2,5,10,100 and S(0)=n-I(0)
def flu_simulation(x):
    R=0
    I=x
    S=n-I
    print("Initial position: R(0)=%s, I(0)=%s, S(0)=%s"%(R,I,S))
    for i in range(1,30):
        R=R+c*I*b
        A=S
        S=S+(-a*S*I)*b
        I=I+(-c*I+a*I*A)*b
        print("Week:%s, R=%s, S=%s, I=%s"%(i/2,R,S,I))

for i in input:
    print("Case: I(0)=",i)
    flu_simulation(i)















