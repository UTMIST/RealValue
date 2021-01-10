f = open("temp.txt", "r")
#for i in range(0,10):
    #print(f.readline())
counter=0

prev=f.readline()
temp=[]


num1=0
num2=0
sum=0

while (prev!=""):
    if (counter%2!=0):
        print(prev)
        print(prev.split())
        num1=0
        num2=0
        for i in prev.split():
            try:
                float(i)
                a=True
            except:
                a=False
            if a==True:
                if num1==0:
                    num1=float(i)
                elif num2==0:
                    num2=float(i)

        print(num1,"NUM1")
        print(num2,"NUM2")
        sum=sum+num1*num2

        temp+=[prev]
    prev=f.readline()
    counter+=1
    print("")
print(temp)
print('__________________________________')
print(sum, "= Square Metres (Assuming temp.txt has metric units)")
print(sum*10.764, "= IMPERIAL Square Feet (Assuming temp.txt has metric units)")
