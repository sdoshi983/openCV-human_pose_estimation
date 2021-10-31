print("Press C => convert Celsius To Fahrenheit")
print("Press F => convert Fahrenheit to Celsius")
x=(input("Enter Your Choise: "))

if (x=='c' or x=='C'):
    Celsius = float(input("Enter Celsius:"))
    Fahrenheit = (Celsius * 9/5) + 32
    print("Fahrenheit: ",Fahrenheit)
elif (x=='f' or x=='F'):
    Fahrenheit = float(input("Enter Fahrenheit: "))
    Celsius = (Fahrenheit - 32) * 5/9
    print("Celsius : ",Celsius)