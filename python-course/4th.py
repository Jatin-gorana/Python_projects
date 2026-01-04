# myDict = {
#     'gaadi': 'car',
#     'ghoda': 'horse',
#     'yeda': 'nonsense'
# }
# print("Options are: ", myDict.keys())
# a = input("Enter hindi word \n")
# print("The meaning of your hindi word is: ", myDict[a])    #this will throw a error if key is not present in dictionary
# print("The meaning of your hindi word is: ", myDict.get(a))

# num1 = int(input("Enter no 1 "))
# num2 = int(input("Enter no 2 "))
# num3 = int(input("Enter no 3 "))
# num4 = int(input("Enter no 4 "))
# num5 = int(input("Enter no 5 "))
# num6 = int(input("Enter no 6 "))
# num7 = int(input("Enter no 7 "))
# num8 = int(input("Enter no 8 "))

# s = {num1,num2,num3,num4,num5,num6,num7,num8}
# print(s)    #set shows only unique nos(which are not repeated)

# s = {20, "20", 20.0}
# print(s)
# print(len(s))

# s = {}
# print(type(s))

favlang = {}
a = input("Enter your fav language om\n")
b = input("Enter your fav language komal\n")
c = input("Enter your fav language shravan\n")
d = input("Enter your fav language sumit\n")

favlang['om'] = a
favlang['komal'] = b
favlang['shravan'] = c
favlang['sumit'] = d

print(favlang)