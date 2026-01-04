print("Enter marks of subject 1: ")
sub1 = int(input())
print("Enter marks of subject 2: ")
sub2 = int(input())
print("Enter marks of subject 3: ")
sub3 = int(input())

perc = (sub1+sub2+sub3)/300 * 100



if(((sub1 | sub2 | sub3)<33) | (perc < 40)):
    print("Failed!!")
else:
    print("Passed!!")

print("You got " , perc , "%")